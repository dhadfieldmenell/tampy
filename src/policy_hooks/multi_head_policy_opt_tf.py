""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class MultiHeadPolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU, dObj, dTarg, dPrimObs):
        tf.reset_default_graph()
        
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.batch_size = self._hyperparams['batch_size']
        self.task_list = self._hyperparams['task_list'] if 'task_list' in self._hyperparams else [""]
        self._dPrim = len(self.task_list)
        self._dObj = dObj
        self._dTarg = dTarg
        self._dPrimObs = dPrimObs
        self.task_map = {}
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None
        self.init_network()
        self.init_solver()
        self.var = {task: self._hyperparams['init_var'] * np.ones(dU) for task in self.task_map}
        self.var[""] = self._hyperparams['init_var'] * np.ones(dU)
        self.distilled_var = self._hyperparams['init_var'] * np.ones(dU)

        self.gpu_fraction = self._hyperparams['gpu_fraction']
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.init_policies(dU)
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        with tf.variable_scope('primitive_filter'):
            tf_map_generator = self._hyperparams['primitive_network_model']
            tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, dim_output=self._dPrim+self._dObj+self._dTarg, batch_size=self.batch_size,
                                      network_config=self._hyperparams['network_params'])
            self.primitive_obs_tensor = tf_map.get_input_tensor()
            self.primitive_precision_tensor = tf_map.get_precision_tensor()
            self.primitive_action_tensor = tf_map.get_target_output_tensor()
            self.primitive_act_op = tf_map.get_output_op()
            self.primitive_feat_op = tf_map.get_feature_op()
            self.primitive_loss_scalar = tf_map.get_loss_op()
            self.primitive_fc_vars = fc_vars
            self.primitive_last_conv_vars = last_conv_vars

            # Setup the gradients
            self.primitive_grads = [tf.gradients(self.primitive_act_op[:,u], self.primitive_obs_tensor)[0] for u in range(self._dPrim+self._dObj+self._dTarg)]

        # with tf.variable_scope('parameterization_filter'):
        #     tf_map_generator = self._hyperparams['parameterization_network_model']
        #     tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dPar, batch_size=self.batch_size,
        #                               network_config=self._hyperparams['network_params'])
        #     self.parameterization_obs_tensor = tf_map.get_input_tensor()
        #     self.parameterization_precision_tensor = tf_map.get_precision_tensor()
        #     self.parameterization_action_tensor = tf_map.get_target_output_tensor()
        #     self.parameterization_act_op = tf_map.get_output_op()
        #     self.parameterization_feat_op = tf_map.get_feature_op()
        #     self.parameterization_loss_scalar = tf_map.get_loss_op()
        #     self.parameterization_fc_vars = fc_vars
        #     self.parameterization_last_conv_vars = last_conv_vars

        #     # Setup the gradients
        #     self.parameterization_grads = [tf.gradients(self.parameterization_act_op[:,u], self.parameterization_obs_tensor)[0] for u in range(self._dPrim+self._dObj+self._dTarg)]

        with tf.variable_scope('distilled'):
            tf_map_generator = self._hyperparams['distilled_network_model']
            tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                      network_config=self._hyperparams['distilled_network_params'])
            self.distilled_obs_tensor = tf_map.get_input_tensor()
            self.distilled_precision_tensor = tf_map.get_precision_tensor()
            self.distilled_action_tensor = tf_map.get_target_output_tensor()
            self.distilled_act_op = tf_map.get_output_op()
            self.distilled_feat_op = tf_map.get_feature_op()
            self.distilled_loss_scalar = tf_map.get_loss_op()
            self.distilled_fc_vars = fc_vars
            self.distilled_last_conv_vars = last_conv_vars

            # Setup the gradients
            self.distilled_grads = [tf.gradients(self.distilled_act_op[:,u], self.distilled_obs_tensor)[0] for u in range(self._dU)]

        for task in self.task_list:
            with tf.variable_scope(task):
                self.task_map[task] = {}
                tf_map_generator = self._hyperparams['network_model']
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                          network_config=self._hyperparams['network_params'])
                self.task_map[task]['obs_tensor'] = tf_map.get_input_tensor()
                self.task_map[task]['precision_tensor'] = tf_map.get_precision_tensor()
                self.task_map[task]['action_tensor'] = tf_map.get_target_output_tensor()
                self.task_map[task]['act_op'] = tf_map.get_output_op()
                self.task_map[task]['feat_op'] = tf_map.get_feature_op()
                self.task_map[task]['loss_scalar'] = tf_map.get_loss_op()
                self.task_map[task]['fc_vars'] = fc_vars
                self.task_map[task]['last_conv_vars'] = last_conv_vars

                # Setup the gradients
                self.task_map[task]['grads'] = [tf.gradients(self.task_map[task]['act_op'][:,u], self.task_map[task]['obs_tensor'])[0] for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='primitive_filter')
        self.primitive_solver = TfSolver(loss_scalar=self.primitive_loss_scalar,
                                       solver_name=self._hyperparams['solver_type'],
                                       base_lr=self._hyperparams['lr'],
                                       lr_policy=self._hyperparams['lr_policy'],
                                       momentum=self._hyperparams['momentum'],
                                       weight_decay=self._hyperparams['weight_decay'],
                                       fc_vars=self.primitive_fc_vars,
                                       last_conv_vars=self.primitive_last_conv_vars,
                                       vars_to_opt=vars_to_opt)

        vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='distilled')
        self.distilled_solver = TfSolver(loss_scalar=self.distilled_loss_scalar,
                                       solver_name=self._hyperparams['solver_type'],
                                       base_lr=self._hyperparams['lr'],
                                       lr_policy=self._hyperparams['lr_policy'],
                                       momentum=self._hyperparams['momentum'],
                                       weight_decay=self._hyperparams['weight_decay'],
                                       fc_vars=self.distilled_fc_vars,
                                       last_conv_vars=self.distilled_last_conv_vars,
                                       vars_to_opt=vars_to_opt)

        for task in self.task_list:
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=task)
            self.task_map[task]['solver'] = TfSolver(loss_scalar=self.task_map[task]['loss_scalar'],
                                                   solver_name=self._hyperparams['solver_type'],
                                                   base_lr=self._hyperparams['lr'],
                                                   lr_policy=self._hyperparams['lr_policy'],
                                                   momentum=self._hyperparams['momentum'],
                                                   weight_decay=self._hyperparams['weight_decay'],
                                                   fc_vars=self.task_map[task]['fc_vars'],
                                                   last_conv_vars=self.task_map[task]['last_conv_vars'],
                                                   vars_to_opt=vars_to_opt)
        self.saver = tf.train.Saver()

    def init_policies(self, dU):
        for task in self.task_list:
            self.task_map[task]['policy'] = TfPolicy(dU, 
                                                    self.task_map[task]['obs_tensor'], 
                                                    self.task_map[task]['act_op'], 
                                                    self.task_map[task]['feat_op'],
                                                    np.zeros(dU), 
                                                    self.sess, 
                                                    self.device_string, 
                                                    copy_param_scope=None)
        self.distilled_policy = TfPolicy(dU,
                                         self.distilled_obs_tensor,
                                         self.distilled_act_op,
                                         self.distilled_feat_op,
                                         np.zeros(dU),
                                         self.sess,
                                         self.device_string,
                                         copy_param_scope=None)

    def task_distr(self, obs):
        distr = self.sess.run(self.primitive_act_op, feed_dict={self.primitive_obs_tensor:obs}).flatten()
        return distr[:self._dPrim], distr[self._dPrim:self._dPrim+self._dObj], distr[self._dPrim+self._dObj:self._dPrim+self._dObj+self._dTarg]

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt, task=""):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        # mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        mn = np.median(tgt_wt[(tgt_wt > 1e-4).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        policy = self.task_map[task]['policy']
        if policy.scale is None or policy.bias is None:
            policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            policy.bias = - np.mean(
                obs[:, self.x_idx].dot(policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(policy.scale) + policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.maximum(np.floor(N*T / self.batch_size), 1)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.task_map[task]['solver'].get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.task_map[task]['last_conv_vars']: conv_values[idx_i],
                             self.task_map[task]['action_tensor']: tgt_mu[idx_i],
                             self.task_map[task]['precision_tensor']: tgt_prc[idx_i]}
                train_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                    i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        print "\nEntering Tensorflow Training Loop"
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.task_map[task]['obs_tensor']: obs[idx_i],
                         self.task_map[task]['action_tensor']: tgt_mu[idx_i],
                         self.task_map[task]['precision_tensor']: tgt_prc[idx_i]}
            train_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0
        print "Leaving Tensorflow Training Loop\n"

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.task_map[task]['feat_op'] is not None:
            self.task_map[task]['feat_vals'] = self.task_map[task]['solver'].get_var_values(self.sess, self.task_map[task]['feat_op'], feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var[task] = 1 / np.diag(A)
        policy.chol_pol_covar = np.diag(np.sqrt(self.var[task]))

        return policy

    def update_primitive_filter(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean filter outputs, N x T x dP.
            tgt_prc: Numpy array of precision matrices, N x T x dP x dP.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N = obs.shape[0]
        dP, dO = self._dPrim+self._dObj+self._dTarg, self._dPrimObs

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N, dP, dP])

        # Renormalize weights.
        tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, dO))
        tgt_mu = np.reshape(tgt_mu, (N, dP))
        tgt_prc = np.reshape(tgt_prc, (N, dP, dP))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # Assuming that N*T >= self.batch_size.
        batch_size = np.minimum(self.batch_size, N)
        batches_per_epoch = np.maximum(np.floor(N / batch_size), 1)
        idx = range(N)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.primitive_solver.get_last_conv_values(self.sess, feed_dict, num_values, batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * batch_size %
                                (batches_per_epoch * batch_size))
                idx_i = idx[start_idx:start_idx+batch_size]
                feed_dict = {self.primitive_last_conv_vars: conv_values[idx_i],
                             self.primitive_action_tensor: tgt_mu[idx_i],
                             self.primitive_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                    i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.primitive_obs_tensor: obs[idx_i],
                         self.primitive_action_tensor: tgt_mu[idx_i],
                         self.primitive_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.primitive_feat_op is not None:
            self.primitive_feat_vals = self.primitive_solver.get_var_values(self.sess, self.primitive_feat_op, feed_dict, num_values, self.batch_size)


    def update_distilled(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, _ = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, dO))
        tgt_mu = np.reshape(tgt_mu, (N, dU))
        tgt_prc = np.reshape(tgt_prc, (N, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.distilled_policy.scale is None or self.distilled_policy.bias is None:
            self.distilled_policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.distilled_policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.distilled_policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.distilled_policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.distilled_policy.scale) + self.distilled_policy.bias

        # Assuming that N >= self.batch_size.
        batches_per_epoch = np.maximum(np.floor(N / self.batch_size), 1)
        idx = range(N)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.distilled_obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.distilled_solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.distilled_last_conv_vars: conv_values[idx_i],
                             self.distilled_action_tensor: tgt_mu[idx_i],
                             self.distilled_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.distilled_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                    i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.distilled_obs_tensor: obs[idx_i],
                         self.distilled_action_tensor: tgt_mu[idx_i],
                         self.distilled_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.distilled_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.distilled_feat_op is not None:
            self.distilled_feat_vals = self.solver.get_var_values(self.sess, self.distilled_feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.distilled_var = 1 / np.diag(A)
        self.distilled_policy.chol_pol_covar = np.diag(np.sqrt(self.distilled_var))

        return self.distilled_policy

    def prob(self, obs, task=""):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        policy = self.task_map[task]['policy']
        if policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(policy.scale)
                                         + policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.task_map[task]['obs_tensor']: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.task_map[task]['act_op'], feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var[task]), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var[task]), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var[task]), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': {task:self.task_map[task]['policy'].scale for task in self.task_list},
            'bias': {task:self.task_map[task]['policy'].bias for task in self.task_list},
            'tf_iter': self.tf_iter,
            'x_idx': {task:self.task_map[task]['policy'].x_idx for task in self.task_list},
            'chol_pol_covar': {task:self.task_map[task]['policy'].chol_pol_covar for task in self.task_list},
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        for task in self.task_list:
            self.policy[task].scale = state['scale']
            self.policy[task].bias = state['bias']
            self.policy[task].x_idx = state['x_idx']
            self.policy[task].chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)

