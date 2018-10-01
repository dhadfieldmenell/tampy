""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np
import tensorflow as tf

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class MultiHeadPolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU, dObj, dTarg, dPrimObs):
        import tensorflow as tf
        self.scope = hyperparams['scope']
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
        self.weight_dir = self._hyperparams['weight_dir']
        self.scope = self._hyperparams['scope'] if 'scope' in self._hyperparams else None 

        self.gpu_fraction = self._hyperparams['gpu_fraction']
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        if self.scope is not None:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            self.saver = tf.train.Saver(variables)
            try:
                self.saver.restore(self.policy_opt.sess, 'tf_saved/'+self.weight_dir+'/'+self.scope+'.ckpt')
            except Exception as e:
                print 'Could not load previous weights.'

        else:
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.policy_opt.sess, 'tf_saved/'+self.weight_dir+'.ckpt')
            except Exception as e:
                print 'Could not load previous weights.'
        

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
        self.prim_x_idx, self.prim_img_idx, i = [], [], 0
        for sensor in self._hyperparams['network_params']['prim_obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.prim_img_idx = self.prim_img_idx + list(range(i, i+dim))
            else:
                self.prim_x_idx = self.prim_x_idx + list(range(i, i+dim))
            i += dim

        self.update_count = 0
        self.update_size = self._hyperparams['update_size']
        self.mu = {}
        self.obs = {}
        self.prc = {}
        self.wt = {}


    def update_weights(self, scope):
        self.saver.restore(self.session, 'tf_saved/'+self.weight_dir+'/'+scope+'.ckpt')

    def store_weights(self, scopes):
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            saver = tf.train.Saver(variables)
            saver.save(self.session, 'tf_saved/'+self.weight_dir+'/'+scope+'.ckpt')

    def store_all_weights(self):
        for task in self.task_list + ['value', 'primitive']:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=task)
            saver = tf.train.Saver(variables)
            saver.save(self.policy_opt.sess, 'tf_saved/'+self.weight_dir+'/'+task+'.ckpt')

    def store_weights(self):
        assert self.scope is not None
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        saver = tf.train.Saver(variables)
        saver.save(self.session, 'tf_saved/'+self.weight_dir+'/'+self.scope+'.ckpt')

    def store(self, mu, obs, prc, wt, task):
        if task not in self.mu:
            self.mu[task] = np.array(mu)
            self.obs[task] = np.array(obs)
            self.prc[task] = np.array(prc)
            self.wt[task] = np.array(wt)
        else:
            self.mu[task] = np.concatenate([self.mu[task], np.array(mu)])
            self.obs[task] = np.concatenate([self.obs[task], np.array(obs)])
            self.prc[task] = np.concatenate([self.prc[task], np.array(prc)])
            self.wt[task] = np.concatenate([self.wt[task], np.array(wt)])

        self.update_count += len(mu)
        if self.update_count > self.update_size:
            self.update(self.mu[task], self.obs[task], self.prc[task], self.wt[task], task)
            self.store_weights(scopes=[task])
            self.update_count = 0
            del self.mu[task]
            del self.obs[task]
            del self.prc[task]
            del self.wt[task]

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        if self.scope is None or 'primitive' == self.scope:
            with tf.variable_scope('primitive'):
                tf_map_generator = self._hyperparams['primitive_network_model']
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, dim_output=self._dPrim+self._dObj+self._dTarg, batch_size=self.batch_size,
                                          network_config=self._hyperparams['primitive_network_params'])
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

        if self.scope is None or 'value' == self.scope:
            with tf.variable_scope('value'):
                tf_map_generator = self._hyperparams['value_network_model']
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=1, batch_size=self.batch_size,
                                          network_config=self._hyperparams['network_params'])
                self.value_obs_tensor = tf_map.get_input_tensor()
                self.value_precision_tensor = tf_map.get_precision_tensor()
                self.value_action_tensor = tf_map.get_target_output_tensor()
                self.value_act_op = tf_map.get_output_op()
                self.value_feat_op = tf_map.get_feature_op()
                self.value_loss_scalar = tf_map.get_loss_op()
                self.value_fc_vars = fc_vars
                self.value_last_conv_vars = last_conv_vars

                # Setup the gradients
                self.value_grads = [tf.gradients(self.value_act_op[:,u], self.value_obs_tensor)[0] for u in range(1)]

        # with tf.variable_scope('distilled'):
        #     tf_map_generator = self._hyperparams['distilled_network_model']
        #     tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, dim_output=self._dU, batch_size=self.batch_size,
        #                               network_config=self._hyperparams['distilled_network_params'])
        #     self.distilled_obs_tensor = tf_map.get_input_tensor()
        #     self.distilled_precision_tensor = tf_map.get_precision_tensor()
        #     self.distilled_action_tensor = tf_map.get_target_output_tensor()
        #     self.distilled_act_op = tf_map.get_output_op()
        #     self.distilled_feat_op = tf_map.get_feature_op()
        #     self.distilled_loss_scalar = tf_map.get_loss_op()
        #     self.distilled_fc_vars = fc_vars
        #     self.distilled_last_conv_vars = last_conv_vars

        #     # Setup the gradients
        #     self.distilled_grads = [tf.gradients(self.distilled_act_op[:,u], self.distilled_obs_tensor)[0] for u in range(self._dU)]

        for task in self.task_list:
            if self.scope is None or task == self.scope:
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
        if self.scope is None or 'primitive' == self.scope:
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='primitive_filter')
            self.primitive_solver = TfSolver(loss_scalar=self.primitive_loss_scalar,
                                               solver_name=self._hyperparams['solver_type'],
                                               base_lr=self._hyperparams['lr'],
                                               lr_policy=self._hyperparams['lr_policy'],
                                               momentum=self._hyperparams['momentum'],
                                               weight_decay=0.,
                                               fc_vars=self.primitive_fc_vars,
                                               last_conv_vars=self.primitive_last_conv_vars,
                                               vars_to_opt=vars_to_opt)

        if self.scope is None or 'value' == self.scope:
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='primitive_filter')
            self.value_solver = TfSolver(loss_scalar=self.value_loss_scalar,
                                           solver_name=self._hyperparams['solver_type'],
                                           base_lr=self._hyperparams['lr'],
                                           lr_policy=self._hyperparams['lr_policy'],
                                           momentum=self._hyperparams['momentum'],
                                           weight_decay=0.,
                                           fc_vars=self.value_fc_vars,
                                           last_conv_vars=self.value_last_conv_vars,
                                           vars_to_opt=vars_to_opt)

        # vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='distilled')
        # self.distilled_solver = TfSolver(loss_scalar=self.distilled_loss_scalar,
        #                                    solver_name=self._hyperparams['solver_type'],
        #                                    base_lr=self._hyperparams['lr'],
        #                                    lr_policy=self._hyperparams['lr_policy'],
        #                                    momentum=self._hyperparams['momentum'],
        #                                    weight_decay=self._hyperparams['weight_decay'],
        #                                    fc_vars=self.distilled_fc_vars,
        #                                    last_conv_vars=self.distilled_last_conv_vars,
        #                                    vars_to_opt=vars_to_opt)

        for task in self.task_list:
            if self.scope is None or task == self.scope:
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

    def init_policies(self, dU):
        for task in self.task_list:
            if self.scope is None or task == self.scope:
                self.task_map[task]['policy'] = TfPolicy(dU, 
                                                        self.task_map[task]['obs_tensor'], 
                                                        self.task_map[task]['act_op'], 
                                                        self.task_map[task]['feat_op'],
                                                        np.zeros(dU), 
                                                        self.sess, 
                                                        self.device_string, 
                                                        copy_param_scope=None)
        # self.distilled_policy = TfPolicy(dU,
        #                                  self.distilled_obs_tensor,
        #                                  self.distilled_act_op,
        #                                  self.distilled_feat_op,
        #                                  np.zeros(dU),
        #                                  self.sess,
        #                                  self.device_string,
        #                                  copy_param_scope=None)

    def task_distr(self, obs):
        distr = self.sess.run(self.primitive_act_op, feed_dict={self.primitive_obs_tensor:obs}).flatten()
        return distr[:self._dPrim], distr[self._dPrim:self._dPrim+self._dObj], distr[self._dPrim+self._dObj:self._dPrim+self._dObj+self._dTarg]

    def value(self, obs):
        value = self.sess.run(self.value_act_op, feed_dict={self.value_obs_tensor:obs}).flatten()
        return value.flatten()

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
        if task == 'primitive':
            return self.update_primitive_filter(obs, tgt_mu, tgt_prc, tgt_wt)
        if task == 'value':
            return self.update_value(obs, tgt_mu, tgt_prc, tgt_wt)
        if np.any(np.isnan(tgt_mu)) or np.any(np.abs(tgt_mu) == np.inf):
            import ipdb; ipdb.set_trace()
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        if np.sum(tgt_wt) == 0 or np.any(np.isnan(tgt_wt)):
            import ipdb; ipdb.set_trace()
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
            # policy.scale = np.diag(
            #     1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-1))
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
        # print "\nEntering Tensorflow Training Loop"
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
        # print "Leaving Tensorflow Training Loop\n"

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
        # print 'Updating primitive network...'
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
            #     LOGGER.info('tensorflow iteration %d, average loss %f',
            #                  i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.primitive_feat_op is not None:
            self.primitive_feat_vals = self.primitive_solver.get_var_values(self.sess, self.primitive_feat_op, feed_dict, num_values, self.batch_size)
        # print 'Updated primitive network.\n'


    def update_value(self, obs, tgt_mu, tgt_prc, tgt_wt):
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
        # print 'Updating value network...'
        N = obs.shape[0]
        dP, dO = 2, self._dO

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
            conv_values = self.value_solver.get_last_conv_values(self.sess, feed_dict, num_values, batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * batch_size %
                                (batches_per_epoch * batch_size))
                idx_i = idx[start_idx:start_idx+batch_size]
                feed_dict = {self.value_last_conv_vars: conv_values[idx_i],
                             self.value_action_tensor: tgt_mu[idx_i],
                             self.value_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.value_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
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
            feed_dict = {self.value_obs_tensor: obs[idx_i],
                         self.value_action_tensor: tgt_mu[idx_i],
                         self.value_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.value_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                # LOGGER.info('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.value_feat_op is not None:
            self.value_feat_vals = self.value_solver.get_var_values(self.sess, self.value_feat_op, feed_dict, num_values, self.batch_size)
        # print 'Updated value network.'


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
        dU, dO = self._dU, self._dPrimObs

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
            self.distilled_policy.x_idx = self.prim_x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.distilled_policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.prim_x_idx], axis=0), 1e-3))
            self.distilled_policy.bias = - np.mean(
                obs[:, self.prim_x_idx].dot(self.distilled_policy.scale), axis=0)
        obs[:, self.prim_x_idx] = obs[:, self.prim_x_idx].dot(self.distilled_policy.scale) + self.distilled_policy.bias

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
                # LOGGER.info('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
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
        if task in self.task_map:
            policy = self.task_map[task]['policy']
        else:
            policy = getattr(self, '{0}_policy'.format(task))
        if policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(policy.scale)
                                         + policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                if task in self.task_map:
                    obs_tensor = self.task_map[task]['obs_tensor']
                    act_op = self.task_map[task]['act_op']
                else:
                    obs_tensor = getattr(self, '{0}_obs_tensor'.format(task))
                    act_op = getattr(self, '{0}_act_op'.format(task))
                feed_dict = {obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(act_op, feed_dict=feed_dict)

        if task in self.var:
            pol_sigma = np.tile(np.diag(self.var[task]), [N, T, 1, 1])
            pol_prec = np.tile(np.diag(1.0 / self.var[task]), [N, T, 1, 1])
            pol_det_sigma = np.tile(np.prod(self.var[task]), [N, T])
        else:
            var = getattr(self, '{0}_var'.format(task))
            pol_sigma = np.tile(np.diag(var), [N, T, 1, 1])
            pol_prec = np.tile(np.diag(1.0 / var), [N, T, 1, 1])
            pol_det_sigma = np.tile(np.prod(var), [N, T])


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

