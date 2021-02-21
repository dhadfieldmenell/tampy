""" This file defines policy optimization for a tensorflow policy. """
import copy
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np
#import tensorflow as tf

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
#from gps.algorithm.policy.tf_policy import TfPolicy
from policy_hooks.utils.tf_utils import TfSolver

from policy_hooks.tf_policy import TfPolicy

MAX_QUEUE_SIZE = 200000
MAX_UPDATE_SIZE = 10000
SCOPE_LIST = ['primitive']


class ControlAttentionPolicyOpt(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU, dPrimObs, dValObs, primBounds, inputs=None):
        global tf
        import tensorflow as tf
        self.scope = hyperparams['scope'] if 'scope' in hyperparams else None
        # tf.reset_default_graph()

        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        self.split_nets = hyperparams.get('split_nets', False)
        self.valid_scopes = ['control'] if not self.split_nets else list(config['task_list'])

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.batch_size = self._hyperparams['batch_size']
        self.load_all = self._hyperparams.get('load_all', False)

        self.input_layer = inputs
        self.share_buffers = self._hyperparams.get('share_buffer', True)
        if self._hyperparams.get('share_buffer', True):
            self.buffers = self._hyperparams['buffers']
            self.buf_sizes = self._hyperparams['buffer_sizes']
        auxBounds = self._hyperparams.get('aux_boundaries', [])
        self._dPrim = max([b[1] for b in primBounds] + [b[1] for b in auxBounds])# primBounds[-1][-1]
        self._dPrimObs = dPrimObs
        self._dValObs = dValObs
        self._primBounds = primBounds
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
        self.last_pkl_t = time.time()
        self.cur_pkl = 0

        self.gpu_fraction = self._hyperparams['gpu_fraction']
        if not self._hyperparams['allow_growth']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        self.init_policies(dU)
        llpol = hyperparams.get('ll_policy', '')
        hlpol = hyperparams.get('hl_policy', '')
        scopes = self.valid_scopes + SCOPE_LIST if self.scope is None else [self.scope]
        for scope in scopes:
            if len(llpol) and scope in self.valid_scopes:
                self.restore_ckpt(scope, dirname=llpol)
            if len(hlpol) and scope not in self.valid_scopes:
                self.restore_ckpt(scope, dirname=hlpol)

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
        for sensor in self._hyperparams['primitive_network_params']['obs_include']:
            dim = self._hyperparams['primitive_network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['primitive_network_params']['obs_image_data']:
                self.prim_img_idx = self.prim_img_idx + list(range(i, i+dim))
            else:
                self.prim_x_idx = self.prim_x_idx + list(range(i, i+dim))
            i += dim

        self.update_count = 0
        if self.scope == 'primitive':
            self.update_size = self._hyperparams['prim_update_size']
        else:
            self.update_size = self._hyperparams['update_size']

        self.update_size *= (1 + self._hyperparams.get('permute_hl', 0))

        self.train_iters = 0
        self.average_losses = []
        self.average_val_losses = []
        self.average_error = []
        self.N = 0
        self.n_updates = 0
        self.buffer_size = MAX_QUEUE_SIZE
        self.lr_scale = 0.9975
        self.lr_policy = 'fixed'
        self._hyperparams['iterations'] = MAX_UPDATE_SIZE // self.batch_size + 1


    def restore_ckpts(self, label=None):
        success = False
        for scope in self.valid_scopes + SCOPE_LIST:
            success = success or self.restore_ckpt(scope, label)
        return success


    def restore_ckpt(self, scope, label=None, dirname=''):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        if not len(variables): return False
        self.saver = tf.train.Saver(variables)
        ext = ''
        if label is not None:
            ext = '_{0}'.format(label)
        success = True
        if not len(dirname):
            dirname = self.weight_dir
        try:
            self.saver.restore(self.sess, 'tf_saved/'+dirname+'/'+scope+'{0}.ckpt'.format(ext))
            if scope in self.task_map:
                self.task_map[scope]['policy'].scale = np.load('tf_saved/'+dirname+'/'+scope+'_scale{0}.npy'.format(ext))
                self.task_map[scope]['policy'].bias = np.load('tf_saved/'+dirname+'/'+scope+'_bias{0}.npy'.format(ext))
                #self.var[scope] = np.load('tf_saved/'+dirname+'/'+scope+'_variance{0}.npy'.format(ext))
                #self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(self.var[scope]))
            self.write_shared_weights([scope])
            print(('Restored', scope, 'from', dirname))
        except Exception as e:
            print(('Could not restore', scope, 'from', dirname))
            print(e)
            success = False

        return success


    def write_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

        for scope in scopes:
            wts = self.serialize_weights([scope])
            with self.buf_sizes[scope].get_lock():
                self.buf_sizes[scope].value = len(wts)
                self.buffers[scope][:len(wts)] = wts


    def read_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

        for scope in scopes:
            start_t = time.time()
            skip = False
            with self.buf_sizes[scope].get_lock():
                if self.buf_sizes[scope].value == 0: skip = True
                wts = self.buffers[scope][:self.buf_sizes[scope].value]

            wait_t = time.time() - start_t
            if wait_t > 0.1 and scope == 'primitive': print('Time waiting on lock:', wait_t)
            #if self.buf_sizes[scope].value == 0: skip = True
            #wts = self.buffers[scope][:self.buf_sizes[scope].value]

            #if skip: continue
            try:
                self.deserialize_weights(wts)
            except Exception as e:
                #traceback.print_exception(*sys.exc_info())
                if not skip:
                    print('Could not load {0} weights'.format(scope))


    def serialize_weights(self, scopes=None, save=True):
        if scopes is None:
            scopes = self.valid_scopes + SCOPE_LIST

        var_to_val = {}
        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            for v in variables:
                var_to_val[v.name] = self.sess.run(v).tolist()

        scales = {task: self.task_map[task]['policy'].scale.tolist() for task in scopes if task in self.task_map}
        biases = {task: self.task_map[task]['policy'].bias.tolist() for task in scopes if task in self.task_map}
        #variances = {task: self.var[task].tolist() for task in scopes if task in self.task_map}
        variances = {}
        scales[''] = []
        biases[''] = []
        variances[''] = []
        if save: self.store_scope_weights(scopes=scopes)
        return pickle.dumps([scopes, var_to_val, scales, biases, variances])


    def deserialize_weights(self, json_wts, save=False):
        scopes, var_to_val, scales, biases, variances = pickle.loads(json_wts)

        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            for var in variables:
                var.load(var_to_val[var.name], session=self.sess)

            if scope not in self.valid_scopes: continue
            # if save:
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_scale', scales['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_bias', biases['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_variance', variances['control'])
            #self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(np.array(variances[scope])))
            self.task_map[scope]['policy'].scale = np.array(scales[scope])
            self.task_map[scope]['policy'].bias = np.array(biases[scope])
            #self.var[scope] = np.array(variances[scope])
        if save: self.store_scope_weights(scopes=scopes)

    def update_weights(self, scope, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        self.saver.restore(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')

    def store_scope_weights(self, scopes, weight_dir=None, lab=''):
        if weight_dir is None:
            weight_dir = self.weight_dir
        for scope in scopes:
            try:
                variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                saver = tf.train.Saver(variables)
                saver.save(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'{0}.ckpt'.format(lab))
            except:
                print('Saving variables encountered an issue but it will not crash:')
                traceback.print_exception(*sys.exc_info())

        if scope in self.task_map:
            policy = self.task_map[scope]['policy']
            np.save('tf_saved/'+weight_dir+'/'+scope+'_scale{0}'.format(lab), policy.scale)
            np.save('tf_saved/'+weight_dir+'/'+scope+'_bias{0}'.format(lab), policy.bias)
            #np.save('tf_saved/'+weight_dir+'/'+scope+'_variance{0}'.format(lab), self.var[scope])

    def store_weights(self, weight_dir=None):
        if self.scope is None:
            self.store_scope_weights(self.valid_scopes+SCOPE_LIST, weight_dir)
        else:
            self.store_scope_weights([self.scope], weight_dir)

    def get_data(self):
        return [self.mu, self.obs, self.prc, self.wt, self.val_mu, self.val_obs, self.val_prc, self.val_wt]


    def update_lr(self):
        if self.method == 'linear':
            self.cur_lr *= self.lr_scale
            self.cur_hllr *= self.lr_scale


    def _create_network(self, name, info):
        with tf.variable_scope(name):
            self.etas[name] = tf.placeholder_with_default(1., shape=())
            tf_map_generator = info['network_model']
            info['network_params']['eta'] = self.etas[name]
            #self.class_tensors[name] = tf.placeholder(shape=[None, 1], dtype='float32')
            tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=info['dO'], \
                                                               dim_output=info['dOut'], \
                                                               batch_size=info['batch_size'], \
                                                               network_config=info['network_params'], \
                                                               input_layer=info['input_layer'])

            self.obs_tensors[name] = tf_map.get_input_tensor()
            self.precision_tensors[name] = tf_map.get_precision_tensor()
            self.action_tensors[name] = tf_map.get_target_output_tensor()
            self.act_ops[name] = tf_map.get_output_op()
            self.feat_ops[name] = tf_map.get_feature_op()
            self.loss_scalars[name] = tf_map.get_loss_op()
            self.fc_vars[name] = fc_vars
            self.last_conv_vars[name] = last_conv_vars


    def init_network(self):
        """ Helper method to initialize the tf networks used """

        input_tensor = None
        if self.load_all or self.scope is None or 'primitive' == self.scope:
            with tf.variable_scope('primitive'):
                inputs = self.input_layer if 'primitive' == self.scope else None
                self.primitive_eta = tf.placeholder_with_default(1., shape=())
                tf_map_generator = self._hyperparams['primitive_network_model']
                self.primitive_class_tensor = None
                if self._hyperparams['split_hl_loss']:
                    self.primitive_class_tensor = tf.placeholder(shape=[None, 1], dtype='float32')
                    tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, \
                                                                       dim_output=self._dPrim, \
                                                                       batch_size=self.batch_size, \
                                                                       network_config=self._hyperparams['primitive_network_params'], \
                                                                       input_layer=inputs, \
                                                                       eta=self.primitive_eta, \
                                                                       class_tensor=self.primitive_class_tensor)
                else:
                    tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, \
                                                                       dim_output=self._dPrim, \
                                                                       batch_size=self.batch_size, \
                                                                       network_config=self._hyperparams['primitive_network_params'], \
                                                                       input_layer=inputs, \
                                                                       eta=self.primitive_eta)
                self.primitive_obs_tensor = tf_map.get_input_tensor()
                self.primitive_precision_tensor = tf_map.get_precision_tensor()
                self.primitive_action_tensor = tf_map.get_target_output_tensor()
                self.primitive_act_op = tf_map.get_output_op()
                self.primitive_feat_op = tf_map.get_feature_op()
                self.primitive_loss_scalar = tf_map.get_loss_op()
                self.primitive_fc_vars = fc_vars
                self.primitive_last_conv_vars = last_conv_vars
                self.primitive_aux_losses = tf_map.aux_loss_ops

                # Setup the gradients
                #self.primitive_grads = [tf.gradients(self.primitive_act_op[:,u], self.primitive_obs_tensor)[0] for u in range(self._dPrim)]

        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                with tf.variable_scope(scope):
                    self.task_map[scope] = {}
                    tf_map_generator = self._hyperparams['network_model']
                    tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, \
                                                                       dim_output=self._dU, \
                                                                       batch_size=self.batch_size, \
                                                                       network_config=self._hyperparams['network_params'], \
                                                                       input_layer=self.input_layer)
                    self.task_map[scope]['obs_tensor'] = tf_map.get_input_tensor()
                    self.task_map[scope]['precision_tensor'] = tf_map.get_precision_tensor()
                    self.task_map[scope]['action_tensor'] = tf_map.get_target_output_tensor()
                    self.task_map[scope]['act_op'] = tf_map.get_output_op()
                    self.task_map[scope]['feat_op'] = tf_map.get_feature_op()
                    self.task_map[scope]['loss_scalar'] = tf_map.get_loss_op()
                    self.task_map[scope]['fc_vars'] = fc_vars
                    self.task_map[scope]['last_conv_vars'] = last_conv_vars

                    # Setup the gradients
                    #self.task_map[scope]['grads'] = [tf.gradients(self.task_map[scope]['act_op'][:,u], self.task_map[scope]['obs_tensor'])[0] for u in range(self._dU)]


    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.dec_tensor = tf.placeholder('float', name='weight_dec')#tf.Variable(initial_value=self._hyperparams['prim_weight_decay'], name='weightdec')
        if self.scope is None or 'primitive' == self.scope:
            self.cur_hllr = self._hyperparams['hllr']
            self.hllr_tensor = tf.Variable(initial_value=self._hyperparams['hllr'], name='hllr')
            self.cur_dec = self._hyperparams['prim_weight_decay']
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='primitive')
            self.primitive_solver = TfSolver(loss_scalar=self.primitive_loss_scalar,
                                               solver_name=self._hyperparams['solver_type'],
                                               base_lr=self.hllr_tensor,
                                               lr_policy=self._hyperparams['lr_policy'],
                                               momentum=self._hyperparams['momentum'],
                                               weight_decay=self.dec_tensor,#self._hyperparams['prim_weight_decay'],
                                               #weight_decay=self._hyperparams['prim_weight_decay'],
                                               fc_vars=self.primitive_fc_vars,
                                               last_conv_vars=self.primitive_last_conv_vars,
                                               vars_to_opt=vars_to_opt,
                                               aux_losses=self.primitive_aux_losses)
        self.lr_tensor = tf.Variable(initial_value=self._hyperparams['lr'], name='lr')
        self.cur_lr = self._hyperparams['lr']
        for scope in self.valid_scopes:
            self.cur_dec = self._hyperparams['weight_decay']
            if self.scope is None or scope == self.scope:
                vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                self.task_map[scope]['solver'] = TfSolver(loss_scalar=self.task_map[scope]['loss_scalar'],
                                                       solver_name=self._hyperparams['solver_type'],
                                                       base_lr=self.lr_tensor,
                                                       lr_policy=self._hyperparams['lr_policy'],
                                                       momentum=self._hyperparams['momentum'],
                                                       #weight_decay=self.dec_tensor,#self._hyperparams['weight_decay'],
                                                       weight_decay=self._hyperparams['weight_decay'],
                                                       fc_vars=self.task_map[scope]['fc_vars'],
                                                       last_conv_vars=self.task_map[scope]['last_conv_vars'],
                                                       vars_to_opt=vars_to_opt)

    def get_policy(self, task):
        if task == 'primitive': return self.prim_policy
        return self.task_map[task]['policy']

    def init_policies(self, dU):
        if self.load_all or self.scope is None or self.scope == 'primitive':
            self.prim_policy = TfPolicy(dU,
                                        self.primitive_obs_tensor,
                                        self.primitive_act_op,
                                        self.primitive_feat_op,
                                        np.zeros(dU),
                                        self.sess,
                                        self.device_string,
                                        copy_param_scope=None,
                                        normalize=False)
        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                self.task_map[scope]['policy'] = TfPolicy(dU,
                                                        self.task_map[scope]['obs_tensor'],
                                                        self.task_map[scope]['act_op'],
                                                        self.task_map[scope]['feat_op'],
                                                        np.zeros(dU),
                                                        self.sess,
                                                        self.device_string,
                                                        copy_param_scope=None)
    
    def task_acc(self, obs, tgt_mu, prc, piecewise=False, scalar=True):
        acc = []
        task = 'primitive'
        for n in range(len(obs)):
            distrs = self.task_distr(obs[n])
            labels = []
            for bound in self._primBounds:
                labels.append(tgt_mu[n, bound[0]:bound[1]])
            accs = []
            for i in range(len(labels)):
                #if prc[n][i] < 1e-3 or np.abs(np.max(labels[i])-np.min(labels[i])) < 1e-2:
                #    accs.append(1)
                #    continue

                if np.argmax(distrs[i]) != np.argmax(labels[i]):
                    accs.append(0)
                else:
                    accs.append(1)

            if piecewise or not scalar:
                acc.append(accs)
            else:
                acc.append(np.min(accs) * np.ones(len(accs)))
            #acc += np.mean(accs) if piecewise else np.min(accs)
        if scalar:
            return np.mean(acc)
        return np.mean(acc, axis=0)


    def task_distr(self, obs, eta=1.):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)

        distr = self.sess.run(self.primitive_act_op, feed_dict={self.primitive_obs_tensor:obs, self.primitive_eta: eta, self.dec_tensor: self.cur_dec})[0].flatten()
        res = []
        for bound in self._primBounds:
            res.append(distr[bound[0]:bound[1]])
        return res


    def check_task_error(self, obs, mu):
        err = 0.
        for o in obs:
            distrs = self.task_distr(o)
            i = 0
            for d in distrs:
                ind1 = np.argmax(d)
                ind2 = np.argmax(mu[i:i+len(d)])
                if ind1 != ind2: err += 1./len(distrs)
                i += len(d)
        err /= len(obs)
        self.average_error.append(err)
        return err


    def check_validation(self, obs, tgt_mu, tgt_prc, task="control"):
        if task == 'primitive':
            feed_dict = {self.primitive_obs_tensor: obs,
                         self.primitive_action_tensor: tgt_mu,
                         self.primitive_precision_tensor: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, train=False)
        else:
            feed_dict = {self.task_map[task]['obs_tensor']: obs,
                         self.task_map[task]['action_tensor']: tgt_mu,
                         self.task_map[task]['precision_tensor']: tgt_prc,
                         self.dec_tensor: self.cur_dec}
            val_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string, train=False)
        #self.average_val_losses.append(val_loss)
        return val_loss


    def update(self, task="control", check_val=False, aux=[]):
        start_t = time.time()
        average_loss = 0
        for i in range(self._hyperparams['iterations']):
            feed_dict = {self.hllr_tensor: self.cur_hllr} if task == 'primitive' else {self.lr_tensor: self.cur_lr}
            feed_dict[self.dec_tensor] = self.cur_dec
            solver = self.primitive_solver if task == 'primitive' else self.task_map[task]['solver']
            train_loss = solver(feed_dict, self.sess, device_string=self.device_string, train=True)[0]
            average_loss += train_loss
        self.tf_iter += self._hyperparams['iterations']
        self.average_losses.append(average_loss / self._hyperparams['iterations'])
        #if task == 'primitive': print('Time to run', self._hyperparams['iterations'], 'updates:', time.time() - start_t)

        '''
        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * NT * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)
        self.var[task] = 1 / np.diag(A)
        policy.chol_pol_covar = np.diag(np.sqrt(self.var[task]))
        '''


    def update_primitive_filter(self, obs, tgt_mu, tgt_prc, tgt_wt, check_val=False, aux=[]):
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

        #tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        # mn = np.median(tgt_wt[(np.abs(tgt_wt) > 1e-3).nonzero()])
        # for n in range(N):
        #     tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        # tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, -1))
        tgt_mu = np.reshape(tgt_mu, (N, -1))

        '''
        tgt_prc = np.reshape(tgt_prc, (N, dP, dP))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc
        '''
        tgt_prc = tgt_prc * tgt_wt.reshape((N, 1)) #tgt_wt.flatten()
        if len(aux): aux = aux.reshape((-1,1))

        # Assuming that N*T >= self.batch_size.
        batch_size = np.minimum(self.batch_size, N)
        batches_per_epoch = np.maximum(np.floor(N / batch_size), 1)
        idx = list(range(N))
        average_loss = 0
        np.random.shuffle(idx)

        '''
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
                             self.primitive_precision_tensor: tgt_prc[idx_i],
                             self.hllr_tensor: self.cur_hllr}
                train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, train=(not check_val), use_fc_solver=True)
            average_loss = 0
        '''

        # actual training.
        # for i in range(self._hyperparams['iterations']):
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            self.train_iters += 1
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.primitive_obs_tensor: obs[idx_i],
                         self.primitive_action_tensor: tgt_mu[idx_i],
                         self.primitive_precision_tensor: tgt_prc[idx_i],
                         self.hllr_tensor: self.cur_hllr}
            if len(aux) and self.primitive_class_tensor is not None:
                feed_dict[self.primitive_class_tensor] = aux[idx_i]
            train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, train=(not check_val))[0]

            average_loss += train_loss

        self.tf_iter += self._hyperparams['iterations']
        if check_val:
            self.average_val_losses.append(average_loss / self._hyperparams['iterations'])
        else:
            self.average_losses.append(average_loss / self._hyperparams['iterations'])
        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        #if self.primitive_feat_op is not None:
        #    self.primitive_feat_vals = self.primitive_solver.get_var_values(self.sess, self.primitive_feat_op, feed_dict, num_values, self.batch_size)


    def traj_prob(self, obs, task="control"):
        assert len(obs.shape) == 2 or obs.shape[0] == 1
        mu, sig, prec, det_sig = self.prob(obs, task)
        traj = np.tri(mu.shape[1]).dot(mu[0])
        return np.array([traj]), sig, prec, det_sig

    def policy_initialized(self, task):
        if task in self.valid_scopes:
            return self.task_map[task]['policy'].scale is not None
        return self.task_map['control']['policy'].scale is not None

    def prob(self, obs, task="control"):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        if len(obs.shape) < 3:
            obs = obs.reshape((1, obs.shape[0], obs.shape[1]))
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if task not in self.valid_scopes:
            task = "control"
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
                # with tf.device(self.device_string):
                #     output[i, t, :] = self.sess.run(act_op, feed_dict=feed_dict)
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
        # LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        # LOGGER.debug('Restoring model from: %s', fname)

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
            'scale': {task:self.task_map[task]['policy'].scale for task in self.task_map},
            'bias': {task:self.task_map[task]['policy'].bias for task in self.task_map},
            'tf_iter': self.tf_iter,
            'x_idx': {task:self.task_map[task]['policy'].x_idx for task in self.task_map},
            'chol_pol_covar': {task:self.task_map[task]['policy'].chol_pol_covar for task in self.task_map},
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        for task in self.task_map:
            self.policy[task].scale = state['scale']
            self.policy[task].bias = state['bias']
            self.policy[task].x_idx = state['x_idx']
            self.policy[task].chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)
