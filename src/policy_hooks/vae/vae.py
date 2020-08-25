import copy
import json
import logging
import os
import sys
import tempfile
import time
import traceback

import h5py
import numpy as np
import tables
import tensorflow as tf

from policy_hooks.vae.vae_networks import *


'''
Random things to remember:
 - End with no-op task (since we go obs + task -> next_obs, we want last obs + task -> last obs for code simplicity)
 - Or cut last timestep?
 - Policy gets a reward for finding bad encode/decode paths?
 - Constrain conditional encoding (i.e. latent output) against prior?
'''

LATENT_DIM = 16

ENCODER_CONFIG = {
    'n_channels': [16, 32, 32],
    'filter_sizes': [5, 5, 5],
    'strides': [3, 3, 3],
    'fc_dims': [LATENT_DIM] # [2 * 3 * 32]
    # 'out_act': 'tanh',
}

DECODER_CONFIG = {
    'conv_init_shape': [2, 3, 32],
    'n_channels': [32, 16, 3],
    'filter_sizes': [5, 5, 5],
    'strides': [3, 3, 3],
    'fc_dims': None,
    'out_act': 'sigmoid',
}

LATENT_DYNAMICS_CONFIG = {
    'fc_dims': [LATENT_DIM, LATENT_DIM],
}

class VAE(object):
    def __init__(self, hyperparams):
        self.config = hyperparams
        tf.reset_default_graph()
        tf.set_random_seed(self.config.get('random_seed', 1234))

        self.tf_iter = 0
        self.batch_size = self.config.get('batch_size', 64)
        self.train_iters = self.config.get('train_iters', 100)
        self.T = self.config['rollout_len'] - 2
        self.rollout_len = self.config['rollout_len'] - 2

        self.obs_dims = [80, 107, 3] # list(hyperparams['obs_dims'])
        self.task_dim = hyperparams['task_dims']

        # The following hyperparameters also describe where the weights are saved
        self.weight_dir = hyperparams['weight_dir']

        # if self.load_step < 0:
        #     is_rnn = 'rnn' if self.use_recurrent_dynamics else 'fc'
        #     overshoot = 'overshoot' if self.use_overshooting else 'onestep'
        #     self.ckpt_name = self.weight_dir+'/vae_{0}_{1}_{2}.ckpt'.format(self.train_mode, is_rnn, overshoot)
        # else:
        #     self.ckpt_name = self.weight_dir+'/vae_{0}_{1}_{2}.ckpt'.format(self.train_mode, is_rnn, overshoot, load_step)

        if hyperparams.get('load_data', True):
            f_mode = 'a'
            self.data_file = self.weight_dir+'/vae_buffer.hdf5'
            self.data = h5py.File(self.data_file, f_mode)

            try:
                self.obs_data = self.data['obs_data']
                self.task_data = self.data['task_data']
                self.task_data = self.task_data[:, :, :self.task_dim]
                self.task_dim = self.task_data.shape[-1]
            except:
                obs_data = np.zeros([0, self.rollout_len]+list(self.obs_dims))
                task_data = np.zeros((0, self.rollout_len, self.task_dim))
                self.obs_data = self.data.create_dataset('obs_data', data=obs_data, maxshape=(None, None, None, None, None), dtype='uint8')
                self.task_data = self.data.create_dataset('task_data', data=task_data, maxshape=(None, None, None), dtype='uint8')

            # self.data.swmr_mode=True

        elif hyperparams.get('data_read_only', False):
            f_mode = 'r'
            self.data_file = self.weight_dir+'/vae_buffer.hdf5'
            self.data = h5py.File(self.data_file, f_mode, swmr=True)
            self.obs_data = self.data['obs_data']
            self.task_data = self.data['task_data']
            # while not os.path.isfile(self.weight_dir+'/vae_buffer.hdf5'):
            #     time.sleep(1)

        self.train_mode = hyperparams.get('train_mode', 'online')
        assert self.train_mode in ['online', 'conditional', 'unconditional']
        self.use_recurrent_dynamics = hyperparams.get('use_recurrent_dynamics', False)
        self.use_overshooting = hyperparams.get('use_overshooting', False)
        self.use_prior = hyperparams.get('use_prior', True)
        self.load_step = hyperparams.get('load_step', 0)
        # self.beta = hyperparams.get('beta', 10)
        # self.beta_d = hyperparams.get('overshoot_beta', 1./self.T)
        self.beta = 0.2 # hyperparams.get('beta', 0.5)
        self.beta_d = hyperparams.get('overshoot_beta', 0.1)
        self.data_limit = hyperparams.get('data_limit', None)
        self.data_limit = self.data_limit if self.data_limit is not None else len(self.obs_data)
        self.obs_data = self.obs_data[:self.data_limit]
        self.task_data = self.task_data[:self.data_limit]
        self.dist_constraint = hyperparams.get('dist_constraint', False)
        self.ckpt_name = self.get_weight_file()

        # self.data_file = self.weight_dir+'/vae_buffer.npz'
        # try:
        #     data = np.load(self.data_file, mmap_mode='w+')
        # except:
        #     pass

        # self.obs_data = np.zeros((0, self.dT, self.dO))
        # self.task_data = np.zeros((0, self.dT, self.dU))
        self.max_buffer = hyperparams.get('max_buffer', 1e6)
        self.dist_constraint = hyperparams.get('distance_constraint', False)

        self.cur_lr = 1e-3
        with tf.variable_scope('vae', reuse=False):
            self.init_network()
            self.init_solver()

        self.scope = 'vae'
        self.gpu_fraction = self.config['gpu_fraction'] if 'gpu_fraction' in self.config else 0.95
        if 'allow_growth' in self.config and not self.config['allow_growth']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        init_op = tf.initialize_all_variables()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        self.saver = tf.train.Saver(variables)
        if self.use_recurrent_dynamics:
            zero_state = self.latent_dynamics.lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.zero_state = tuple(self.sess.run(zero_state))

        try:
            self.saver.restore(self.sess, self.ckpt_name)
        except Exception as e:
            self.sess.run(init_op)
            print(('\n\nCould not load previous weights for {0} from {1}\n\n'.format(self.scope, self.weight_dir)))

        self.update_count = 0
        self.n_updates = 0
        self.update_size = self.config.get('update_size', 1)


    def get_weight_file(self, addendum=None):
        is_rnn = 'rnn' if self.use_recurrent_dynamics else 'fc'
        overshoot = 'overshoot' if self.use_overshooting else 'onestep'
        step = self.load_step
        mode = self.train_mode
        prior = 'prior' if self.use_prior else 'noprior'
        beta = 'beta'+str(self.beta)
        overshoot_beta = 'beta_d'+str(self.beta_d)
        limit = self.data_limit if self.data_limit is not None else len(self.obs_data)
        limit = str(limit)+'nsamples'
        dist = 'distconstr' if self.dist_constraint else 'nodistconstr'

        if addendum is None:
            ext = "vae_{0}_{1}_{2}_{3}_{4}_{5}_{6}.ckpt".format(mode, is_rnn, overshoot, prior, beta, dist, limit)
        else:
            ext = "vae_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.ckpt".format(mode, is_rnn, overshoot, prior, beta, dist, limit, addendum)
        file_name = self.weight_dir + ext
        return file_name


    def serialize_weights(self):
        print('Serializing vae weights')
        var_to_val = {}
        variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')
        for v in variables:
            var_to_val[v.name] = self.sess.run(v).tolist()

        return json.dumps(var_to_val)


    def deserialize_weights(self, json_wts, save=True):
        var_to_val = json.loads(json_wts)

        # print 'Deserializing', scopes
        variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')
        for var in variables:
            var.load(var_to_val[var.name], session=self.sess)

        if save: self.store_scope_weights()
        # print 'Weights for {0} successfully deserialized and stored.'.format(scopes)


    # def update_weights(self, weight_dir=None):
    #     if weight_dir is None:
    #         weight_dir = self.weight_dir
    #     self.saver.restore(self.sess, weight_dir+'/vae_{0}.ckpt'.format(self.train_mode))


    def store_scope_weights(self, weight_dir=None, addendum=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        try:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')
            saver = tf.train.Saver(variables)
            saver.save(self.sess, self.get_weight_file(addendum))
            print(('Saved vae weights for', self.train_mode, 'in', self.weight_dir))
        except:
            print('Saving variables encountered an issue but it will not crash:')
            traceback.print_exception(*sys.exc_info())


    def store_weights(self, weight_dir=None):
        self.store_scope_weights(weight_dir)


    def store(self, obs, task_list):
        print(('Storing data for', self.scope))
        assert len(obs) == len(task_list)
        # self.T = len(obs)

        # self.obs_data = np.r_[self.obs_data, obs]
        # self.task_data = np.r_[self.task_data, task_list]
        # obs = obs[:self.T]
        # task_list = task_list[:self.T]

        obs = obs.reshape((1,)+obs.shape)
        task_list = task_list.reshape((1,)+task_list.shape)

        self.obs_data.resize((len(self.obs_data)+1,) + obs.shape[1:])
        self.obs_data[-1] = obs.astype(np.uint8)

        self.task_data.resize((len(self.task_data)+1,) + task_list.shape[1:])
        self.task_data[-1] = task_list.astype(np.uint8)

        # if len(self.obs_data) > self.max_buffer:
        #     self.obs_data = self.obs_data[-self.max_buffer:]
        #     self.task_data = self.task_data[-self.max_buffer:]

        self.update_count += 1
        if self.update_count > self.update_size and len(self.obs_data) > 10:
            print('Updating vae')
            # self.update()
            self.n_updates += 1
            self.update_count = 0

        if not self.n_updates % 5:
            self.save_buffers()

        if self.n_updates > 10:
            self.store_scope_weights()
            self.n_updates = 0
            return True

        return False


    def save_buffers(self):
        # np.savez(self.data_file, task_data=self.task_data, obs_data=self.obs_data)
        self.data.flush()


    def init_network(self):
        import tensorflow as tf
        self.x_in = tf.placeholder(tf.float32, shape=[self.batch_size*self.T]+list(self.obs_dims))
        self.latent_in = tf.placeholder(tf.float32, shape=[1, 1, LATENT_DIM])
        self.task_in = tf.placeholder(tf.float32, shape=[self.batch_size*self.T]+[self.task_dim])
        self.latent_task_in = tf.placeholder(tf.float32, shape=[1, 1, self.task_dim])
        self.offset_in = tf.placeholder(tf.float32, shape=[self.batch_size*self.T]+list(self.obs_dims))
        self.before_offset_in = tf.placeholder(tf.float32, shape=[self.batch_size*self.T]+list(self.obs_dims))
        self.training = tf.placeholder(tf.bool)

        if len(self.obs_dims) == 1:
            pass
        else:
            pass

        self.fc_in = None # tf.placeholder(tf.float32, shape=[None, self.task_dim])
        self.offset_fc_in = None #tf.placeholder(tf.float32, shape=[None, self.task_dim])
        self.far_offset_fc_in = None # tf.placeholder(tf.float32, shape=[None, self.task_dim])

        # mask = tf.ones((self.batch_size, self.T))
        # mask[:,-1] = 0
        # self.far_offset_loss_mask = tf.constant(mask.reshape([self.batch_size*self.T]))

        self.encoder = Encoder()
        self.encode_mu, self.encode_logvar = self.encoder.get_net(self.x_in / 255., self.training, fc_in=self.fc_in, config=ENCODER_CONFIG)
        self.encode_posterior = tf.distributions.Normal(self.encode_mu, tf.sqrt(tf.exp(self.encode_logvar)))

        # self.offset_encode_mu, self.offset_encode_logvar = self.encoder.get_net(self.offset_in, self.training, fc_in=self.offset_fc_in, reuse=True, config=ENCODER_CONFIG)
        # self.far_offset_encode_mu, self.far_offset_encode_logvar = self.encoder.get_net(self.far_offset_in, self.training, fc_in=self.far_offset_fc_in, reuse=True, config=ENCODER_CONFIG)

        self.decoder_in = self.encode_mu +  tf.sqrt(tf.exp(self.encode_logvar)) * tf.random_normal(tf.shape(self.encode_mu), 0, 1)
        self.decoder = Decoder()
        self.decode_mu, self.decode_logvar = self.decoder.get_net(self.decoder_in, self.training, config=DECODER_CONFIG)
        self.decode_posterior = tf.distributions.Normal(self.decode_mu, tf.sqrt(tf.exp(self.decode_logvar)))

        # self.sample_decode_mu, self.sample_decode_logvar = self.decoder.get_net(self.decoder_in, self.training, config=DECODER_CONFIG, reuse=reuse)
        # self.sample_decode_posterior = tf.distributions.Normal(self.sample_decode_mu, tf.sqrt(tf.exp(self.sample_decode_logvar)))

        if 'unconditional' not in self.train_mode:
            if self.use_recurrent_dynamics:
                self.latent_dynamics = RecurrentLatentDynamics()
                in_shape = tf.shape(self.decoder_in)
                z_in = tf.reshape(self.decoder_in, (self.batch_size, self.T, LATENT_DIM))
                task_in = tf.reshape(self.task_in, (self.batch_size, self.T, self.task_dim))

                mu, logvar, self.rnn_initial_state, self.rnn_final_state = self.latent_dynamics.get_net(z_in, task_in, self.T, self.training, config=LATENT_DYNAMICS_CONFIG)
                self.conditional_encode_mu = tf.reshape(mu, in_shape)
                self.conditional_encode_logvar = tf.reshape(logvar, in_shape)
                self.conditional_encode_posterior = tf.distributions.Normal(self.conditional_encode_mu, tf.sqrt(tf.exp(self.conditional_encode_logvar)))

                trans_mu, trans_logvar, self.trans_rnn_initial_state, self.trans_rnn_final_state = self.latent_dynamics.get_net(self.latent_in, self.latent_task_in, 1, self.training, config=LATENT_DYNAMICS_CONFIG, reuse=True)
                self.latent_trans_mu = tf.reshape(trans_mu, [1, 1, LATENT_DIM])
                self.latent_trans_logvar = tf.reshape(trans_logvar, [1, 1, LATENT_DIM])
                self.latent_trans_posterior = tf.distributions.Normal(self.latent_trans_mu, tf.sqrt(tf.exp(self.latent_trans_logvar)))


            else:
                self.latent_dynamics = LatentDynamics()

                self.conditional_encode_mu, self.conditional_encode_logvar = self.latent_dynamics.get_net(self.decoder_in, self.task_in, self.training, config=LATENT_DYNAMICS_CONFIG)
                self.conditional_encode_posterior = tf.distributions.Normal(self.conditional_encode_mu, tf.sqrt(tf.exp(self.conditional_encode_logvar)))

                self.latent_trans_mu, self.latent_trans_logvar = self.latent_dynamics.get_net(tf.reshape(self.latent_in, (1, LATENT_DIM)), tf.reshape(self.latent_task_in, (1, self.task_dim)), self.training, config=LATENT_DYNAMICS_CONFIG, reuse=True)
                self.latent_trans_posterior = tf.distributions.Normal(self.latent_trans_mu, tf.sqrt(tf.exp(self.latent_trans_logvar)))

            self.conditional_decoder_in = self.conditional_encode_mu +  tf.sqrt(tf.exp(self.conditional_encode_logvar)) * tf.random_normal(tf.shape(self.conditional_encode_mu), 0, 1)
            self.conditional_decode_mu, self.conditional_decode_logvar = self.decoder.get_net(self.conditional_decoder_in, self.training, config=DECODER_CONFIG, reuse=True)
            self.conditional_decode_posterior = tf.distributions.Normal(self.conditional_decode_mu, tf.sqrt(tf.exp(self.conditional_decode_logvar)))

            self.offset_encode_mu, self.offset_encode_logvar = self.encoder.get_net(self.offset_in / 255., self.training, fc_in=self.offset_fc_in, config=ENCODER_CONFIG, reuse=True)
            self.offset_encode_posterior = tf.distributions.Normal(self.offset_encode_mu, tf.sqrt(tf.exp(self.offset_encode_logvar)))

            if self.dist_constraint:
                self.before_offset_encode_mu, self.before_offset_encode_logvar = self.Encoder.get_net(self.before_offset_in/255., self.training, fc_in=self.fc_in, config=ENCODER_CONFIG, reuse=True)
                self.before_offset_encode_posterior = tf.distributions.Normal(self.before_offset_encode_mu, tf.sqrt(tf.exp(self.before_offset_encode_logvar)))

        self.latent_prior = tf.distributions.Normal(tf.zeros_initializer()(tf.shape(self.encode_mu)), 1.)
        self.fitted_prior = tf.distributions.Normal(tf.zeros_initializer()(LATENT_DIM), 1.)


    def overshoot_latents(self, d=-1):
        if d < 0:
            d = self.T

        if self.use_recurrent_dynamics:
            latent_in = tf.reshape(self.decoder_in, [self.batch_size, self.T, LATENT_DIM])
            task_in = tf.reshape(self.task_in, [self.batch_size, self.T, self.task_dim])
            z_in = tf.concat([latent_in, task_in], axis=-1)
            latent_mu = tf.reshape(self.conditional_encode_mu, [self.batch_size, self.T, LATENT_DIM])
            latent_logvar= tf.reshape(self.conditional_encode_logvar, [self.batch_size, self.T, LATENT_DIM])
            cell = self.latent_dynamics.lstm_cell
            w = self.latent_dynamics.weights
            b = self.latent_dynamics.bias
            init_state = self.latent_dynamics.initial_state
            last_state = self.latent_dynamics.last_state
            zero_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            outs = {i: [] for i in range(self.T)}
            cur_state = zero_state
            for i in range(self.T):
                cur_out = z_in[:, i, :]
                for j in range(i+1, np.minimum(self.T, i+d+1)):
                    cur_out, cur_state = cell(cur_out, cur_state)
                    if j == i+1:
                        next_state = cur_state
                    cur_out = tf.nn.bias_add(tf.matmul(cur_out, w), b)
                    outs[j].append(cur_out)
                    cur_out = tf.split(cur_out, 2, -1)[0]
                    cur_out = tf.concat([cur_out, task_in[:, j, :]], axis=-1)
                cur_state = next_state
        else:
            latent_in = tf.reshape(self.decoder_in, [self.batch_size, self.T, LATENT_DIM])
            task_in = tf.reshape(self.task_in, [self.batch_size, self.T, self.task_dim])
            z_in = tf.concat([latent_in, task_in], axis=-1)
            latent_mu = tf.reshape(self.conditional_encode_mu, [self.batch_size, self.T, LATENT_DIM])
            latent_logvar= tf.reshape(self.conditional_encode_logvar, [self.batch_size, self.T, LATENT_DIM])
            outs = {i: [] for i in range(self.T)}
            for i in range(self.T):
                cur_out = z_in[:, i, :]
                for j in range(i+1, self.T):
                    cur_out = self.latent_dynamics.apply(cur_out)
                    outs[j].append(cur_out)
                    cur_out = tf.split(cur_out, 2, -1)[0]
                    cur_out = tf.concat([cur_out, task_in[:, j, :]], axis=-1)

        return outs


    def init_solver(self):
        import tensorflow as tf
        beta = self.beta
        beta_d = self.beta_d
        # self.decoder_loss = -tf.reduce_sum(tf.log(ecode_posterior.prob(self.x_in)+1e-6), axis=tuple(range(1, len(self.decode_mu.shape))))
        self.decoder_loss = tf.reduce_sum(((self.x_in / 255.) - self.decode_mu)**2)#, axis=tuple(range(1, len(self.decode_mu.shape))))
        self.loss = self.decoder_loss
        if self.use_prior:
            self.kl_loss = beta*tf.reduce_sum(tf.distributions.kl_divergence(self.encode_posterior, self.latent_prior))#, axis=tuple(range(1, len(self.encode_mu.shape))))
            self.loss += self.kl_loss
        # self.elbo = self.decoder_loss + beta * self.kl_loss
        # self.loss = self.elbo

        if 'unconditional' not in self.train_mode:
            # self.conditional_decoder_loss = -tf.reduce_sum(tf.log(conditional_decode_posterior.prob(self.offset_in)+1e-6))#, axis=tuple(range(1, len(self.conditional_decode_mu.shape))))
            self.conditional_decoder_loss = tf.reduce_sum((self.offset_in / 255. - self.conditional_decode_mu)**2)#, axis=tuple(range(1, len(self.conditional_decode_mu.shape))))
            self.loss += self.conditional_decoder_loss
            if self.use_prior:
                self.conditional_kl_loss = beta*tf.reduce_sum(tf.distributions.kl_divergence(self.conditional_encode_posterior, self.latent_prior))#, axis=tuple(range(1, len(self.conditional_encode_mu.shape))))
                self.loss += self.conditional_kl_loss
            # self.conditional_elbo = self.conditional_decoder_loss + beta * self.conditional_kl_loss

            self.conditional_prediction_loss = tf.reduce_sum(tf.distributions.kl_divergence(self.conditional_encode_posterior, self.offset_encode_posterior))#, axis=tuple(range(1, len(self.conditional_encode_mu.shape))))
            self.loss += self.conditional_prediction_loss

            if self.dist_constraint:
                self.near_loss = 0.1*tf.reduce_sum(tf.distributions.kl_divergence(self.encode_posterior, self.offset_encode_posterior))#, axis=tuple(range(1, len(self.far_encode_mu.shape))))
                self.dist_loss = -0.1*tf.reduce_sum(tf.distributions.kl_divergence(self.offset_encode_posterior, self.before_offset_encode_posterior))#, axis=tuple(range(1, len(self.far_encode_mu.shape))))
                self.loss += self.dist_loss + self.near_loss

            if self.use_overshooting:
                outs = self.overshoot_latents(5)
                for t in range(1, self.T):
                    true_mu, true_logvar = self.offset_encode_mu[t*self.batch_size:(t+1)*self.batch_size], self.offset_encode_logvar[t*self.batch_size:(t+1)*self.batch_size]
                    true_mu = tf.stop_gradient(true_mu)
                    true_logvar = tf.stop_gradient(true_logvar)
                    prior = tf.distributions.Normal(true_mu, tf.sqrt(tf.exp(true_logvar)))
                    for out in outs[t]:
                        mu, logvar = tf.split(out, 2, axis=-1)
                        posterior = tf.distributions.Normal(mu, tf.sqrt(tf.exp(logvar)))
                        self.loss += 1./(self.T) * beta_d * tf.reduce_sum(tf.distributions.kl_divergence(posterior, prior))#, axis=tuple(range(1, len(self.conditional_encode_mu.shape))))
                        # self.loss += 1./(self.T) * beta_d * tf.reduce_sum(tf.distributions.kl_divergence(posterior, self.latent_prior))#, axis=tuple(range(1, len(self.conditional_encode_mu.shape))))

        self.loss = self.loss / (self.batch_size * self.T)
        # if self.dist_constraint:
        #     offset_loss = tf.reduce_sum((self.encode_mu-self.offset_encode_mu)**2 axis=tuple(range(1, len(self.encode_mu.shape))))
        #     self.loss += offset_loss

        #     far_offset_loss = -tf.reduce_sum((self.encode_mu-self.far_offset_encode_mu)**2 axis=tuple(range(1, len(self.encode_mu.shape))))
        #     self.loss += self.far_offset_loss_mask * far_offset_loss

        self.lr = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(self.lr)
        # sess.run(tf.variables_initializer(self.opt.variables()))
        train_op = self.opt.minimize(self.loss)
        # opt_grad_vars = self.opt.compute_gradients(self.loss)
        # clip_grad = [(tf.clip_by_norm(grad, 1), var) for grad, var in opt_grad_vars if grad is not None]
        # train_op = self.opt.apply_gradients(clip_grad)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([train_op, update_ops])


    def update(self):
        for step in range(self.train_iters):
            # start_t = time.time()
            ind = np.random.choice(list(range(len(self.obs_data)-self.batch_size)), 1)[0]
            # print 'ind:', time.time() - start_t
            obs_batch = self.obs_data[ind:ind+self.batch_size]
            task_batch = self.task_data[ind:ind+self.batch_size]
            # print 'data:', time.time() - start_t
            obs = obs_batch[:, 1:self.T+1]
            next_obs = obs_batch[:, 2:self.T+2]
            before_obs = obs_batch[:, :self.T]
            task_path = task_batch[:, :self.T]
            # obs = np.concatenate([obs_batch[:, :self.T], np.zeros([self.batch_size, 1]+list(self.obs_dims))], axis=1)
            # next_obs = np.concatenate([ obs_batch[:, 1:self.T+1], np.zeros([self.batch_size, 1]+list(self.obs_dims))], axis=1)
            # far_obs = np.concatenate([obs_batch[:, 2:self.T+2], np.zeros([self.batch_size, 2]+list(self.obs_dims))], axis=1)
            # task_path = np.concatenate([task_batch[:, :self.T], -1*np.ones([self.batch_size, 1, self.task_dim])], axis=1)
            obs = obs.reshape([self.batch_size*self.T]+self.obs_dims)
            next_obs = next_obs.reshape([self.batch_size*self.T]+self.obs_dims)
            before_obs = before_obs.reshape([self.batch_size*self.T]+self.obs_dims)
            task_path = task_path.reshape([self.batch_size*self.T, self.task_dim])

            # inds = np.random.choice(range(len(self.obs_data)), self.batch_size)

            # obs = []
            # next_obs = []
            # task_path = []
            # for i in inds:
            #     print i
            #     next_obs_batch = np.array([self.obs_data[i] for i in inds])[0]
            #     next_task_batch = np.array([self.task_data[i] for i in inds])[0]

            #     obs1 = next_obs_batch[:self.T-1].reshape([self.T-1]+list(self.obs_dims))
            #     obs.append(np.concatenate([obs1, np.zeros([1]+list(self.obs_dims))], 0))
            #     obs2 = next_obs_batch[1:self.T].reshape([self.T-1]+list(self.obs_dims))
            #     next_obs.append(np.concatenate([np.zeros([1]+list(self.obs_dims)), obs2], 0))
            #     task = next_task_batch[:self.T-1].reshape([self.T-1, self.task_dim])
            #     task_path.append(np.concatenate([task, -1*np.ones([1, self.task_dim])], 0))
            # print 'start:', time.time() - start_t

            self.sess.run(self.train_op, feed_dict={self.x_in: obs,
                                                    self.offset_in: next_obs,
                                                    self.before_offset_in: before_obs,
                                                    self.task_in: task_path,
                                                    self.training: True,
                                                    self.lr: self.cur_lr,})
            # print 'train:', time.time() - start_t
            # print step

            # inds = np.random.choice(range(len(self.task_data)), 1)#self.batch_size)
            # next_obs_batch = np.array([self.obs_data[i] for i in inds])[0]
            # next_task_batch = np.array([self.task_data[i] for i in inds])[0]

            # obs1 = next_obs_batch[:self.T-1].reshape([-1]+list(self.obs_dims))
            # obs2 = next_obs_batch[1:self.T].reshape([-1]+list(self.obs_dims))
            # task = next_task_batch[:self.T-1].reshape([-1, self.task_dim])

            # self.sess.run(self.train_op, feed_dict={self.x_in: obs1,
            #                                         self.offset_in: obs2,
            #                                         self.task_in: task,
            #                                         self.lr: self.cur_lr,
            #                                         self.training: True})
            self.cur_lr *= 0.99999
        self.load_step += self.train_iters
        print(('Updated VAE', self.load_step))


    def fit_prior(self):
        latents = []
        inds = np.random.choice(list(range(len(self.obs_data))), np.minimum(1000, len(self.obs_data)))
        for i in range(len(inds)):
            print(i)
            batch = self.obs_data[inds[i]]
            latents.extend(self.get_latents(batch))
        self.prior_mean = np.mean(latents, axis=0)
        self.prior_std = np.std(latents, axis=0)
        self.fitted_prior = tf.distributions.Normal(self.prior_mean, self.prior_std)


    def sample_prior(self):
        return self.sess.run(self.fitted_prior.sample())


    def check_loss(self):
        ind = np.random.choice(list(range(len(self.obs_data)-self.batch_size)), 1)[0]
        obs_batch = self.obs_data[ind:ind+self.batch_size]
        task_batch = self.task_data[ind:ind+self.batch_size]
        before_obs = obs_batch[:, :self.T]
        obs = obs_batch[:, 1:self.T+1]
        next_obs = obs_batch[:, 2:self.T+2]
        task_path = task_batch[:, :self.T]
        # obs = np.concatenate([obs_batch[:, :self.T-1], np.zeros([self.batch_size, 1]+list(self.obs_dims))], axis=1)
        # next_obs = np.concatenate([np.zeros([self.batch_size, 1]+list(self.obs_dims)), obs_batch[:, 1:self.T]], axis=1)
        # task_path = np.concatenate([task_batch[:, :self.T-1], -1*np.ones([self.batch_size, 1, self.task_dim])], axis=1)
        before_obs = obs.reshape([self.batch_size*self.T]+self.obs_dims)
        obs = next_obs.reshape([self.batch_size*self.T]+self.obs_dims)
        next_obs = far_obs.reshape([self.batch_size*self.T]+self.obs_dims)
        task_path = task_path.reshape([self.batch_size*self.T, self.task_dim])
        # inds = np.random.choice(range(len(self.obs_data)), self.batch_size)

        # obs = []
        # next_obs = []
        # task_path = []
        # for i in inds:
        #     print i
        #     next_obs_batch = np.array([self.obs_data[i] for i in inds])[0]
        #     next_task_batch = np.array([self.task_data[i] for i in inds])[0]

        #     obs1 = next_obs_batch[:self.T-1].reshape([self.T-1]+list(self.obs_dims))
        #     obs.append(np.concatenate([obs1, np.zeros([1]+list(self.obs_dims))], 0))
        #     obs2 = next_obs_batch[1:self.T].reshape([self.T-1]+list(self.obs_dims))
        #     next_obs.append(np.concatenate([np.zeros([1]+list(self.obs_dims)), obs2], 0))
        #     task = next_task_batch[:self.T-1].reshape([1, self.task_dim])
        #     task_path.append(np.concatenate([task, -1*np.ones([1, self.task_dim])], 0))

        return self.sess.run(self.loss, feed_dict={self.x_in: obs,
                                                    self.offset_in: next_obs,
                                                    self.before_offset_in: before_obs,
                                                    self.task_in: task_path,
                                                    self.training: True,
                                                    self.lr: self.cur_lr,})


    def get_latents(self, obs):
        if len(obs) < self.batch_size*self.T:
            s = obs.shape
            obs = np.r_[obs, np.zeros((self.batch_size*self.T-s[0], s[1], s[2], s[3]))]
        return self.sess.run(self.encode_mu, feed_dict={self.x_in: obs, self.training: True})


    def get_next_latents(self, z, task, h=None):
        z = np.array(z)
        task = np.array(task)
        if self.use_recurrent_dynamics:
            z = z.reshape((1, 1, LATENT_DIM))
            task = task.reshape((1, 1, self.task_dim))
            z, h = self.sess.run([self.latent_trans_mu, self.trans_rnn_final_state], feed_dict={self.latent_in: z, self.latent_task_in: task, self.trans_rnn_initial_state: h, self.training: True})
        else:
            z = self.sess.run(self.latent_trans_mu, feed_dict={self.latent_in: z, self.latent_task_in: task, self.training: True})
            h = None
        return z.reshape(LATENT_DIM), h


    def next_latents_kl_pentalty(self, obs, task):
        return self.sess.run(self.conditional_kl_loss, feed_dict={self.x_in: obs, self.task_in: task, self.training: True})


    def decode_latent(self, latents):
        if len(latents) < self.batch_size*self.T:
            s = latents.shape
            latents = np.r_[latents, np.zeros((self.batch_size*self.T-s[0], s[1]))]
        return self.sess.run(self.decode_mu, feed_dict={self.decoder_in: latents, self.training: True})


    def test_decode(self, i=10000, t=3):
        o = self.obs_data[i, t].copy()
        z = self.get_latents(np.array([o]))
        d = self.decode_latent(np.array([z[0]]))
        d[d < 0] = 0
        d[d > 1] = 1
        d = (255*d).astype(np.uint8)


        if len(o) < self.batch_size*self.T:
            s = o.shape
            o = np.r_[[o], np.zeros((self.batch_size*self.T-1, s[0], s[1], s[2]))]
        d2 = self.sess.run(self.decode_mu, feed_dict={self.x_in: o, self.training: True})
        d2[d2 < 0] = 0
        d2[d2 > 1] = 1
        d2 = (255.*d2).astype(np.uint8)
        return o, d, d2
