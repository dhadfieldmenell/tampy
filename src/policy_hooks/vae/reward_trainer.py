import copy
import json
import logging
import os
import sys
import tempfile
import traceback

import h5py
import numpy as np
import tensorflow as tf

from policy_hooks.vae.vae import VAE


ENCODER_CONFIG = {
    'n_channels': [16, 32, 32],
    'filter_sizes': [5, 5, 5],
    'strides': [3, 3, 3],
    'fc_dims': [16] # [2 * 3 * 32]
}

DECODER_CONFIG = {
    'conv_init_shape': [2, 3, 32],
    'n_channels': [32, 16, 3],
    'filter_sizes': [5, 5, 5],
    'strides': [3, 3, 3],
    'fc_dims': None
}

LATENT_DYNAMICS_CONFIG = {
    'fc_dims': [16, 16],
}

LATENT_DIM = 16


class LatentOffsetPredictor(object):
    def __init__(self, x1_in, x2_in, y, reuse):
        with tf.variable_scope('latent_offsets'):
            out = tf.concat([x1_in, x2_in], axis=1)
            out = tf.layers.dense(out, 64, activation=tf.nn.relu, name='dense1', reuse=reuse)
            out = tf.layers.dense(out, 1, activation=None, name='dense_out', reuse=reuse)
        self.x1_in = x1_in
        self.x2_in = x2_in
        self.y = y
        self.pred = out
        self.loss = tf.reduce_sum((self.pred - y)**2, axis=1)
        self.lr = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss)


    def train_step(self, obs1, obs2, labels, lr, sess):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.x1_in: obs1, self.x2_in: obs2, self.y: labels, self.lr: lr})
        return loss


    def pred(self, obs1, obs2):
        return sess.run(self.pred, feed_dict={self.x1_in: obs1, self.x2_in: obs2})


class RewardTrainer(object):
    def __init__(self, hyperparams):
        self.config = hyperparams
        tf.reset_default_graph()
        tf.set_random_seed(self.config.get('random_seed', 1234))

        self.tf_iter = 0
        self.batch_size = self.config.get('batch_size', 128)
        self.train_iters = self.config.get('train_iters', 10000)
        self.T = self.config.get('rollout_len', 20)
        self.vae = VAE(self.config['vae'])

        self.obs_dims = [LATENT_DIM]
        self.task_dim = hyperparams['vae']['task_dims']

        self.weight_dir = hyperparams['vae']['weight_dir']
        self.load_step = hyperparams.get('load_step', -1)
        self.train_mode = hyperparams['vae'].get('train_mode', 'conditional')
        if self.load_step < 0:
            self.ckpt_name = self.weight_dir+'/reward.ckpt'.format(self.train_mode)
        else:
            self.ckpt_name = self.weight_dir+'/reward_{0}_{1}.ckpt'.format(self.train_mode, self.load_step)


        self.data_file = self.weight_dir+'/vae_buffer.hdf5'
        self.data = h5py.File(self.data_file, 'r')
        self.obs_data = self.data['obs_data']
        self.task_data = self.data['task_data']
        # self.data_file = self.weight_dir+'/vae_buffer.npz'
        # try:
        #     data = np.load(self.data_file, mmap_mode='w+')
        # except:
        #     pass

        # self.obs_data = np.zeros((0, self.dT, self.dO))
        # self.task_data = np.zeros((0, self.dT, self.dU))
        self.max_buffer = hyperparams.get('max_buffer', 1e6)
        self.dist_constraint = hyperparams.get('distance_constraint', False)

        self.cur_lr = 5e-4
        with tf.variable_scope('reward', reuse=False):
            self.init_network()
            self.init_solver()

        self.scope = 'reward'
        self.gpu_fraction = self.config['gpu_fraction'] if 'gpu_fraction' in self.config else 0.95
        if 'allow_growth' in self.config and not self.config['allow_growth']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        init_op = tf.initialize_all_variables()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(variables)
        try:
            self.saver.restore(self.sess, self.ckpt_name)
        except Exception as e:
            self.sess.run(init_op)
            print('\n\nCould not load previous weights for {0} from {1}\n\n'.format(self.scope, self.weight_dir))

        self.update_count = 0
        self.update_size = self.config.get('update_size', 100)


    def serialize_weights(self):
        print('Serializing reward weights')
        var_to_val = {}
        variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reward')
        for v in variables:
            var_to_val[v.name] = self.sess.run(v).tolist()

        return json.dumps(var_to_val)


    def deserialize_weights(self, json_wts, save=True):
        var_to_val = json.loads(json_wts)

        # print 'Deserializing', scopes
        variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reward')
        for var in variables:
            var.load(var_to_val[var.name], session=self.sess)

        if save: self.store_scope_weights(scopes='reward')
        # print 'Weights for {0} successfully deserialized and stored.'.format(scopes)


    def update_weights(self, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        self.saver.restore(self.sess, weight_dir+'/reward.ckpt')


    def store_scope_weights(self, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        try:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reward')
            saver = tf.train.Saver(variables)
            saver.save(self.sess, weight_dir+'/reward.ckpt')
        except:
            print('Saving variables encountered an issue but it will not crash:')
            traceback.print_exception(*sys.exc_info())


    def store_weights(self, weight_dir=None):
        self.store_scope_weights('reward', weight_dir)


    def store(self, obs, task_list):
        print('Storing data for', self.scope)
        assert len(obs) == len(task_list)
        self.T = len(obs)

        # self.obs_data = np.r_[self.obs_data, obs]
        # self.task_data = np.r_[self.task_data, task_list]
        obs = obs[:self.T]
        task_list = task_list[:self.T]

        obs = obs.reshape((1,)+obs.shape)
        task_list = task_list.reshape((1,)+task_list.shape)

        self.obs_data.resize((len(self.obs_data)+1,) + obs.shape[1:])
        self.obs_data[-1] = obs

        self.task_data.resize((len(self.task_data)+1,) + task.shape[1:])
        self.task_data[-1] = task_list

        if len(self.obs_data) > self.max_buffer:
            self.obs_data = self.obs_data[-self.max_buffer:]
            self.task_data = self.task_data[-self.max_buffer:]

        self.update_count += 1
        if self.update_count > self.update_size:
            print('Updating', net)
            self.update()
            self.store_scope_weights(scopes=[net])
            self.save_buffers()
            self.update_count = 0
            return True

        return False


    # def save_buffers(self):
    #     np.savez(self.data_file, task_data=self.task_data, obs_data=self.obs_data)


    def init_network(self):
        import tensorflow as tf
        self.x1_in = tf.placeholder(tf.float32, shape=[None]+list(self.obs_dims))
        self.x2_in = tf.placeholder(tf.float32, shape=[None]+list(self.obs_dims))
        self.x_in = tf.concat([self.x1_in, self.x2_in], axis=1)
        self.y = tf.placeholder(tf.float32, shape=[None])

        self.task_in = tf.placeholder(tf.float32, shape=[None, self.task_dim])
        self.training = tf.placeholder(tf.bool)

        self.net = LatentOffsetPredictor(self.x1_in, self.x2_in, self.y, reuse=False)


    def init_solver(self):
        pass


    def train(self):
        for i in range(10000):
            self.update()
            self.saver.save(self.sess, self.ckpt_name)


    def update(self):
        for i in range(self.train_iters):
            next_batch1 = []
            next_batch2 = []
            ts = []
            for j in range(self.batch_size):
                ind = np.random.choice(list(range(len(self.obs_data))), 1)[0]
                next_obs_batch = np.array(self.obs_data[ind])
                t1 = np.random.randint(0, len(next_obs_batch)-1)
                t2 = np.random.randint(t1+1, len(next_obs_batch))
                next_batch1.append(next_obs_batch[t1])
                next_batch2.append(next_obs_batch[t2])
                ts.append(t2 - t1)
            next_batch1 = self.vae.get_latents(next_batch1)
            next_batch2 = self.vae.get_latents(next_batch2)

            self.net.train_step(next_batch1, next_batch2, ts, self.cur_lr, self.sess)
            print(i)


    def get_offsets(self, obs, goals):
        return self.net.pred(obs, goals)
