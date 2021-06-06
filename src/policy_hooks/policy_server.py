import pickle
import pprint
import random
import threading
import time
import queue
import numpy as np
import os, psutil

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.msg_classes import *
from policy_hooks.policy_data_loader import DataLoader
from policy_hooks.utils.policy_solver_utils import *

LOG_DIR = 'experiment_logs/'
MAX_QUEUE_SIZE = 100
UPDATE_TIME = 60

class PolicyServer(object):
    def __init__(self, hyperparams):
        global tf
        import tensorflow as tf
        self.group_id = hyperparams['group_id']
        self.task = hyperparams['scope']
        self.task_list = hyperparams['task_list']
        self.weight_dir = LOG_DIR+hyperparams['weight_dir']
        self.seed = int((1e2*time.time()) % 1000.)
        np.random.seed(self.seed)
        random.seed(self.seed)
        n_gpu = hyperparams['n_gpu']
        if n_gpu == 0:
            gpus = -1
        elif n_gpu == 1:
            gpu = 0
        else:
            gpus = np.random.choice(range(n_gpu))
            #gpus = n_gpu-1
            #gpus = str(list(range(1, n_gpu+1)))[1:-1]
            #gpus = str(list(range(0, n_gpu)))[1:-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = "{0}".format(gpus)
        self.start_t = hyperparams['start_t']
        self.config = hyperparams
        self.permute = hyperparams['permute_hl'] > 0

        ratios = hyperparams.get('ratios', {})
        for key in ['negative', 'optimal', 'dagger', 'rollout', 'human']:
            if key not in ratios: ratios[key] = hyperparams['perc_{}'.format(key)]

        for key in ratios: ratios[key] /= np.sum(list(ratios.values()))
        hyperparams['ratios'] = ratios

        hyperparams['policy_opt']['scope'] = self.task
        hyperparams['policy_opt']['load_label'] = hyperparams['classify_labels']
        hyperparams['policy_opt']['split_hl_loss'] = hyperparams['split_hl_loss']
        hyperparams['policy_opt']['gpu_id'] = 0
        hyperparams['policy_opt']['use_gpu'] = 1
        hyperparams['policy_opt']['load_all'] = self.task not in ['cont', 'primitive']
        hyperparams['agent']['master_config'] = hyperparams
        hyperparams['agent']['load_render'] = False
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.map_cont_discr_tasks()
        self.prim_opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        self.stopped = False
        self.warmup = hyperparams['tf_warmup_iters']
        self.queues = hyperparams['queues']
        self.min_buffer = hyperparams['prim_update_size'] if self.task in ['cont', 'primitive'] else hyperparams['update_size']
        if self.task == 'label':
            self.min_buffer = 600

        if self.task == 'primitive':
            self.in_queue = hyperparams['hl_queue']
        elif self.task == 'cont':
            self.in_queue = hyperparams['cont_queue']
        elif self.task == 'label':
            self.in_queue = hyperparams['label_queue']
        else:
            self.in_queue = hyperparams['ll_queue'][self.task]

        self.batch_size = hyperparams['batch_size']
        if self.task in ['cont', 'primitive', 'label']:
            normalize = False
        else:
            normalize = IM_ENUM not in hyperparams['obs_include']

        feed_prob = hyperparams['end_to_end_prob']
        in_inds, out_inds = None, None
        if len(self.continuous_opts):
            in_inds, out_inds = [], []
            opt1 = None
            if END_POSE_ENUM in self.agent._obs_data_idx:
                opt1 = END_POSE_ENUM

            for opt in self.continuous_opts:
                inopt = opt1 if opt1 is not None else opt
                in_inds.append(self.agent._obs_data_idx[inopt])
                out_inds.append(self.agent._cont_out_data_idx[opt])
                
            in_inds = np.concatenate(in_inds, axis=0)
            out_inds = np.concatenate(out_inds, axis=0)

        aug_f = None
        no_im = IM_ENUM not in hyperparams['prim_obs_include']
        if self.task == 'primitive' and hyperparams['permute_hl'] > 0 and no_im:
            aug_f = self.agent.permute_hl_data

        #if self.task == 'cont':
        #    aug_f = self.agent.permute_cont_data

        self.data_gen = DataLoader(hyperparams, \
                                   self.task, \
                                   self.in_queue, \
                                   self.batch_size, \
                                   normalize, \
                                   min_buffer=self.min_buffer, \
                                   aug_f=aug_f, \
                                   feed_prob=feed_prob, \
                                   feed_inds=(in_inds, out_inds), \
                                   feed_map=self.agent.center_cont, \
                                   save_dir=self.weight_dir+'/samples/')
      
        hyperparams['dPrim'] = len(hyperparams['prim_bounds'])
        hyperparams['dCont'] = len(hyperparams['cont_bounds'])
        if self.task == 'primitive' or (self.task == 'cont' and not len(self.cont_bounds)):
            dO = hyperparams['dPrimObs']
            dU = max([b[1] for b in self.discr_bounds] + [b[1] for b in hyperparams['aux_bounds']])
            dP = hyperparams['dPrim']
            precShape = tf.TensorShape([None, dP])
        elif self.task == 'cont':
            dO = hyperparams['dContObs']
            dU = max([b[1] for b in self.cont_bounds])
            dP = dU
            precShape = tf.TensorShape([None, dP, dP])
        elif self.task == 'label':
            dO = hyperparams['dPrimObs']
            dU = 2
            dP = 2
            precShape = tf.TensorShape([None, dP])
        else:
            dO = hyperparams['dO']
            dU = hyperparams['dU']
            dP = hyperparams['dU']
            precShape = tf.TensorShape([None, dP, dP])

        data = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3])
        #self.load_f = lambda x: tf.data.Dataset.from_generator(self.data_gen.gen_load, \
        #                                                 output_types=tf.int32, \
        #                                                 args=())
        #data = data.interleave(self.load_f, \
        #                         cycle_length=4, \
        #                         block_length=1)
        self.gen_f = lambda x: tf.data.Dataset.from_generator(self.data_gen.gen_items, \
                                                         output_types=(tf.float32, tf.float32, tf.float32), \
                                                         output_shapes=(tf.TensorShape([None, dO]), tf.TensorShape([None, dU]), precShape),
                                                         args=(x,))

        try:
            self.data = data.interleave(self.gen_f, \
                                         cycle_length=4, \
                                         block_length=1, \
                                         num_parallel_calls=4)
        except:
            self.data = data.interleave(self.gen_f, \
                                         cycle_length=4, \
                                         block_length=1)

        self.data = self.data.prefetch(4)

        self.input, self.act, self.prc = self.data.make_one_shot_iterator().get_next()

        hyperparams['policy_opt']['primitive_network_params']['output_boundaries'] = self.discr_bounds
        hyperparams['policy_opt']['cont_network_params']['output_boundaries'] = self.cont_bounds
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'],
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dPrimObs'],
            hyperparams['dContObs'],
            hyperparams['dValObs'],
            self.discr_bounds,
            contBounds=self.cont_bounds,
            inputs=(self.input, self.act, self.prc),
        )
        self.policy_opt.lr_policy = hyperparams['lr_policy']
        self.lr_policy = hyperparams['lr_policy']
        if self.task == 'primitive':
            self.data_gen.data_buf.x_idx = self.policy_opt.prim_x_idx
        elif self.task == 'cont':
            self.data_gen.data_buf.x_idx = self.policy_opt.cont_x_idx
        elif self.task == 'label':
            self.data_gen.data_buf.x_idx = self.policy_opt.label_x_idx
        else:
            self.data_gen.data_buf.x_idx = self.policy_opt.x_idx

        self.data_gen.policy = self.policy_opt.get_policy(self.task)
        self.data_gen.data_buf.policy = self.policy_opt.get_policy(self.task)

        self.policy_opt_log = LOG_DIR + hyperparams['weight_dir'] + '/policy_{0}_log.txt'.format(self.task)
        self.policy_info_log = LOG_DIR + hyperparams['weight_dir'] + '/policy_{0}_info.txt'.format(self.task)
        self.data_file = LOG_DIR + hyperparams['weight_dir'] + '/{0}_data.pkl'.format(self.task)
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[]}
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.task)+'_exp_data.npy'
        self.n_updates = 0
        self.update_t = time.time()
        self.n_data = []
        self.update_queue = []
        self.policy_var = {}
        self.policy_loss = []
        self.train_losses = {'all': [], 'optimal':[], 'rollout':[], 'aux': []}
        self.val_losses = {'all': [], 'optimal':[], 'rollout':[], 'aux': []}
        self.policy_component_loss = []
        self.log_infos = []
        self.cur_ratio = 1.
        with open(self.policy_opt_log, 'w+') as f:
            f.write('')


    def map_cont_discr_tasks(self):
        self.task_types = []
        self.discrete_opts = []
        self.continuous_opts = []
        self.discr_bounds = []
        self.cont_bounds = []
        cur_discr = 0
        cur_cont = 0
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for key, val in opts.items():
            if hasattr(val, '__len__'):
                self.task_types.append('discrete')
                self.discrete_opts.append(key)
                next_discr = cur_discr + len(val)
                self.discr_bounds.append((cur_discr, next_discr))
                cur_discr = next_discr
            else:
                self.task_types.append('continuous')
                self.continuous_opts.append(key)
                next_cont = cur_cont + int(val)
                self.cont_bounds.append((cur_cont, next_cont))
                cur_cont = next_cont


    def run(self):
        self.iters = 0
        write_freq = 50
        while not self.stopped:
            self.iters += 1
            init_t = time.time()
            self.data_gen.load_data()
            #if self.task == 'primitive': print('\nTime to get update:', time.time() - init_t, '\n')
            self.policy_opt.update(self.task)
            #if self.task == 'primitive': print('\nTime to run update:', time.time() - init_t, '\n')
            self.n_updates += 1
            mu, obs, prc = self.data_gen.get_batch()
            if len(mu):
                losses = self.policy_opt.check_validation(mu, obs, prc, task=self.task)
                self.train_losses['all'].append(losses[0])
                self.train_losses['aux'].append(losses)
            mu, obs, prc = self.data_gen.get_batch(val=True)
            if len(mu): 
                losses = self.policy_opt.check_validation(mu, obs, prc, task=self.task)
                self.val_losses['all'].append(losses[0])
                self.val_losses['aux'].append(losses)

            if self.lr_policy == 'adaptive':
                if len(self.train_losses['all']) and len(self.val_losses['all']) and self.policy_opt.cur_dec > 0:
                    ratio = np.mean(self.val_losses['optimal'][-5:]) / np.mean(self.train_losses['optimal'][-5:])
                    self.cur_ratio = ratio
                    if ratio < 1.2:
                        self.policy_opt.cur_dec *= 0.975
                    elif ratio > 1.7:
                        self.policy_opt.cur_dec *= 1.025
                    self.policy_opt.cur_dec = max(self.policy_opt.cur_dec, 1e-12)
                    self.policy_opt.cur_dec = min(self.policy_opt.cur_dec, 1e-1)

            for lab in ['optimal', 'rollout']:
                mu, obs, prc = self.data_gen.get_batch(label=lab, val=True)
                if len(mu): self.val_losses[lab].append(self.policy_opt.check_validation(mu, obs, prc, task=self.task)[0])

            if not self.iters % write_freq:
                self.policy_opt.write_shared_weights([self.task])
                if len(self.continuous_opts) and self.task not in ['cont', 'primitive', 'label']:
                    self.policy_opt.read_shared_weights(['cont'])
                    self.data_gen.feed_in_policy = self.policy_opt.cont_policy

                n_train = self.data_gen.get_size()
                n_val = self.data_gen.get_size(val=True)
                print('Ran', self.iters, 'updates on', self.task, 'with', n_train, 'train and', n_val, 'val')

            if self.config['save_data']:
                if not self.iters % write_freq and self.task in ['cont', 'primitive']:
                    self.data_gen.write_data(n_data=1024)

            if not self.iters % write_freq and len(self.val_losses['all']):
                with open(self.policy_opt_log, 'a+') as f:
                    info = self.get_log_info()
                    pp_info = pprint.pformat(info, depth=60)
                    f.write(str(pp_info))
                    f.write('\n\n')
            #if self.task == 'primitive': print('\nTime to finish update:', time.time() - init_t, '\n')
        self.policy_opt.sess.close()


    def get_log_info(self):
        test_acc, train_acc = -1, -1
        test_component_acc, train_component_acc = -1, -1
        #if self.task == 'primitive':
        #    obs, mu, prc = self.data_gen.get_batch()
        #    train_acc = self.policy_opt.task_acc(obs, mu, prc)
        #    train_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
        #    obs, mu, prc = self.data_gen.get_batch(val=True)
        #    test_acc = self.policy_opt.task_acc(obs, mu, prc)
        #    test_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
        info = {
                'time': time.time() - self.start_t,
                'train_loss': np.mean(self.train_losses['all'][-10:]),
                'train_component_loss': np.mean(self.train_losses['all'][-10:], axis=0),
                'val_loss': np.mean(self.val_losses['all'][-10:]),
                'val_component_loss': np.mean(self.val_losses['all'][-10:], axis=0),
                'scope': self.task,
                'n_updates': self.n_updates,
                'n_data': self.policy_opt.N,
                'tf_iter': self.policy_opt.tf_iter,
                'N': self.policy_opt.N,
                'reg_val': self.policy_opt.cur_dec,
                'loss_ratio': self.cur_ratio,
                }

        info['memory'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        info['in_queue_size'] = self.in_queue.qsize()

        for key in self.data_gen.data_buf.lens:
            info['n_train_{}'.format(key)] = self.data_gen.data_buf.lens[key]

        for key in self.policy_opt.buf_sizes:
            if key.find('n_') >= 0:
                 info[key] = self.policy_opt.buf_sizes[key].value

        info['labels'] = list(self.data_gen.data_buf.lens.keys())
        if len(self.val_losses['rollout']):
            info['rollout_val_loss'] = np.mean(self.val_losses['rollout'][-10:]),
            info['rollout_val_component_loss'] = np.mean(self.val_losses['rollout'][-10:], axis=0),
        if len(self.val_losses['optimal']):
            info['optimal_val_loss'] = np.mean(self.val_losses['optimal'][-10:]),
            info['optimal_val_component_loss'] = np.mean(self.val_losses['optimal'][-10:], axis=0),
        if len(self.train_losses['optimal']):
            info['optimal_train_loss'] = np.mean(self.train_losses['optimal'][-10:]),
            info['optimal_train_component_loss'] = np.mean(self.train_losses['optimal'][-10:], axis=0),
        if test_acc >= 0:
            info['test_accuracy'] = test_acc
            info['test_component_accuracy'] = test_component_acc
            info['train_accuracy'] = train_acc
            info['train_component_accuracy'] = train_component_acc
        #self.log_infos.append(info)
        return info #self.log_infos


    def update_expert_demos(self, obs, acs, rew=None):
        self.expert_demos['acs'].append(acs)
        self.expert_demos['obs'].append(obs)
        if rew is None:
            rew = np.ones(len(obs))
        self.expert_demos['ep_rets'].append(rew)
        self.expert_demos['rews'].append(rew)
        if self.n_updates % 200:
            np.save(self.expert_data_file, self.expert_demos)

