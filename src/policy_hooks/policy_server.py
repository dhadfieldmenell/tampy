import pickle
import pprint
import random
import threading
import time
import queue
import numpy as np

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.msg_classes import *
from policy_hooks.policy_data_loader import DataLoader

LOG_DIR = 'experiment_logs/'
MAX_QUEUE_SIZE = 100
UPDATE_TIME = 60

class PolicyServer(object):
    def __init__(self, hyperparams):
        import tensorflow as tf
        self.group_id = hyperparams['group_id']
        self.task = hyperparams['scope']
        self.task_list = hyperparams['task_list']
        self.seed = int((1e2*time.time()) % 1000.)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.start_t = hyperparams['start_t']
        self.config = hyperparams
        self.permute = hyperparams['permute_hl'] > 0
        hyperparams['policy_opt']['scope'] = self.task
        hyperparams['policy_opt']['split_hl_loss'] = hyperparams['split_hl_loss']
        hyperparams['policy_opt']['gpu_id'] = 0
        hyperparams['policy_opt']['use_gpu'] = 1
        hyperparams['agent']['master_config'] = hyperparams
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.stopped = False
        self.warmup = hyperparams['tf_warmup_iters']
        self.queues = hyperparams['queues']
        self.min_buffer = hyperparams['prim_update_size'] if self.task == 'primitive' else hyperparams['update_size']
        self.in_queue = hyperparams['hl_queue'] if self.task == 'primitive' else hyperparams['ll_queue']
        self.batch_size = hyperparams['batch_size']
        normalize = self.task != 'primitive'
        self.data_gen = DataLoader(hyperparams, self.task, self.in_queue, self.batch_size, normalize, min_buffer=self.min_buffer)
        aug_f = None
        if self.task == 'primitive' and hyperparams['permute_hl'] > 0:
            aug_f = self.agent.permute_hl_data
        self.data_gen = DataLoader(hyperparams, self.task, self.in_queue, self.batch_size, normalize, min_buffer=self.min_buffer, aug_f=aug_f)
      
        hyperparams['dPrim'] = len(hyperparams['prim_bounds'])
        dO = hyperparams['dPrimObs'] if self.task == 'primitive' else hyperparams['dO']
        dU = hyperparams['prim_bounds'][-1][-1] if self.task == 'primitive' else hyperparams['dU']
        dP = hyperparams['dPrim'] if self.task == 'primitive' else hyperparams['dU']
        precShape = tf.TensorShape([None, dP]) if self.task == 'primitive' else tf.TensorShape([None, dP, dP])
        data = tf.data.Dataset.from_tensor_slices([0, 1, 2])
        self.load_f = lambda x: tf.data.Dataset.from_generator(self.data_gen.gen_load, \
                                                         output_types=tf.int32, \
                                                         args=())
        data = data.interleave(self.load_f, \
                                 cycle_length=3, \
                                 block_length=1)
        self.gen_f = lambda x: tf.data.Dataset.from_generator(self.data_gen.gen_items, \
                                                         output_types=(tf.float32, tf.float32, tf.float32), \
                                                         output_shapes=(tf.TensorShape([None, dO]), tf.TensorShape([None, dU]), precShape),
                                                         args=(x,))
        self.data = data.interleave(self.gen_f, \
                                     cycle_length=3, \
                                     block_length=1)
        self.data = self.data.prefetch(3)
        self.input, self.act, self.prc = self.data.make_one_shot_iterator().get_next()

        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'],
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dPrimObs'],
            hyperparams['dValObs'],
            hyperparams['prim_bounds'],
            (self.input, self.act, self.prc),
        )
        self.policy_opt.lr_policy = hyperparams['lr_policy']
        self.data_gen.x_idx = self.policy_opt.x_idx if self.task != 'primitive' else self.policy_opt.prim_x_idx
        self.data_gen.policy = self.policy_opt.get_policy(self.task)

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
        self.train_losses = {'all': [], 'optimal':[], 'rollout':[]}
        self.val_losses = {'all': [], 'optimal':[], 'rollout':[]}
        self.policy_component_loss = []
        self.log_infos = []
        with open(self.policy_opt_log, 'w+') as f:
            f.write('')


    def run(self):
        self.iters = 0
        while not self.stopped:
            self.iters += 1
            self.policy_opt.update(self.task)
            self.n_updates += 1
            mu, obs, prc = self.data_gen.get_batch()
            if len(mu): self.train_losses['all'].append(self.policy_opt.check_validation(mu, obs, prc, task=self.task))
            mu, obs, prc = self.data_gen.get_batch(val=True)
            if len(mu): self.val_losses['all'].append(self.policy_opt.check_validation(mu, obs, prc, task=self.task))

            for lab in ['optimal', 'rollout']:
                mu, obs, prc = self.data_gen.get_batch(label=lab, val=True)
                if len(mu): self.val_losses[lab].append(self.policy_opt.check_validation(mu, obs, prc, task=self.task))

            if not self.iters % 10:
                self.policy_opt.write_shared_weights([self.task])
                n_train = self.data_gen.get_size()
                n_val = self.data_gen.get_size(val=True)
                print('Ran', self.iters, 'updates on', self.task, 'with', n_train, 'train and', n_val, 'val')
            if not self.iters % 10 and len(self.val_losses['all']):
                with open(self.policy_opt_log, 'a+') as f:
                    info = self.get_log_info()
                    pp_info = pprint.pformat(info, depth=60)
                    f.write(str(pp_info))
                    f.write('\n\n')
        self.policy_opt.sess.close()


    def get_log_info(self):
        test_acc, train_acc = -1, -1
        if self.task == 'primitive':
            obs, mu, prc = self.data_gen.get_batch()
            train_acc = self.policy_opt.task_acc(obs, mu, prc)
            train_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
            obs, mu, prc = self.data_gen.get_batch(val=True)
            test_acc = self.policy_opt.task_acc(obs, mu, prc)
            test_component_acc = self.policy_opt.task_acc(obs, mu, prc, scalar=False)
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
                'n_plans': self.policy_opt.buf_sizes['n_plans'].value,
                'n_postcond': self.policy_opt.buf_sizes['n_postcond'].value,
                'n_explore': self.policy_opt.buf_sizes['n_explore'].value,
                'n_rollout': self.policy_opt.buf_sizes['n_rollout'].value,
                }
        if len(self.val_losses['rollout']):
            info['rollout_val_loss'] = np.mean(self.val_losses['rollout'][-10:]),
            info['rollout_val_component_loss'] = np.mean(self.val_losses['rollout'][-10:], axis=0),
        if len(self.val_losses['optimal']):
            info['optimal_val_loss'] = np.mean(self.val_losses['optimal'][-10:]),
            info['optimal_val_component_loss'] = np.mean(self.val_losses['optimal'][-10:], axis=0),
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

