import pickle
import pprint
import random
import threading
import time
import queue
import numpy as np

from software_constants import USE_ROS

if USE_ROS:
    import rospy
    from std_msgs.msg import Float32MultiArray, String
    from tamp_ros.msg import *
    from tamp_ros.srv import *

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.msg_classes import *


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
        hyperparams['policy_opt']['scope'] = self.task
        if USE_ROS: rospy.init_node(self.task+'_update_server_{0}'.format(self.group_id))
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'],
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dPrimObs'],
            hyperparams['dValObs'],
            hyperparams['prim_bounds']
        )
        self.policy_opt.lr_policy = hyperparams['lr_policy']
        # self.policy_opt = policy_opt
        # self.policy_opt.hyperparams['scope'] = task
        self.stopped = False
        if USE_ROS:
            # self.prob_service = rospy.Service(self.task+'_policy_prob', PolicyProb, self.prob)
            # self.act_service = rospy.Service(self.task+'_policy_act', PolicyAct, self.act)
            self.weight_publisher = rospy.Publisher('tf_weights_{0}'.format(self.group_id), UpdateTF, queue_size=1)
            self.stop = rospy.Subscriber('terminate', String, self.end)
            self.time_log = 'tf_saved/' + hyperparams['weight_dir']+'/timing_info.txt'
            self.log_timing = hyperparams['log_timing']
            # self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)
            self.update_listener = rospy.Subscriber('{0}_update_{1}'.format(self.task, self.group_id), PolicyUpdate, self.update, queue_size=2, buff_size=2**25)
        else:
            self.queues = hyperparams['queues']
        self.policy_opt_log = 'tf_saved/' + hyperparams['weight_dir'] + '/policy_{0}_log.txt'.format(self.task)
        self.policy_info_log = 'tf_saved/' + hyperparams['weight_dir'] + '/policy_{0}_info.txt'.format(self.task)
        self.data_file = 'tf_saved/' + hyperparams['weight_dir'] + '/{0}_data.pkl'.format(self.task)
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[]}
        self.expert_data_file = 'tf_saved/'+hyperparams['weight_dir']+'/'+str(self.task)+'_exp_data.npy'
        self.n_updates = 0
        self.update_t = time.time()
        self.n_data = []
        self.update_queue = []
        self.policy_var = {}
        self.policy_loss = []
        self.policy_component_loss = []
        self.log_infos = []
        with open(self.policy_opt_log, 'w+') as f:
            f.write('')


    def run(self):
        while not self.stopped:
            if not USE_ROS: self.parse_data()
            self.parse_update_queue()
            self.update_network()
            #if time.time() - self.start_t > self.config['time_limit']:
            #    break
        self.policy_opt.sess.close()


    def end(self, msg):
        print('SHUTTING DOWN')
        self.stopped = True
        # rospy.signal_shutdown('Received notice to terminate.')


    def parse_data(self):
        q = self.queues['{0}_pol'.format(self.task)]
        i = 0
        retrieved = False
        while i < q._maxsize and not q.empty():
            i += 1
            try:
                msg = q.get_nowait()
                self.update(msg)
                retrieved = True
            except queue.Empty:
                break

        if not retrieved:
            try:
                msg = q.get(block=True, timeout=0.1)
                self.update(msg)
                retrieved = True
            except queue.Empty:
                pass


    def update(self, msg):
        mu = np.array(msg.mu)
        mu_dims = (msg.n, msg.rollout_len, msg.dU)
        mu = mu.reshape(mu_dims)

        obs = np.array(msg.obs)
        if msg.task == "value":
            obs_dims = (msg.n, msg.rollout_len, msg.dValObs)
        elif msg.task == "primitive":
            obs_dims = (msg.n, msg.rollout_len, msg.dPrimObs)
        else:
            obs_dims = (msg.n, msg.rollout_len, msg.dO)
        obs = obs.reshape(obs_dims)

        prc = np.array(msg.prc)
        if msg.task == "primitive":
            prc_dims = (msg.n, msg.dOpts) # Use prc as the masking
        elif msg.task == "switch" or msg.task == "value":
            prc_dims = (msg.n, 1)
        else:
            prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
        prc = prc.reshape(prc_dims)

        wt_dims = (msg.n, msg.rollout_len, 1) if msg.rollout_len > 1 else (msg.n,1)
        wt = np.array(msg.wt).reshape(wt_dims)
        if msg.task == 'value':
            act_dims = (msg.n, msg.rollout_len, msg.dAct)
            acts = np.reshape(msg.acts, act_dims)
            ref_act_dims = (msg.n, msg.rollout_len, msg.nActs, msg.dAct)
            ref_acts = np.reshape(msg.ref_acts, ref_act_dims)
            done = np.reshape(msg.terminal, (msg.n,1))
            self.update_queue.append((obs, mu, prc, wt, msg.task, acts, ref_acts, done))
        else:
            self.update_queue.append((obs, mu, prc, wt, msg.task))
        self.update_queue = self.update_queue[-MAX_QUEUE_SIZE:]


    def parse_update_queue(self):
        queue_len = len(self.update_queue)
        for i in range(queue_len):
            if self.task == 'value':
                obs, mu, prc, wt, task_name, acts, ref_acts, done = self.update_queue.pop()
            else:
                obs, mu, prc, wt, task_name, = self.update_queue.pop()
            start_time = time.time()
            self.n_data.append(self.policy_opt.N)
            if self.task == 'value':
                update = self.policy_opt.store(obs, mu, prc, wt, self.task, task_name, update=(i==(queue_len-1)), acts=acts, ref_acts=ref_acts, done=done)
            else:
                update = self.policy_opt.store(obs, mu, prc, wt, self.task, task_name, update=(i==(queue_len-1)), val_ratio=0.1)
                #if self.config.get('save_expert', False):
                #    self.update_expert_demos(obs, mu)
            end_time = time.time()


    def update_network(self, n_updates=20):
        for _ in range(n_updates):
            update = self.policy_opt.run_update([self.task])
        # print('Weights updated:', update, self.task)
        if update:
            self.n_updates += n_updates
            if not USE_ROS or self.policy_opt.share_buffers:
                self.policy_opt.write_shared_weights([self.task])
            else:
                msg = UpdateTF()
                msg.scope = str(self.task)
                msg.data = self.policy_opt.serialize_weights([self.task])
                self.weight_publisher.publish(msg)
            self.update_t = time.time()
            print(('Updated weights for {0}'.format(self.task)))

            incr = 10
            if len(self.policy_opt.average_losses) and len(self.policy_opt.average_val_losses):
                losses = (self.policy_opt.average_losses[-1], self.policy_opt.average_val_losses[-1])
                self.policy_loss.append((np.sum(losses[0]), np.sum(losses[1])))
                self.policy_component_loss.append(losses)

                for net in self.policy_opt.mu:
                    if net not in self.policy_var:
                        self.policy_var[net] = []
                    data = np.concatenate(list(self.policy_opt.mu[net].values()), axis=0)
                    self.policy_var[net].append(np.var(data))

                with open(self.policy_opt_log, 'w+') as f:
                    info = self.get_log_info()
                    pp_info = pprint.pformat(info, depth=60)
                    f.write(str(pp_info))
            # if not self.n_updates % 20:
            #     with open(self.data_file, 'w+') as f:
            #         pickle.dump(self.policy_opt.get_data(), f)


    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                'var': {net: self.policy_var[net][-1] for net in self.policy_var},
                'train_loss': self.policy_loss[-1][0],
                'train_component_loss': self.policy_component_loss[-1][0],
                'val_loss': self.policy_loss[-1][1],
                'val_component_loss': self.policy_component_loss[-1][1],
                'scope': self.task,
                'n_updates': self.n_updates,
                'n_data': self.policy_opt.N,
                'tf_iter': self.policy_opt.tf_iter,
                'N': self.policy_opt.N,
                }
        self.log_infos.append(info)
        return self.log_infos


    def prob(self, req):
        obs = np.array([req.obs[i].data for i in range(len(req.obs))])
        mu_out, sigma_out, _, _ = self.policy_opt.prob(np.array([obs]), task)
        mu, sigma = [], []
        for i in range(len(mu_out[0])):
            next_line = Float32MultiArray()
            next_line.data = mu[0, i]
            mu.append(next_line)

            next_line = Float32MultiArray()
            next_line.data = np.diag(sigma[0, i])

        return PolicyProbResponse(mu, sigma)


    def act(self, req):
        # Assume time invariant policy
        obs = np.array(req.obs)
        noise = np.array(req.noise)
        policy = self.policy_opt.task_map[self.task]['policy']
        if policy.scale is None:
            policy.scale = 0.01
            policy.bias = 0
            act = policy.act([], obs, 0, noise)
            policy.scale = None
            policy.bias = None
        else:
            act = policy.act([], obs, 0, noise)
        return PolicyActResponse(act)


    def get_prim_update(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0, 1))
        for sample in samples:
            mu = np.concatenate([sample.get(enum) for enum in self.config['prim_out_include']], axis=-1)
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = np.ones((sample.T,1)) # np.array([self.prim_decay**t for t in range(sample.T)])
            wt[0] *= 1. # self.prim_first_wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dP), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))
            if False: # self.add_negative:
                mu = self.find_negative_ex(sample)
                tgt_mu = np.concatenate((tgt_mu, mu))
                wt = -np.ones((sample.T,))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs = sample.get_prim_obs()
                obs_data = np.concatenate((obs_data, obs))
                prc = np.tile(np.eye(dP), (sample.T,1,1))
                tgt_prc = np.concatenate((tgt_prc, prc))

        return obs_data, tgt_mu, tgt_prc, tgt_wt


    def update_expert_demos(self, obs, acs, rew=None):
        self.expert_demos['acs'].append(acs)
        self.expert_demos['obs'].append(obs)
        if rew is None:
            rew = np.ones(len(obs))
        self.expert_demos['ep_rets'].append(rew)
        self.expert_demos['rews'].append(rew)
        if self.n_updates % 200:
            np.save(self.expert_data_file, self.expert_demos)

