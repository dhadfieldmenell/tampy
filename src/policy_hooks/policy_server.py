import random
import threading
import time

import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, String

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf

from tamp_ros.msg import *
from tamp_ros.srv import *


MAX_QUEUE_SIZE = 100
UPDATE_TIME = 60

class PolicyServer(object):
    def __init__(self, hyperparams):
        import tensorflow as tf
        self.group_id = hyperparams['group_id']
        self.task = hyperparams['scope']
        self.task_list = hyperparams['task_list']
        self.seed = int(time.time() % 10000.)
        np.random.seed(self.seed)
        random.seed(self.seed)
        hyperparams['policy_opt']['scope'] = self.task
        rospy.init_node(self.task+'_update_server_{0}'.format(self.group_id))
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'], 
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dPrimObs'],
            hyperparams['dValObs'],
            hyperparams['prim_bounds']
        )
        # self.policy_opt = policy_opt
        # self.policy_opt.hyperparams['scope'] = task
        # self.prob_service = rospy.Service(self.task+'_policy_prob', PolicyProb, self.prob)
        # self.act_service = rospy.Service(self.task+'_policy_act', PolicyAct, self.act)
        self.weight_publisher = rospy.Publisher('tf_weights_{0}'.format(self.group_id), UpdateTF, queue_size=1)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        self.stopped = False
        self.time_log = 'tf_saved/' + hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']
        # self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)

        self.update_listener = rospy.Subscriber('{0}_update_{1}'.format(self.task, self.group_id), PolicyUpdate, self.update, queue_size=2, buff_size=2**25)
        self.poicy_opt_log = 'tf_saved/' + hyperparams['weight_dir'] + '/policy_{0}_log.txt'.format(self.task)
        self.n_updates = 0
        self.full_N = 0
        self.start_t = time.time()
        self.update_t = time.time()

        self.update_queue = []


    def run(self):
        while not self.stopped:
            self.parse_update_queue()


    def end(self, msg):
        print('SHUTTING DOWN')
        self.stopped = True
        # rospy.signal_shutdown('Received notice to terminate.')


    def update(self, msg):
        mu = np.array(msg.mu)
        mu_dims = (msg.n, msg.rollout_len, msg.dU)
        mu = mu.reshape(mu_dims)

        obs = np.array(msg.obs)
        if self.task == "value":
            obs_dims = (msg.n, msg.rollout_len, msg.dValObs)
        elif self.task == "primitive":
            obs_dims = (msg.n, msg.rollout_len, msg.dPrimObs)
        else:
            obs_dims = (msg.n, msg.rollout_len, msg.dO)
        obs = obs.reshape(obs_dims)

        prc = np.array(msg.prc)
        prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
        prc = prc.reshape(prc_dims)

        wt_dims = (msg.n, msg.rollout_len) if msg.rollout_len > 1 else (msg.n,)
        wt = np.array(msg.wt).reshape(wt_dims)
        self.update_queue.append((obs, mu, prc, wt, msg.task))
        self.update_queue = self.update_queue[-MAX_QUEUE_SIZE:]


    def parse_update_queue(self):
        queue_len = len(self.update_queue)
        for i in range(queue_len):
            obs, mu, prc, wt, task_name = self.update_queue.pop()
            start_time = time.time()
            self.full_N += len(mu)
            update = self.policy_opt.store(obs, mu, prc, wt, self.task, task_name, update=(i==(queue_len-1)))
            end_time = time.time()

            # print 'Weights updated:', update, self.task
            if update: #  and time.time() - self.update_t > UPDATE_TIME:
                self.n_updates += 1
                if self.policy_opt.share_buffers:
                    self.policy_opt.write_shared_weights([self.task])
                else:
                    msg = UpdateTF()
                    msg.scope = str(self.task)
                    msg.data = self.policy_opt.serialize_weights([self.task])
                    self.weight_publisher.publish(msg)
                self.update_t = time.time()
                print('Updated weights for {0}'.format(self.task))


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
