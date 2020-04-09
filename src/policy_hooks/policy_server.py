import pickle
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
        self.data_file = 'tf_saved/' + hyperparams['weight_dir'] + '/data.pkl'.format(self.task)
        self.n_updates = 0
        self.full_N = 0
        self.update_t = time.time()
        self.n_data = []
        self.update_queue = []
        with open(self.policy_opt_log, 'w+') as f:
            f.write('')


    def run(self):
        while not self.stopped:
            if not USE_ROS: self.parse_data()
            self.parse_update_queue()
            self.update_network()
            if time.time() - self.start_t > self.config['time_limit']:
                break
        self.policy_opt.sess.close()


    def end(self, msg):
        print('SHUTTING DOWN')
        self.stopped = True
        # rospy.signal_shutdown('Received notice to terminate.')

    
    def parse_data(self):
        q = self.queues['{0}_pol'.format(self.task)]
        i = 0
        while i < q._maxsize and not q.empty():
            try:
                msg = q.get_nowait()
                self.update(msg)
            except queue.Empty:
                break


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

        wt_dims = (msg.n, msg.rollout_len, 1) if msg.rollout_len > 1 else (msg.n,1)
        wt = np.array(msg.wt).reshape(wt_dims)
        self.update_queue.append((obs, mu, prc, wt, msg.task))
        self.update_queue = self.update_queue[-MAX_QUEUE_SIZE:]


    def parse_update_queue(self):
        queue_len = len(self.update_queue)
        for i in range(queue_len):
            obs, mu, prc, wt, task_name = self.update_queue.pop()
            start_time = time.time()
            self.full_N += len(mu)
            self.n_data.append(self.full_N)
            update = self.policy_opt.store(obs, mu, prc, wt, self.task, task_name, update=(i==(queue_len-1)))
            end_time = time.time()
        with open(self.policy_info_log, 'w+') as f:
            f.write(str(self.n_data))


    def update_network(self):
        update = self.policy_opt.run_update([self.task])
        # print 'Weights updated:', update, self.task
        if update:
            self.n_updates += 1
            if not USE_ROS or self.policy_opt.share_buffers:
                self.policy_opt.write_shared_weights([self.task])
            else:
                msg = UpdateTF()
                msg.scope = str(self.task)
                msg.data = self.policy_opt.serialize_weights([self.task])
                self.weight_publisher.publish(msg)
            self.update_t = time.time()
            print('Updated weights for {0}'.format(self.task))

            incr = 10
            lossess = [self.policy_opt.average_losses[::incr], self.policy_opt.average_val_losses[::incr]]
            with open(self.policy_opt_log, 'w+') as f:
                f.write(str(lossess))
            if not self.n_updates % 20:
                with open(self.data_file, 'w+') as f:
                    pickle.dump(self.policy_opt.get_data(), f)


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
