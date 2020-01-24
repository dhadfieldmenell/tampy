import copy
import numpy as np
import pickle
import pprint
import rospy
import time

from std_msgs.msg import Float32MultiArray, String

from tamp_ros.msg import *
from tamp_ros.srv import *


class LogServer(object):
    def __init__(self, hyperparams):
        rospy.init_node('log_update_server')
        self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)
        self.stopped = False
        self.update_queue = []
        self.hist_len = 50 # Used to average costs from most recent updates
        self.start_time = time.time()
        self.last_ckpt = self.start_time
        self.n_opt_calls = {}
        self.n_hl_calls = 0
        self.traj_init_costs = {}
        self.postcondition_costs = {}
        self.tree_sizes = []
        self.passes_per_tree = []
        self.ckpts = []
        self.log_file = 'tf_saved/'+hyperparams['weight_dir']+'/master_log.pkl'
        self.updater = rospy.Subscriber('log_update', String, self.update, queue_size=10)
        # rospy.spin()


    def checkpoint(self):
        self.ckpts.append({
            'time': time.time(),
            'n_opt_calls': copy.deepcopy(self.n_opt_calls),
            'n_hl_calls': self.n_hl_calls,
            'traj_init_costs': copy.deepcopy(self.traj_init_costs),
            'postcondition_costs': copy.deepcopy(self.postcondition_costs),
            'tree_sizes': copy.deepcopy(self.tree_sizes)
        })

        with open(self.log_file, 'wb') as f:
            pickle.dump(self.ckpts, f)
        self.last_ckpt = time.time()


    # def end(self, msg):
    #     self.stopped = True
    #     rospy.signal_shutdown('Received notice to terminate.')


    def update(self, msg):
        self.queue(msg, 'update_queue')


    def queue(self, data, queue_name, key=None):
        if key is None:
            queue = getattr(self, queue_name)
        elif key in getattr(self, queue_name):
            queue = getattr(self, queue_name)[key]
        else:
            queue = []
            getattr(self, queue_name)[key] = queue

        queue.append(data)
        while len(queue) > self.hist_len:
            queue.pop(0)


    def run(self):
        while not self.stopped:
            rospy.sleep(1)
            self.parse_update_queue()


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def parse_update_queue(self):
        while len(self.update_queue):
            update = eval(msg.data)
            if 'rollouts' in update: self.update_rollout(update['rollouts'])
            if 'motion_plan' in update: self.update_mp(update['motion_plan'])
            if 'high_level' in update: self.update_hl(update['high_level'])

        if time.time() - self.last_ckpt > 60:
            self.checkpoint()


    def update_rollout(self, update):
        task_name = update['task_name']
        for queue in ['tree_sizes']:
            self.queue(update[queue], queue)

        for queue in ['postcondition_costs']:
            self.queue(update[queue], queue, key=task_name)


    def update_mp(self, update):
        task_name = update['task_name']
        for queue in ['traj_init_costs']:
            self.queue(update[queue], queue, key=task_name)
        if task_name not in self.n_opt_calls:
            self.n_opt_calls[task_name] = 0
        self.n_opt_calls[task_name] += 1


    def update_hl(self, update):
        for queue in []:
            self.queue(update[queue], queue)
        self.n_hl_calls += 1

