import threading
import time

import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, String

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.vae.reward_trainer import RewardTrainer


class RewrdServer(object):
    def __init__(self, hyperparams):

        self.task ='reward' # hyperparams['scope']
        rospy.init_node(self.task+'_update_server')

        # self.update_listener = rospy.Subscriber('reward_update', PathRewardUpdate, self.update, queue_size=2, buff_size=2**25)

        # Support for having one ros topic for both to listen on
        self.aux_update_listener = rospy.Subscriber('vae_update', VAEUpdate, self.update, queue_size=2, buff_size=2**25)
        self.weight_publisher = rospy.Publisher('reward_weights', UpdateTF, queue_size=1)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        self.stopped = False
        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']
        self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)
        self.reward_trainer = RewardTrainer(hyperparams['reward'])
        self.vae = VAE(hyperparams['vae'])

        self.update_queue = []
        self.weights_to_store = {}

        # rospy.spin()


    def run(self):
        while not self.stopped:
            rospy.sleep(0.01)
            self.parse_update_queue()


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def store_weights(self, msg):
        self.weights_to_store[msg.scope] = msg.data


    def update_weights(self):
        scopes = list(self.weights_to_store.keys())
        for scope in scopes:
            save = self.id.endswith('0')
            data = self.weights_to_store[scope]
            self.weights_to_store[scope] = None
            if data is not None:
                if scope == 'vae':
                    self.vae.deserialize_weights(data, save=save)


    def update(self, msg):
        obs = np.array(msg.obs).reshape(msg.dO)
        task_path = np.array(msg.tasks).reshape(msg.dTask)
        self.update_queue.append((obs, task_path))


    def parse_update_queue(self):
        while len(self.update_queue):
            obs, task_path = self.update_queue.pop()
            update = self.reward_trainer.store(obs, task_path)
            print('Reward Weights updated:', update, self.task)
            if update:
                msg = UpdateTF()
                msg.scope = 'reward'
                msg.data = self.reward_trainer.serialize_weights()
                self.weight_publisher.publish(msg)
