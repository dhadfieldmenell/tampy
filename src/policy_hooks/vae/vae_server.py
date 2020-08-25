import threading
import time

import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, String

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.vae.vae import VAE


class VAEServer(object):
    def __init__(self, hyperparams):

        self.task = 'vae' # hyperparams['scope']
        topic = hyperparams.get('topic', '')
        rospy.init_node(self.task+'_update_server{0}'.format(topic))

        self.update_listener = rospy.Subscriber('vae_update{0}'.format(topic), VAEUpdate, self.update, queue_size=2, buff_size=2**25)
        self.weight_publisher = rospy.Publisher('vae_weights{0}'.format(topic), UpdateTF, queue_size=1)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        self.stopped = False
        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']
        # self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)
        self.vae = VAE(hyperparams['vae'])

        self.update_queue = []

        # rospy.spin()


    def run(self):
        while not self.stopped:
            rospy.sleep(0.01)
            self.parse_update_queue()


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def update(self, msg):
        obs = np.array(msg.obs).reshape(msg.dO)
        task_path = np.array(msg.tasks).reshape(msg.dTask)
        self.update_queue.append((obs, task_path))


    def parse_update_queue(self):
        while len(self.update_queue):
            obs, task_path = self.update_queue.pop()
            update = self.vae.store(obs, task_path)
            print('VAE Weights updated:', update, self.task)
            if update:
                msg = UpdateTF()
                msg.scope = 'vae'
                msg.data = self.vae.serialize_weights()
                self.weight_publisher.publish(msg)
