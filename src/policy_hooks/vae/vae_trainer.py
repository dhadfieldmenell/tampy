from datetime import datetime
import numpy as np
import os
import random
import sys
import time

from numba import cuda
from scipy.cluster.vq import kmeans2 as kmeans
from scipy.stats import multivariate_normal
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.agent_env_wrapper import AgentEnvWrapper
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.vae.vae import VAE


class VAETrainer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        np.random.seed(int(time.time()/100000))
        self.num_samples = 1
        self.rollout_len = hyperparams['rollout_len']
        sub_env = hyperparams['env']()
        self.stopped = False

        hyperparams['vae']['data_read_only'] = True
        hyperparams['vae']['train_mode'] = 'conditional'
        self.vae = VAE(hyperparams['vae'])
        self.weights_to_store = {}
        self.prior = multivariate_normal

        seed = int(1000*time.time()) % 1000
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        for i in range(10000):
            self.vae.update()
            if not i % 10:
                self.vae.store_scope_weights(addendum=i*self.vae.train_iters)