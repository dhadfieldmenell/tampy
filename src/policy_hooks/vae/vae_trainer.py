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
        # sub_env = hyperparams['env']()
        self.stopped = False

        hyperparams['vae']['data_read_only'] = True
        self.vae = VAE(hyperparams['vae'])

        seed = int(1000*time.time()) % 1000
        np.random.seed(seed)
        random.seed(seed)


    def train(self):
        for i in range(50000):
            self.vae.update()
            print((self.vae.get_weight_file(), np.mean([self.vae.check_loss() for _ in range(10)])))
            if not i % 100:
                self.vae.store_scope_weights()
