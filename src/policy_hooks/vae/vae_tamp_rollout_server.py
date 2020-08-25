from datetime import datetime
import numpy as np
import os
import random
import sys
import time

from numba import cuda
import rospy
from scipy.cluster.vq import kmeans2 as kmeans
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.vae.vae_rollout_server import VAERolloutServer


from datetime import datetime
import numpy as np
import os
import random
import sys
import time

from numba import cuda
import rospy
from scipy.cluster.vq import kmeans2 as kmeans
from scipy.stats import multivariate_normal
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.agent_env_wrapper import AgentEnvWrapper
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.vae.vae import VAE
from policy_hooks.vae.vae_rollout_server import VAERolloutServer


class DummyPolicy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call

    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


class VAETampRolloutServer(VAERolloutServer):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        np.random.seed(int(time.time()/100000))
        topic = hyperparams.get('topic', '')
        rospy.init_node('vae_rollout_server_'+str(self.id)+'{0}'.format(topic))
        self.mcts = hyperparams['mcts'][0]
        self.num_samples = 1
        self.rollout_len = hyperparams['rollout_len']
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.env = AgentEnvWrapper(agent=self.agent, use_solver=True)
        self.mcts.num_samples = self.num_samples
        self.mcts.agent = self.env
        self.num_rollouts = hyperparams.get('num_rollouts', 100)
        self.rollout_counter = 0
        self.stopped = False

        self.updaters = {}
        self.updaters['vae'] = rospy.Publisher('vae_update{0}'.format(topic), VAEUpdate, queue_size=5)
        hyperparams['vae']['load_data'] = False
        self.vae = VAE(hyperparams['vae'])
        self.weights_to_store = {}

        self.weight_subscriber = rospy.Subscriber('vae_weights{0}'.format(topic), UpdateTF, self.store_weights, queue_size=1, buff_size=2**22)
        self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)

        self.prior = multivariate_normal

        seed = int(1000*time.time()) % 1000
        np.random.seed(seed)
        random.seed(seed)
