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


class VAERolloutServer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        np.random.seed(int(time.time()/100000))
        topic = hyperparams.get('topic', '')
        rospy.init_node('vae_rollout_server_'+str(self.id)+'{0}'.format(topic))
        self.mcts = hyperparams['mcts']
        self.num_samples = 1
        self.rollout_len = hyperparams['rollout_len']
        sub_env = hyperparams['env']()
        self.env = AgentEnvWrapper(env=sub_env, agent=None, use_solver=False)
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


    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def update(self, obs, task):
        msg = VAEUpdate()
        msg.obs = np.array(obs).flatten().tolist()
        msg.tasks = np.array(task).flatten().tolist()
        msg.dO = list(np.array(obs).shape)
        msg.dTask = list(np.array(task).shape)
        msg.n = 0
        self.updaters['vae'].publish(msg)


    def store_weights(self, msg):
        self.weights_to_store[msg.scope] = msg.data


    def update_weights(self):
        scopes = list(self.weights_to_store.keys())
        for scope in scopes:
            save = self.id.endswith('0')
            data = self.weights_to_store[scope]
            self.weights_to_store[scope] = None
            if data is not None:
                self.vae.deserialize_weights(data, save=save)


    def parse_state(self, xs):
        state_info = []
        params = list(self.agent.plans.values())[0].params
        for x in xs:
            info = []
            for param_name, attr in self.agent.state_inds:
                if params[param_name].is_symbol(): continue
                value = x[self.agent._x_data_idx[STATE_ENUM]][self.agent.state_inds[param_name, attr]]
                info.append((param_name, attr, value))
            state_info.append(info)
        return state_info


    def step(self):
        self.env.reset()

        obs_path = []
        task_path = []
        obs = self.env.get_obs()
        times_sampled = {}
        switch_to_stack = np.random.uniform() < 0.25
        stack_only = False
        for n in range(self.rollout_len):
            acts = set()
            for _ in range(100):
                next_act = self.env.action_space.sample()
                if type(next_act) is int:
                    acts.add(next_act)
                else:
                    acts.add(tuple(next_act))
            acts = list(acts)
            stack_only = stack_only or (switch_to_stack and np.random.uniform() > 0.5)
            if stack_only:
                for i in range(len(acts)):
                    act = list(acts[i])
                    act[0] = 0
                    acts[i] = tuple(act)

            encode_acts = list(acts)
            for i in range(len(acts)):
                encode_acts[i] = self.env.encode_action(acts[i])

            # latent_preds = self.vae.get_next_latents(np.tile(obs, [len(acts), 1, 1, 1]), np.array(acts))
            # p = np.array([1./self.prior.pdf(latent_preds[i],
            #                                 mean=np.zeros(len(latent_preds[i])),
            #                                 cov=np.ones(len(latent_preds[i])))
            #               for i in range(len(latent_preds))])

            temp = 1 # Softmax temperature
            # p = np.exp(-temp*self.vae.next_latents_kl_pentalty(np.tile(obs, [len(acts), 1, 1, 1]), np.array(encode_acts)))
            # p = p / np.sum(p)
            p = np.ones((len(acts)), dtype=np.float32) / len(acts)
            # ind = np.argmax(p)
            if np.any(np.isnan(p)):
                p = np.ones(len(acts), dtype=np.float64) / len(acts)
            ind = np.random.choice(list(range(len(acts))), p=p)
            act = acts[ind]
            obs_path.append(obs)
            task_path.append(self.env.encode_action(act))
            obs, _, _, _ = self.env.step(act)
        self.update(obs_path, task_path)
        self.rollout_counter += 1
        if self.rollout_counter > self.num_rollouts:
            self.env.reset_init_state()
            self.rollout_counter = 0


    def run(self):
        while not self.stopped:
            self.step()
            rospy.sleep(0.1)
