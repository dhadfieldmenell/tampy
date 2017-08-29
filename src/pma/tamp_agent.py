""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import pma.policy_solver_utils as utils


class TAMPAgent(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        # Store samples, along with size/index information for samples.
        self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['sensor_dims'][utils.ACTION_ENUM]

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        if 'meta_include' in self._hyperparams:
            self.meta_data_types = self._hyperparams['meta_include']
        else:
            self.meta_data_types = []

        # List of indices for each data type in state X.
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i

        # List of indices for each data type in observation.
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i

        # List of indices for each data type in meta data.
        self._meta_idx, i = [], 0
        for sensor in self.meta_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._meta_idx.append(list(range(i, i+dim)))
            i += dim
        self.dM = i

        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}
        self._meta_data_idx = {d: i for d, i in zip(self.meta_data_types,
                                                   self._meta_idx)}


    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        sample = Sample(self)
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        utils.reset_action(self.action, self.state_inds, self.x0[condition])
        utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[0, :], 0, self.dX)
        active_ts = self.action.active_ts
        for t in range(active_ts[0], active_ts[1]+1):
            utils.fill_trajectory_ts_from_policy(policy, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0]], t, self.dX)
            utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0], :], t, self.dX)
        if save:
            self._samples[condition].append(sample)
        return sample
