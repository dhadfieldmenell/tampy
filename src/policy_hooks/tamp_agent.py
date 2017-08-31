""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import policy_hooks.policy_solver_utils as utils


class TAMPAgent(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        self.plans = self._hyperparams['plans']
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.solver = self._hyperparams['solver']
        self.x0 = self._hyperparams['x0']

        Agent.__init__(self, config)


    # TODO: Fill this in with proper behavior
    def sample(self, policy, condition, verbose=False, save=True, noisy=False):
        sample = Sample(self)
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        utils.reset_plan(self.plans[condition], self.state_inds, self.x0[condition])
        #TODO: Enforce this sample is close to the global policy
        self.solver.solve(self.plans[condition], n_resamples=5, active_ts=self.action.active_timesteps force_init=True)
        utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[0, :], 0, self.dX)
        active_ts = self.action.active_ts
        for t in range(active_ts[0], active_ts[1]+1):
            utils.fill_trajectory_ts_from_policy(policy, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0]], t, self.dX)
            utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0], :], t, self.dX)
        if save:
            self._samples[condition].append(sample)
        return sample
