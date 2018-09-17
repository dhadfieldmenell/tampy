""" This file defines the PIGPS algorithm. 

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy
import logging

import numpy as np

from gps.algorithm.config import ALG_PIGPS
from gps.sample.sample_list import SampleList

from policy_hooks.algorithm_mdgps import AlgorithmMDGPS
from policy_hooks.utils.policy_solver_utils import OBJ_ENUM, STATE_ENUM, TARG_ENUM

LOGGER = logging.getLogger(__name__)


class AlgorithmOPTGPS(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams, task):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        self.task = task
        self.fail_value = hyperparams['fail_value']
        AlgorithmMDGPS.__init__(self, config)

    def iteration(self, sample_lists, optimal_samples, reset=True):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # if not len(self.cur) or self.replace_conds:
        #     self.set_conditions(len(sample_lists))
        if reset:
            self.set_conditions(len(sample_lists))

        # Store the samples and evaluate the costs.
        del_inds = []
        success= False
        all_opt_samples = []
        for m in range(len(self.cur)):
            opt_samples = []
            for sample in sample_lists[m]:
                agent = sample.agent
                obj = agent.plans.values()[0].params[sample.obj]
                targ = agent.plans.values()[0].params[sample.targ]
                opt_sample, _, success = agent.sample_optimal_trajectory(sample.get_X(t=0), sample.task, sample.condition, traj_mean=sample.get(STATE_ENUM), fixed_targets=[obj, targ])
                if success:
                    opt_samples.append(opt_sample)
                else:
                    break
            if len(opt_samples):
                all_opt_samples.append(SampleList(opt_samples))

        self.set_conditions(len(all_opt_samples))
        for m in range(len(self.cur)):
            self.cur[m].sample_list = all_opt_samples[m]


        # Update dynamics linearizations.
        # self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        # if self.iteration_count == 0:
        #     self.new_traj_distr = [
        #         self.cur[cond].traj_distr for cond in range(len(self.cur))
        #     ]
        #     self._update_policy(optimal_samples)

        # Update policy linearizations.
        # for m in range(len(self.cur)):
        #     self._update_policy_fit(m)

        # # C-step        
        # self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        # self._advance_iteration_variables()

        return all_opt_samples

    def _update_policy(self, optimal_samples=[]):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        data_len = int(self.sample_ts_prob * T)
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, data_len, dO)), np.zeros((0, data_len, dU))
        tgt_prc, tgt_wt = np.zeros((0, data_len, dU, dU)), np.zeros((0, data_len))
        
        # Optimize global polciies with optimal samples as well
        for sample in optimal_samples:
            mu = np.zeros((1, data_len, dU))
            prc = np.zeros((1, data_len, dU, dU))
            wt = np.zeros((1, data_len))
            obs = np.zeros((1, data_len, dO))

            ts = np.random.choice(xrange(T), data_len, replace=False)
            ts.sort()
            for t in range(data_len):
                prc[0,t] = 1e0 * np.eye(dU)
                wt[:,t] = self._hyperparams['opt_wt']

            for i in range(data_len):
                t = ts[i]
                mu[0, i, :] = sample.get_U(t=t)
                obs[0, i, :] = sample.get_obs(t=t)
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, obs))

        for m in range(len(self.cur)):
            samples = self.cur[m].sample_list
            for sample in samples:
                mu = np.zeros((1, data_len, dU))
                prc = np.zeros((1, data_len, dU, dU))
                wt = np.zeros((1, data_len))
                obs = np.zeros((1, data_len, dO))

                ts = np.random.choice(xrange(T), data_len, replace=False)
                ts.sort()
                for t in range(data_len):
                    prc[0,t] = 1e0 * np.eye(dU)
                    wt[:,t] = self._hyperparams['opt_wt']

                for i in range(data_len):
                    t = ts[i]
                    mu[0, i, :] = sample.get_U(t=t)
                    obs[0, i, :] = sample.get_obs(t=t)
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, self.task)
