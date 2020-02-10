""" This file defines the PIGPS algorithm. 

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy
import logging
import sys
import time
import traceback

import numpy as np

from scipy.cluster.vq import kmeans2 as kmeans

from gps.algorithm.config import ALG_PIGPS
from gps.sample.sample_list import SampleList

from policy_hooks.algorithm_mdgps import AlgorithmMDGPS
from policy_hooks.utils.policy_solver_utils import OBJ_ENUM, STATE_ENUM, TARG_ENUM


LOGGER = logging.getLogger(__name__)


class AlgorithmIMPGPS(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        self.task = hyperparams['task']
        self.fail_value = hyperparams['fail_value']
        self.traj_centers = hyperparams['n_traj_centers']
        self.use_centroids = hyperparams['use_centroids']

        policy_prior = hyperparams['policy_prior']
        mp_policy_prior = hyperparams['mp_policy_prior']

        # MDGPS uses a prior per-condition; IMGPS wants one per task as well
        self.policy_prior = policy_prior['type'](policy_prior)
        self.mp_policy_prior = mp_policy_prior['type'](mp_policy_prior)

        AlgorithmMDGPS.__init__(self, config)

    def iteration(self, samples_with_opt, reset=True):
        all_opt_samples = []
        individual_opt_samples = []
        sample_lists = []
        all_samples = []
        for opt_s, s_list in samples_with_opt:
            all_opt_samples.append(SampleList([opt_s]))
            individual_opt_samples.append(opt_s)
            all_samples.append(opt_s)
            for s in s_list:
                s.set_ref_X(opt_s.get_ref_X())
                s.set_ref_U(opt_s.get_ref_U())
            if not len(s_list): continue
            sample_lists.append(s_list)
            all_samples.extend(s_list)

        # for m in range(len(self.cur)):
        #     s = sample_lists[m][0]
        #     for t in range(s.T):
        #         if np.any(np.isnan(s.get_U(t))):
        #             raise Exception('nans in samples at time {0} out of {1}'.format(t, s.T))

        # for s in individual_opt_samples:
        #     if np.any(np.isnan(s.get_U())):
        #         raise Exception('nans in opt samples')

        # if len(self.cur) != len(all_opt_samples) or reset:
        #     self.set_conditions(len(all_opt_samples))

        start_t = time.time()
        try:
            self._update_prior(self.policy_prior, SampleList(all_samples))
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print('Failed to update policy prior, alg iteration continuing')

        try:
            self._update_prior(self.mp_policy_prior, SampleList(all_samples))
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print('Failed to update mp policy prior, alg iteration continuing')
        print('Time to update priors', time.time() - start_t)

        # if len(sample_lists) and self.traj_centers >= len(sample_lists[0]):
        if not len(sample_lists) or self.traj_centers >= len(sample_lists[0]):
            if len(self.cur) != len(all_opt_samples) or reset:
                self.set_conditions(len(all_opt_samples))
            for m in range(len(self.cur)):
                self.cur[m].sample_list = all_opt_samples[m]

            self.T = all_opt_samples[0][0].T
            self._update_policy_no_cost()
            return all_opt_samples

        if len(self.cur) != len(sample_lists) or reset:
            self.set_conditions(len(sample_lists))


        start_t = time.time()
        for m in range(len(self.cur)):
            if type(sample_lists[m]) is list:
                sample_lists[m] = SampleList(sample_lists[m])
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)
        print('Time to eval cost', time.time() - start_t)

        # Update dynamics linearizations.
        # self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        start_t = time.time()
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(len(self.cur))
            ]
            self._update_policy(individual_opt_samples)
            print('Time to update policy', time.time() - start_t)

        # Update policy linearizations.
        # for m in range(len(self.cur)):
        #     self._update_policy_fit(m)

        # C-step
        start_t = time.time()  
        self._update_trajectories()
        print('Time to update trajs', time.time() - start_t)

        # S-step
        self._update_policy(individual_opt_samples)

        # Prepare for next iteration
        self._advance_iteration_variables()

        return sample_lists


    def preiteration_step(self, sample_lists):
        if not self.local_policy_opt.policy_initialized(self.task): return
        if self.M != len(sample_lists):
            self.set_conditions(len(sample_lists))
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._update_policy_fit(m)


    def _update_prior(self, prior, samples):
        if not self.local_policy_opt.policy_initialized(self.task): return
        mode = self._hyperparams['policy_sample_mode']
        try:
            prior.update(samples, self.local_policy_opt, mode, self.task)
        except Exception as e:
            print('Policy prior update threw exception: ', e, '\n')


    def _update_policy_no_cost(self):
        """ Compute the new policy. """
        print('Calling update policy without PI^2')
        dU, dO, T = self.dU, self.dO, self.T
        data_len = int(self.sample_ts_prob * T)
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, data_len, dO)), np.zeros((0, data_len, dU))
        tgt_prc, tgt_wt = np.zeros((0, data_len, dU, dU)), np.zeros((0, data_len))
        
        # # Optimize global policies with optimal samples as well
        # for sample in optimal_samples:
        #     mu = np.zeros((1, data_len, dU))
        #     prc = np.zeros((1, data_len, dU, dU))
        #     wt = np.zeros((1, data_len))
        #     obs = np.zeros((1, data_len, dO))

        #     ts = np.random.choice(xrange(T), data_len, replace=False)
        #     ts.sort()
        #     for t in range(data_len):
        #         prc[0,t] = 1e0 * np.eye(dU)
        #         wt[:,t] = self._hyperparams['opt_wt']

        #     for i in range(data_len):
        #         t = ts[i]
        #         mu[0, i, :] = sample.get_U(t=t)
        #         obs[0, i, :] = sample.get_obs(t=t)
        #     tgt_mu = np.concatenate((tgt_mu, mu))
        #     tgt_prc = np.concatenate((tgt_prc, prc))
        #     tgt_wt = np.concatenate((tgt_wt, wt))
        #     obs_data = np.concatenate((obs_data, obs))

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
                    wt[:,t] = self._hyperparams['opt_wt'] # * sample.use_ts[ts[t]]

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
        else:
            print('Update no cost called with no data.')
