""" This file defines the PIGPS algorithm. 

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy

import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_PIGPS
from gps.sample.sample_list import SampleList


class AlgorithmTAMPGPS(AlgorithmPIGPS):
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        AlgorithmMDGPS.__init__(self, config)
        self.policy_transfer_coeff = self._hyperparams['policy_transfer_coeff']
        self.policy_scale_factor = self._hyperparams['policy_scale_factor']

    def iteration(self, sample_lists):
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
        #     self._eval_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        # if self.iteration_count == 0:
        #     self.new_traj_distr = [
        #         self.cur[cond].traj_distr for cond in range(self.M)
        #     ]
        #     self._update_policy()

        # Update policy linearizations.
        # for m in range(self.M):
        #     self._update_policy_fit(m)

        # C-step        
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

        self.policy_transfer_coeff *= self.policy_scale_factor

    def _update_trajectories(self):
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        # for cond in range(self.M):
        #     self.new_traj_distr[cond], self.cur[cond].eta = \
        #             self.traj_opt.update(cond, self)

    def _update_policy(self):
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t] = pol_info.pol_wt[t]
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def _update_policy_fit(self, m):
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        # X = samples.get_X()
        # obs = samples.get_obs().copy()
        # pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        # pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # # Update policy prior.
        # policy_prior = pol_info.policy_prior
        # samples = SampleList(self.cur[m].sample_list)
        # mode = self._hyperparams['policy_sample_mode']
        # policy_prior.update(samples, self.policy_opt, mode)

        # # Fit linearization and store in pol_info.
        # pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
        #         policy_prior.fit(X, pol_mu, pol_sig)
        # try:
        #     for t in range(T):
        #         pol_info.chol_pol_S[t, :, :] = \
        #                 sp.linalg.cholesky(pol_info.pol_S[t, :, :])
        # except:
        #     import ipdb; ipdb.set_trace()

        # Weight policy by sample costs
        pol_info.pol_wt = np.zeros((T, N))
        self._eval_cost(m)
        for t in range(T):
            exponent = -self.cur[m].cs[:, t]
            exp_cost = np.exp(exponent - np.max(exponent))
            pol_info.pol_wt[t] = exp_cost / np.sum(exp_cost)
