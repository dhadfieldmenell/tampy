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
from policy_hooks.sample_list import SampleList

from policy_hooks.algorithm_mdgps import AlgorithmMDGPS
from policy_hooks.utils.policy_solver_utils import OBJ_ENUM, STATE_ENUM, TARG_ENUM, ACTION_ENUM


LOGGER = logging.getLogger(__name__)
RETIME = True


class AlgorithmIMPGPS(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        self.task = hyperparams['task']
        self.her = hyperparams.get('her', False)
        self.rollout_opt = hyperparams.get('rollout_opt', False)
        self.fail_value = hyperparams['fail_value']
        self.traj_centers = hyperparams['n_traj_centers']
        self.use_centroids = hyperparams['use_centroids']
        self.mp_opt = hyperparams.get('mp_opt', False)

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
            if opt_s is not None:
                all_opt_samples.append(SampleList([opt_s]))
                individual_opt_samples.append(opt_s)
                all_samples.append(opt_s)
                for s in s_list:
                    s.set_ref_X(opt_s.get_ref_X())
                    s.set_ref_U(opt_s.get_ref_U())
            if not len(s_list): continue
            # s_list.append(opt_s)
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

        '''
        start_t = time.time()
        try:
            self._update_prior(self.policy_prior, SampleList(all_samples))
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print(SampleList(all_samples).get_X())
            print('Failed to update policy prior, alg iteration continuing')

        try:
            self._update_prior(self.mp_policy_prior, SampleList(all_samples))
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print('Failed to update mp policy prior, alg iteration continuing')
        # print('Time to update priors', time.time() - start_t)
        '''

        '''
        # if len(sample_lists) and self.traj_centers >= len(sample_lists[0]):
        if not len(sample_lists) or self.traj_centers >= len(sample_lists[0]):
            if len(self.cur) != len(all_opt_samples) or reset:
                self.set_conditions(len(all_opt_samples))
            for m in range(len(self.cur)):
                self.cur[m].sample_list = all_opt_samples[m]

            self.T = all_opt_samples[0][0].T
            self._update_policy_no_cost()
            return all_opt_samples
        '''

        if len(self.cur) != len(sample_lists) or reset:
            self.set_conditions(len(sample_lists))

        if self.mp_opt:
            train_data = []
            for m in range(len(self.cur)):
                sample = sample_lists[m][0]
                pol_info = self.cur[m].pol_info
                assert not np.any(np.isnan(sample.get_obs()))
                try:
                    inf_f = None # (pol_info.pol_K, pol_info.pol_k, np.linalg.inv(pol_info.pol_S))
                except:
                    inf_f = None

                start_t = time.time()
                if not sample.opt_suc and self.agent.postcond_cost(sample) > 0:
                    sample.node = None
                    sample.next_sample = None
                    # new_sample = copy.deepcopy(sample)
                    # new_sample.agent = self.agent
                    # attr_dict = self.agent.relabel_traj(new_sample)
                    init_x = sample.get_X()
                    tol = 1e1
                    out, failed, success = self.agent.solve_sample_opt_traj(sample.get_X(t=0), sample.task, sample.condition, sample.get_X(), inf_f, t_limit=5, n_resamples=1, out_coeff=tol, smoothing=True, targets=sample.targets)

                    '''
                    if self.agent.precond_cost(sample) < 1e-2:
                        new_out, new_failed, new_success = self.agent.solve_sample_opt_traj(sample.get_X(t=0), sample.task, sample.condition, sample.get_X(), inf_f, t_limit=5, n_resamples=1, out_coeff=1e2, smoothing=True, targets=new_sample.targets, attr_dict=attr_dict)
                        for t in range(sample.T):
                            new_sample.set(ACTION_ENUM, new_out.get(ACTION_ENUM, t), t)
                    '''
                # print('Time in quick solve:', time.time() - start_t)

                    self.cur[m].sample_list = []
                    '''
                    if self.her:
                        new_sample = copy.deepcopy(sample)
                        new_sample.agent = self.agent
                        attr_dict = self.agent.relabel_traj(new_sample)
                        if self.agent.precond_cost(new_sample) < 1e-2:
                            self.cur[m].sample_list.append(new_sample)
                    '''
                    if not np.any(np.isnan(out.get_U())):
                        if self.rollout_opt:
                            self.cur[m].sample_list.append(out)
                        else:
                            for t in range(sample.T):
                                sample.set(ACTION_ENUM, out.get(ACTION_ENUM, t), t)
                            self.cur[m].sample_list.append(sample)
                else:
                    self.cur[m].sample_list = [sample]
            # individual_opt_samples.extend(train_data)
            return self._update_policy_no_cost(individual_opt_samples)


        start_t = time.time()
        for m in range(len(self.cur)):
            if type(sample_lists[m]) is list:
                sample_lists[m] = SampleList(sample_lists[m])
            self.cur[m].sample_list = sample_lists[m]
            if len(sample_lists[m]) > 1:
                self._eval_cost(m)
            else:
                individual_opt_samples.extend(sample_lists[m])
                self.cur[m].sample_list = []
        # print('Time to eval cost', time.time() - start_t)

        # Update dynamics linearizations.
        # self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        start_t = time.time()
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(len(self.cur))
            ]
            self._update_policy(individual_opt_samples)
            # print('Time to update policy', time.time() - start_t)

        # Update policy linearizations.
        # for m in range(len(self.cur)):
        #     self._update_policy_fit(m)

        # C-step
        try:
            start_t = time.time()
            self._update_trajectories()
            # print('Time to update trajs', time.time() - start_t)

            # S-step
            self._update_policy(individual_opt_samples)

            # Prepare for next iteration
            self._advance_iteration_variables()
        except:
            self._update_policy(individual_opt_samples)
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
        prior.update(samples, self.local_policy_opt, mode, self.task)


    def _update_policy_no_cost(self, optimal_samples=[], label='optimal', inv_cov=None):
        """ Compute the new policy. """
        if not len(self.cur) and not len(optimal_samples): return

        dU, dO = self.dU, self.dO
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dU))
        tgt_prc, tgt_wt = np.zeros((0, dU, dU)), np.zeros((0,))
        prim_obs_data = np.zeros((0, self.dPrim))

        for sample in optimal_samples:
            prc = np.zeros((1, sample.T, dU, dU))
            wt = np.zeros((sample.T,))

            m = 0
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            for t in range(len(sample.use_ts)):
                if inv_cov is None:
                    prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :], [1, 1, 1])
                else:
                    prc[:, t, :, :] = np.tile(inv_cov, [1, 1, 1])
                wt[t] = self._hyperparams['opt_wt'] * sample.use_ts[t]

            tgt_mu = np.concatenate((tgt_mu, sample.get_U()))
            obs_data = np.concatenate((obs_data, sample.get_obs()))
            prim_obs_data = np.concatenate((prim_obs_data, sample.get_prim_obs()))
            tgt_wt = np.concatenate((tgt_wt, wt))
            tgt_prc = np.concatenate((tgt_prc, prc.reshape(-1, dU, dU)))


        for m in range(len(self.cur)):
            samples = self.cur[m].sample_list
            if samples is None: continue
            for sample in samples:
                mu = np.zeros((1, data_len, dU))
                prc = np.zeros((1, data_len, dU, dU))
                wt = np.zeros((1, data_len))
                obs = np.zeros((1, data_len, dO))

                traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
                ts = np.random.choice(range(T), data_len, replace=False)
                ts.sort()
                for t in range(data_len):
                    if inv_cov is None:
                        prc[:, i, :, :] = np.tile(traj.inv_pol_covar[ts[t], :, :], [1, 1, 1])
                    else:
                        prc[:, i, :, :] = np.tile(inv_cov, [1, 1, 1])
                    wt[:,t] = 1. * sample.use_ts[ts[t]]

                for i in range(data_len):
                    t = ts[i]
                    mu[0, i, :] = sample.get_U(t=t)
                    obs[0, i, :] = sample.get_obs(t=t)
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))
                prim_obs_data = np.concatenate((prim_obs_data, sample.get_prim_obs()))
        obs_data = obs_data.reshape((-1, dO))
        tgt_mu = tgt_mu.reshape((-1, dU))
        tgt_wt = tgt_wt.reshape((-1, 1))
        tgt_prc = tgt_prc.reshape((-1, dU, dU))
        if len(tgt_mu):
            self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, self.task, label)
        else:
            print('Update no cost called with no data.')
