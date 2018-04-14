""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo, PolicyInfo
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList

LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        self.set_conditions()

        if self._hyperparams['policy_opt']['prev'] is None:
            self.policy_opt = self._hyperparams['policy_opt']['type'](
                self._hyperparams['policy_opt'], self.dO, self.dU
            )
        else:
            self.policy_opt = self._hyperparams['policy_opt']['prev']

        self.task = self._hyperparams['task']
        self.task_breaks = self._hyperparams['task_breaks']

    def set_conditions(self):
        self.cur = [{} for _ in range(self.M)]
        self.prev = [{} for _ in range(self.M)]
        policy_prior = self._hyperparams['policy_prior']

        for m in range(len(self.M)):
            cur_t = 0
            for next_t, task in task_breaks:
                if task == self.task:
                    self.cur[m][cur_t] = IterationData()
                    self.prev[m][cur_t] = IterationData()
                    self.cur[m][cur_t].traj_info = TrajectoryInfo()
                    if self._hyperparams['fit_dynamics']:
                        dynamics = self._hyperparams['dynamics']
                        self.cur[m][cur_t].traj_info.dynamics = dynamics['type'](dynamics)

                    init_traj_distr = extract_condition(
                        self._hyperparams['init_traj_distr'], self._cond_idx[m]
                    )

                    self.cur[m][cur_t].traj_distr = init_traj_distr['type'](init_traj_distr)

                    self.cur[m][cur_ts].pol_info = PolicyInfo(self._hyperparams)
                    self.cur[m][cur_ts].pol_info.policy_prior = policy_prior['type'](policy_prior)

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            for ts in self.cur[m]:
                self.cur[m][ts].sample_list = sample_lists[ts]
                self._eval_cost(m, ts)

        # Update dynamics linearizations.
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                {ts: self.cur[cond][ts].traj_distr for ts in self.cur[cond]}for cond in range(self.M)
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            for ts in self.cur[m]:
                samples = self.cur[m][ts].sample_list
                X = samples.get_X()
                N = len(samples)
                traj, pol_info = self.new_traj_distr[m][ts], self.cur[m][ts].pol_info
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
                    wt[:, t].fill(pol_info.pol_wt[t])
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, self.task)

    # def _update_policy(self):
    #     """ Compute the new policy. """
    #     obs_data = {}
    #     tgt_mu = {}
    #     tgt_prc = {}
    #     tgt_wt = {}
    #     dU, dO = self.dU, self.dO
    #     for task in self.task_list:
    #         # Compute target mean, cov, and weight for each sample.
    #         T = self.task_durations[task]
    #         obs_data[task], tgt_mu[task] = np.zeros((0, T, dO)), np.zeros((0, T, dU))
    #         tgt_prc[task], tgt_wt[task] = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        
    #     for m in range(self.M):
    #         T = self.agent.Ts[m]
    #         samples = self.cur[m].sample_list
    #         task_breaks = self.agent.task_breaks[m]
    #         X = samples.get_X()
    #         N = len(samples)
    #         traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
    #         mu = np.zeros((N, T, dU))
    #         prc = np.zeros((N, T, dU, dU))
    #         wt = np.zeros((N, T))
    #         # Get time-indexed actions.
    #         for t in range(T):
    #             # Compute actions along this trajectory.
    #             prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
    #                                       [N, 1, 1])
    #             for i in range(N):
    #                 mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
    #             wt[:, t].fill(pol_info.pol_wt[t])

    #         obs = samples.get_obs()
    #         for task, ts in task_breaks:
    #             tgt_mu[task] = np.concatenate((tgt_mu[task], mu[ts[0]:ts[1], :]))
    #             tgt_prc[task] = np.concatenate((tgt_prc[task], prc[ts[0]:ts[1], :, :]))
    #             tgt_wt[task] = np.concatenate((tgt_wt[task], wt[ts[0]:ts[1]]))
    #             obs_data[task] = np.concatenate((obs_data[task], obs[ts[0]:ts[1], :]))

    #     for task in self.task_list:
    #         if len(obs_data):
    #             self.policy_opt.update(obs_data[task], tgt_mu[task], tgt_prc[task], tgt_wt[task], task)


    def _update_policy_fit(self, m):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        for ts in self.cur[m][ts]:
            samples = self.cur[m][ts].sample_list
            N = len(samples)
            pol_info = self.cur[m][ts].pol_info
            X = samples.get_X()
            obs = samples.get_obs().copy()
            pol_mu, pol_sig = self.policy_opt.prob(obs, self.task)[:2]
            pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

            # Update policy prior.
            policy_prior = pol_info.policy_prior
            samples = SampleList(self.cur[m][ts].sample_list)
            mode = self._hyperparams['policy_sample_mode']
            policy_prior.update(samples, self.policy_opt, mode)

            # Fit linearization and store in pol_info.
            pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                    policy_prior.fit(X, pol_mu, pol_sig)
            for t in range(T):
                pol_info.chol_pol_S[t, :, :] = \
                        sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _eval_cost(self, cond, ts):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond][ts].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond][ts].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond][ts].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond][ts].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond][ts].cs = cs  # True value of cost.

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                {ts: self.cur[cond].traj_distr for ts in self.cur[cond]} for cond in range(self.M)
            ]

        for cond in range(self.M):
            for ts in self.cur[m]:
                self.new_traj_distr[cond][ts], self.cur[cond][ts].eta = \
                        self.traj_opt.update(cond, ts, self)

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for m in range(self.M):
            for ts in self.cur[m]:
                cur_data = self.cur[m][ts].sample_list
                X = cur_data.get_X()
                U = cur_data.get_U()

                # Update prior and fit dynamics.
                self.cur[m][ts].traj_info.dynamics.update_prior(cur_data)
                self.cur[m][ts].traj_info.dynamics.fit(X, U)

                # Fit x0mu/x0sigma.
                x0 = X[:, 0, :]
                x0mu = np.mean(x0, axis=0)
                self.cur[m][ts].traj_info.x0mu = x0mu
                self.cur[m][ts].traj_info.x0sigma = np.diag(
                    np.maximum(np.var(x0, axis=0),
                               self._hyperparams['initial_state_var'])
                )

                prior = self.cur[m][ts].traj_info.dynamics.get_prior()
                if prior:
                    mu0, Phi, priorm, n0 = prior.initial_state()
                    N = len(cur_data)
                    self.cur[m][ts].traj_info.x0sigma += \
                            Phi + (N*priorm) / (N+priorm) * \
                            np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        Algorithm._advance_iteration_variables(self)
        for m in range(self.M):
            for ts in self.cur[m]:
                self.cur[m][ts].traj_info.last_kl_step = \
                        self.prev[m][ts].traj_info.last_kl_step
                self.cur[m][ts].pol_info = copy.deepcopy(self.prev[m][ts].pol_info)

    def _stepadjust(self):
        """
        Calculate new step sizes. This version uses the same step size
        for all conditions.
        """
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev) # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(
                    prev_nn, self.prev[m].traj_info
            ).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(
                    prev_lg, self.prev[m].traj_info
            ).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = self.traj_opt.estimate_cost(
                    cur_nn, self.cur[m].traj_info
            ).sum()
            cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('Previous cost: Laplace: %f, MC: %f',
                     prev_laplace, prev_mc)
        LOGGER.debug('Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('Actual cost: Laplace: %f, MC: %f',
                     cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv
