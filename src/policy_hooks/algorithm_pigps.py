""" This file defines the PIGPS algorithm. 

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy
import logging

import numpy as np

from gps.algorithm.config import ALG_PIGPS

from policy_hooks.algorithm_mdgps import AlgorithmMDGPS
from policy_hooks.utils.policy_solver_utils import STATE_ENUM

LOGGER = logging.getLogger(__name__)


class AlgorithmPIGPS(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams, task):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        self.task = task
        AlgorithmMDGPS.__init__(self, config)

    def iteration(self, sample_lists, optimal_samples):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # if not len(self.cur) or self.replace_conds:
        #     self.set_conditions(len(sample_lists))
        self.set_conditions(len(sample_lists))

        # Store the samples and evaluate the costs.
        del_inds = []
        for m in range(len(self.cur)):
            self.cur[m].sample_list = sample_lists[m]
            if not np.any(sample_lists[m][0].get_ref_X()):
                traj_mean = np.sum([sample.get_U() for sample in sample_lists[m]], axis=0) / len(sample_lists[m])
                opt_sample, _ = self.agent.sample_optimal_trajectory(sample_lists[m][0].get_X(t=0), sample_lists[m][0].task, sample_lists[m][0].condition, traj_mean=traj_mean)
                if opt_sample is None:
                    del_inds.append(m)
                    continue
                self.cur[m].sample_list._samples.append(opt_sample)
                for sample in self.cur[m].sample_list:
                    sample.set_ref_X(opt_sample.get(STATE_ENUM))
                    sample.set_ref_U(opt_sample.get_U())
            self._eval_cost(m)

        for i in range(len(del_inds)):
            ind = del_inds[i]
            del self.cur[ind - i]

        if not len(self.cur):
            return

        # Update dynamics linearizations.
        # self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(len(self.cur))
            ]
            self._update_policy(optimal_samples)

        # Update policy linearizations.
        # for m in range(len(self.cur)):
        #     self._update_policy_fit(m)

        # C-step        
        self._update_trajectories()

        # S-step
        self._update_policy(optimal_samples)

        # Prepare for next iteration
        # self._advance_iteration_variables()

