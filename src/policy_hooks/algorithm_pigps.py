""" This file defines the PIGPS algorithm. 

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy
import logging

from policy_hooks.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.config import ALG_PIGPS

LOGGER = logging.getLogger(__name__)


class AlgorithmPIGPS(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PIGPS)
        config.update(hyperparams)
        AlgorithmMDGPS.__init__(self, config)

    def iteration(self, sample_lists):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            for ts in self.cur[m]:
                self.cur[m][ts].sample_list = sample_lists[m][ts]
                self._eval_cost(m, ts)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                {ts: self.cur[cond][ts].traj_distr for ts in self.cur[cond]} for cond in range(self.M)
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit(m)

        # C-step        
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()