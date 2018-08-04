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

        self.set_conditions(len(sample_lists))
        # Store the samples and evaluate the costs.
        for m in range(len(self.cur)):
            try:
                self.cur[m].sample_list = sample_lists[m]
                self._eval_cost(m)
            except:
                import ipdb; ipdb.set_trace()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(len(self.cur))
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(len(self.cur)):
            self._update_policy_fit(m)

        # C-step        
        self._update_trajectories()

        # S-step
        self._update_policy(optimal_samples)

        # Prepare for next iteration
        self._advance_iteration_variables()

