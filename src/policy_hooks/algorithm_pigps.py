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
from policy_hooks.utils.policy_solver_utils import OBJ_ENUM, STATE_ENUM, TARG_ENUM

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
        for m in range(len(self.cur)):
            if not np.any(sample_lists[m][0].get_ref_X()):
                s = sample_lists[m][0]
                obj = list(s.agent.plans.values())[0].params[s.agent.obj_list[np.argmax(s.get(OBJ_ENUM, t=0))]]
                targ = list(s.agent.plans.values())[0].params[s.agent.targ_list[np.argmax(s.get(TARG_ENUM, t=0))]]

                traj_mean = np.sum([sample.get_U() for sample in sample_lists[m]], axis=0) / len(sample_lists[m])
                opt_sample, _, success = s.agent.sample_optimal_trajectory(sample_lists[m][0].get_X(t=0), sample_lists[m][0].task, sample_lists[m][0].condition, traj_mean=traj_mean, fixed_targets=[obj, targ])
                if not success:
                    for sample in sample_lists[m]:
                        sample.task_cost = self.fail_value
                    del_inds.append(m)
                    continue
                sample_lists[m]._samples.append(opt_sample)
                for sample in sample_lists[m]:
                    sample.set_ref_X(opt_sample.get(STATE_ENUM))
                    sample.set_ref_U(opt_sample.get_U())
                    if not np.any(sample.get_ref_X()):
                        import ipdb; ipdb.set_trace()
            if m not in del_inds:
                self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        del_inds.reverse()
        for i in del_inds:
            del self.cur[i]

        sample_list = []
        for m in range(len(self.cur)):
            if not len(self.cur[m].sample_list):
                import ipdb; ipdb.set_trace()
            sample_list.append(self.cur[m].sample_list)

        # Update dynamics linearizations.
        # self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        # if self.iteration_count == 0:
        #     self.new_traj_distr = [
        #         self.cur[cond].traj_distr for cond in range(len(self.cur))
        #     ]
        #     self._update_policy(optimal_samples)
        # print sample_list[0][0].get_ref_X()
        # print sample_list[0][0].get_ref_U()
        # import ipdb; ipdb.set_trace()

        # Update policy linearizations.
        for m in range(len(self.cur)):
            self._update_policy_fit(m)

        # C-step
        self._update_trajectories()

        # S-step
        self._update_policy(optimal_samples)

        # Prepare for next iteration
        # self._advance_iteration_variables()

        return sample_list
