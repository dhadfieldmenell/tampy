import copy
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

import rospy
from std_msgs.msg import *

from policy_hooks.abstract_motion_plan_server import AbstractMotionPlanServer
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.baxter.sorting_prob import *
from policy_hooks.baxter.baxter_policy_solver import BaxterPolicySolver

from tamp_ros.msg import *
from tamp_ros.srv import *


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class BaxterMotionPlanServer(AbstractMotionPlanServer):
    def __init__(self, hyperparams):
        self.solver = BaxterPolicySolver(hyperparams)
        super(NAMOMotionPlanServer, self).__init__(hyperparams)

    def sample_optimal_trajectory(self, state, task_tuple, condition, traj_mean=[], fixed_targets=[]):
        exclude_targets = []
        success = False
        task = self.task_list[task_tuple[0]]

        targets = fixed_targets
        obj = targets[0]
        targ = targets[1]

        failed_preds = []
        iteration = 0

        start_time = time.time()
        while not success:
            iteration += 1

            plan = self.agent.plans[task, targets[0]]
            targets[0] = plan.params[targets[0]]
            targets[1] = plan.params[targets[1]]
            obj, targ = targets
            set_params_attrs(plan.params, plan.state_inds, state, 0)

            for param_name in plan.params:
                param = plan.params[param_name]
                if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                    plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

            for target in self.targets[condition]:
                plan.params[target].value[:,0] = self.targets[condition][target]

            if targ.name in self.targets[condition]:
                plan.params['{0}_end_target'.format(obj.name)].value[:,0] = self.targets[condition][targ.name]

            if task == 'grasp':
                plan.params[targ.name].value[:,0] = plan.params[obj.name].pose[:,0]

            plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
            plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
            plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
            plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
            # self.env.SetViewer('qtcoin')
            # success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean, task=(self.task_list.index(task), self.obj_list.index(obj.name), self.targ_list.index(targ.name)))
            try:
                self.solver.save_free(plan)
                success = self.solver._backtrack_solve(plan, n_resamples=3, traj_mean=traj_mean, task=task_tuple)
                # viewer = OpenRAVEViewer._viewer if OpenRAVEViewer._viewer is not None else OpenRAVEViewer(plan.env)
                # if task == 'putdown':
                #     import ipdb; ipdb.set_trace()
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
            except Exception as e:
                traceback.print_exception(*sys.exc_info())
                self.solver.restore_free(plan)
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
                success = False

            failed_preds = []
            for action in plan.actions:
                try:
                    failed_preds += [(pred.get_type(), targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
                except:
                    pass
            exclude_targets.append(targets[0].name)

            if len(fixed_targets):
                break

        if len(failed_preds):
            success = False
        else:
            success = True

        output_traj = np.zeros((plan.horizon, self.agent.dX))
        for t in range(plan.horizon):
            fill_vector(plan.params, self.agent.state_inds, output_traj[t], t)

        end_time = time.time()
        if self.log_timing:
            self.update_timing_info(start_time-end_time)

        return output_traj[:,:self.agent.symbolic_bound], failed_preds, success
