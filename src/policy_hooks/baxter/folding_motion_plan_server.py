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
from policy_hooks.baxter.folding_prob import *
from policy_hooks.baxter.baxter_policy_solver import BaxterPolicySolver

from tamp_ros.msg import *
from tamp_ros.srv import *


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class FoldingMotionPlanServer(AbstractMotionPlanServer):
    def __init__(self, hyperparams):
        self.solver = BaxterPolicySolver(hyperparams)
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        super(FoldingMotionPlanServer, self).__init__(hyperparams)

    def sample_optimal_trajectory(self, mp_state, task_tuple, condition, traj_mean=[]):
        exclude_targets = []
        success = False
        task = self.task_list[task_tuple[0]]

    
        failed_preds = []
        iteration = 0

        start_time = time.time()
        while not success:
            iteration += 1

            plan = self.agent.plans[task_tuple] 
            set_params_attrs(plan.params, plan.state_inds, mp_state, 0)
            free_attrs = plan.get_free_attrs()

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
                plan.store_free_attrs(free_attrs)
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

    def publish_hl_plan(self, msg):
        if msg.solver_id != self.id: return
        paths = []
        failed = []
        new_failed = []
        stop = False
        attempt = 0
        cur_sample = None
        cond = msg.cond
        opt_hl_plan = []
        cur_path = []
        cur_state = np.array(msg.init_state)

        try:
            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
        except:
            hl_plan = []

        last_reset = 0
        while not stop and attempt < 4 * len(self.agent.obj_list):
            last_reset += 1
            for step in hl_plan:
                targets = [self.agent.plans.values()[0].params[p_name] for p_name in step[1]]
                plan = self.config['plan_f'](step[0], targets)
                if len(targets) < 2:
                    targets.append(plan.params['{0}_end_target'.format(targets[0].name)])
                next_sample, new_failed, success = self.agent.solve_sample_opt_traj(cur_state, step[0], cond, fixed_targets=targets)
                next_sample.success = FAIL_LABEL
                if not success:
                    if last_reset > 5:
                        failed = []
                        last_reset = 0
                    next_sample.success = FAIL_LABEL
                    if not len(new_failed):
                        stop = True
                    else:
                        failed.extend(new_failed)
                        try:
                            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
                        except:
                            hl_plan = []
                        # attempt += 1
                    break

                cur_path.append(next_sample)
                cur_sample = next_sample
                cur_state = cur_sample.get_X(t=cur_sample.T-1)
                opt_hl_plan.append(step)

            if self.config['goal_f'](cur_state, self.agent.targets[cond], self.agent.plans.values()[0]) == 0:
                for sample in cur_path:
                    sample.success = SUCCESS_LABEL
                break

            attempt += 1

        resp = HLPlanResult()
        steps = []
        for sample in cur_path:
            mp_step = MotionPlanResult()

            mp_step.traj = []
            out = sample.get(STATE_ENUM)
            for t in range(len(out)):
                next_line = Float32MultiArray()
                next_line.data = out[t]
                mp_step.traj.append(next_line)
            mp_step.failed = ''
            mp_step.success = True
            mp_step.plan_id = -1
            mp_step.cond = msg.cond
            mp_step.task = sample.task
            mp_step.obj = sample.obj
            mp_step.targ = sample.targ

            steps.append(mp_step)
        resp.steps = steps
        resp.path_to = msg.path_to
        resp.success = len(cur_path) and cur_path[0].success == SUCCESS_LABEL
        resp.cond = msg.cond
        self.hl_publishers[msg.server_id].publish(resp)
