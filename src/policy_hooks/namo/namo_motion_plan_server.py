import copy
import sys
import traceback

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

import rospy


# from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.namo_predicates import dsafe
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.namo.sorting_prob_2 import *


MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100

class NamoMotionPlanServer():
    def __init__(self, hyperparams):
        self.state_inds = hyperparams[state_inds]
        self.task_list = hyperparams[task_list]
        self.targets = hyperparams['targets']
        self.solver = NAMOPolicySolver()
        plans = {}
        env = None
        openrave_bodies = {}
        for task in self.task_list:
            for c in range(num_cans):
                plans[task, 'can{0}'.format(c)] = get_plan_for_task(task, ['can{0}'.format(c), 'can{0}_end_target'.format(c)], num_cans, env, openrave_bodies)
                if env is None:
                    env = plans[task, 'can{0}'.format(c)].env
                    for param in plans[task, 'can{0}'.format(c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body


    def sample_optimal_trajectory(self, state, task, condition, traj_mean=[], fixed_targets=[]):
        exclude_targets = []
        success = False

        targets = fixed_targets
        obj = fixed_targets[0]
        targ = fixed_targets[1]

        failed_preds = []
        iteration = 0
        while not success:
            iteration += 1

            plan = self.plans[task, targets[0]] 
            targets[0] = plan.params[targets[0]]
            targets[1] = plan.params[targets[1]]
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
            
            plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
            dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
            if task == 'putdown':
                plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]
            if task == 'grasp':
                plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist+0.2]
            # self.env.SetViewer('qtcoin')
            # success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean, task=(self.task_list.index(task), self.obj_list.index(obj.name), self.targ_list.index(targ.name)))
            try:
                self.solver.save_free(plan)
                success = self.solver._backtrack_solve(plan, n_resamples=3, traj_mean=traj_mean, task=(self.task_list.index(task), self.obj_list.index(obj.name), self.targ_list.index(targ.name)))
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
                    failed_preds += [(pred, targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
                except:
                    pass
            exclude_targets.append(targets[0].name)

            if len(fixed_targets):
                break

        if len(failed_preds):
            success = False
        else:
            success = True

        if not success:
            # import ipdb; ipdb.set_trace()
            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[self.task_list.index(task)] = 1.
            obj_vec = np.zeros((len(self.obj_list)), dtype='float32')
            targ_vec = np.zeros((len(self.targ_list)), dtype='float32')
            obj_vec[self.obj_list.index(targets[0].name)] = 1.
            targ_vec[self.targ_list.index(targets[1].name)] = 1.
            target_vec = np.zeros((self.target_dim,))
            set_params_attrs(plan.params, plan.state_inds, state, 0)
            for target_name in self.targets[condition]:
                target = plan.params[target_name]
                target.value[:,0] = self.targets[condition][target.name]
                target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

            sample = Sample(self)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TASK_ENUM, task_vec, 0)
            sample.set(OBJ_ENUM, obj_vec, 0)
            sample.set(TARG_ENUM, targ_vec, 0)
            sample.set(OBJ_POSE_ENUM, self.state_inds[targets[0].name, 'pose'], 0)
            sample.set(TARG_POSE_ENUM, self.targets[condition][targets[1].name], 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), 0)
            sample.set(TARGETS_ENUM, target_vec, 0)
            sample.condition = condition
            sample.task = task
            return sample, failed_preds, success

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    fill_vector(plan.params, plan.action_inds, U, t+1)
                else:
                    fill_vector(plan.params, plan.action_inds, U, t)
                return U

        sample = self.sample_task(optimal_pol(), condition, state, [task, targets[0].name, targets[1].name], noisy=False, fixed_obj=True)
        self.optimal_samples[task].append(sample)
        sample.set_ref_X(sample.get(STATE_ENUM))
        sample.set_ref_U(sample.get_U())
        return sample, failed_preds, success


    def get_hl_plan(self, state, condition, failed_preds):
        return self._get_hl_plan(state, self.targets[condition], self.plans.values()[0].params, self.state_inds, failed_preds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets


    def get_sample_constr_cost(self, sample):
        obj = self.plans.values()[0].params[self.obj_list[np.argmax(sample.get(OBJ_ENUM, t=0))]]
        targ = self.plans.values()[0].params[self.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]]
        targets = [obj, targ]
        # targets = get_next_target(self.plans.values()[0], sample.get(STATE_ENUM, t=0), sample.task, self.targets[sample.condition])
        plan = self.plans[sample.task, targets[0].name]
        for t in range(sample.T):
            set_params_attrs(plan.params, plan.state_inds, sample.get(STATE_ENUM, t=t), t)

        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        for target in targets:
            if target.name in self.targets[sample.condition]:
                plan.params[target.name].value[:,0] = self.targets[sample.condition][target.name]

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
        plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]

        return check_constr_violation(plan)


    def replace_conditions(self, conditions, keep=(0.2, 0.5)):
        self.targets = []
        for i in range(conditions):
            self.targets.append(get_end_targets(self.num_cans))
        self.init_vecs = get_random_initial_state_vec(self.num_cans, self.targets, self.dX, self.state_inds, conditions)
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.target_vecs = []
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)

        if keep != (1., 1.):
            self.clear_samples(*keep)
