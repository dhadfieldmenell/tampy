import copy
import sys
import traceback

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

import ctrajoptpy


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
from policy_hooks.tamp_agent import TAMPAgent


MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100


class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise):
        u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            u[self.action_inds[param, attr]] = self.opt_traj[t, self.state_inds[param, attr]]
        return u


class NAMOSortingAgent(TAMPAgent):
    def sample_task(self, policy, condition, x0, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True):
        task = tuple(task)
        plan = self.plans[task[:2]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon - 1
        sample = Sample(self)
        sample.init_t = 0

        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for target_name in self.targets[condition]:
            target = plan.params[target_name]
            target.value[:,0] = self.targets[condition][target.name]
            target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = 1e0 * generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            X = np.zeros((plan.symbolic_bound))
            fill_vector(plan.params, plan.state_inds, X, t)

            sample.set(STATE_ENUM, X.copy(), t)
            if LIDAR_ENUM in self._hyperparams['obs_include']:
                lidar = self.dist_obs(plan, t)
                sample.set(LIDAR_ENUM, lidar.flatten(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            # sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[self.task_list.index(task[0])] = 1.
            sample.task_ind = self.task_list.index(task[0])
            sample.set(TASK_ENUM, task_vec, t)
            sample.set(TARGETS_ENUM, target_vec.copy(), t)

            obj_vec = np.zeros((len(self.obj_list)), dtype='float32')
            targ_vec = np.zeros((len(self.targ_list)), dtype='float32')
            obj_vec[self.obj_list.index(task[1])] = 1.
            targ_vec[self.targ_list.index(task[2])] = 1.
            sample.obj_ind = self.obj_list.index(task[1])
            sample.targ_ind = self.targ_list.index(task[2])
            sample.set(OBJ_ENUM, obj_vec, t)
            sample.set(TARG_ENUM, targ_vec, t)
            sample.set(OBJ_POSE_ENUM, X[self.state_inds[task[1], 'pose']].copy(), t)
            sample.set(TARG_POSE_ENUM, self.targets[condition][task[2]].copy(), t)
            sample.set(EE_ENUM, X[self.state_inds['pr2', 'pose']], t)
            sample.task = task[0]
            sample.obj = task[1]
            sample.targ = task[2]
            sample.condition = condition

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U = policy.act(sample.get_X(t=t).copy(), obs.copy(), t, noise[t])
            # for param_name, attr in self.action_inds:
            #     if (param_name, attr) in self.state_inds:
            #         inds1 = self.action_inds[param_name, attr]
            #         inds2 = self.state_inds[param_name, attr]
            #         for i in range(len(inds1)):
            #             if U[inds1[i]] - X[inds2[i]] > 1:
            #                 U[inds1[i]] = X[inds2[i]] + 1
            #             elif U[inds1[i]] - X[inds2[i]] < -1:
            #                 U[inds1[i]] = X[inds2[i]] - 1
            # if np.all(np.abs(U - self.traj_hist[-1]) < self.move_limit):
            #     sample.use_ts[t] = 0

            if np.any(np.isnan(U)):
                U[np.isnan(U)] = 0
            if np.any(np.abs(U) == np.inf):
                U[np.abs(U) == np.inf] = 0
            # robot_start = X[plan.state_inds['pr2', 'pose']]
            # robot_vec = U[plan.action_inds['pr2', 'pose']] - robot_start
            # if np.sum(np.abs(robot_vec)) != 0 and np.linalg.norm(robot_vec) < 0.005:
            #     U[plan.action_inds['pr2', 'pose']] = robot_start + 0.1 * robot_vec / np.linalg.norm(robot_vec)
            sample.set(ACTION_ENUM, U.copy(), t)
                # import ipdb; ipdb.set_trace()
            
            # self.traj_hist.append(U)
            # while len(self.traj_hist) > self.hist_len:
            #     self.traj_hist.pop(0)

            obj = task[1] if fixed_obj else None

            self.run_policy_step(U, X, self.plans[task[:2]], t, obj)
            # if np.any(np.abs(U) > 1e10):
            #     import ipdb; ipdb.set_trace()

        if policy not in self.n_policy_calls:
            self.n_policy_calls[policy] = 1
        else:
            self.n_policy_calls[policy] += 1
        # print 'Called policy {0} times.'.format(self.n_policy_calls[policy])

        X = np.zeros((plan.symbolic_bound))
        fill_vector(plan.params, plan.state_inds, X, plan.horizon-1)
        sample.end_state = X
        return sample


    def dist_obs(self, plan, t):
        pr2 = plan.params['pr2']
        n_dirs = self.n_dirs
        obs = 1e1*np.ones(n_dirs)
        angles = 2 * np.array(range(n_dirs), dtype='float32') / n_dirs
        pr2.openrave_body.set_pose(pr2.pose[:,t])

        for p_name in plan.params:
            p = plan.params[p_name]
            if p.is_symbol() or p is pr2: continue
            p.openrave_body.set_pose(p.pose[:,t])
            collisions = self._cc.BodyVsBody(pr2.openrave_body.env_body, 
                                             p.openrave_body.env_body)
            for i, c in enumerate(collisions):
                linkA = c.GetLinkAParentName()
                linkB = c.GetLinkBParentName()

                if linkA == 'pr2' and linkB == p.name:
                    pt0 = c.GetPtA()
                    pt1 = c.GetPtB()
                elif linkB == 'pr2' and linkA == p.name:
                    pt0 = c.GetPtB()
                    pt1 = c.GetPtA()
                else:
                    continue

                distance = c.GetDistance()
                normal = c.GetNormal()

                # assert np.abs(np.linalg.norm(normal) - 1) < 1e-3
                angle = np.arccos(normal[0])
                if normal[1] < 0:
                    angle = 2*np.pi - angle
                closest_angle = np.argmin(np.abs(angles - angle))
                if distance < obs[closest_angle]:
                    obs[closest_angle] = distance

        return obs


    def run_policy_step(self, u, x, plan, t, obj):
        u_inds = self.action_inds
        x_inds = self.state_inds
        in_gripper = False

        if t < plan.horizon - 1:
            for param, attr in u_inds:
                getattr(plan.params[param], attr)[:, t+1] = x[x_inds[param, attr]] + u[u_inds[param, attr]]

            for param in plan.params.values():
                if param._type == 'Can':
                    dist = plan.params['pr2'].pose[:, t] - plan.params[param.name].pose[:, t]
                    radius1 = param.geom.radius
                    radius2 = plan.params['pr2'].geom.radius
                    grip_dist = radius1 + radius2 + dsafe
                    if plan.params['pr2'].gripper[0, t] > 0.2 and np.abs(dist[0]) < 0.1 and np.abs(grip_dist + dist[1]) < 0.1 and (obj is None or param.name == obj):
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + [0, grip_dist+dsafe]
                    elif param._type == 'Can':
                        param.pose[:, t+1] = param.pose[:, t]

        return True


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], fixed_targets=[]):
        exclude_targets = []
        success = False

        if len(fixed_targets):
            targets = fixed_targets
            obj = fixed_targets[0]
            targ = fixed_targets[1]
        else:
            task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))
            obj = self.plans.values()[0].params[self.obj_list[np.argmax(obj_distr)]]
            targ = self.plans.values()[0].params[self.targ_list[np.argmax(targ_distr)]]
            targets = [obj, targ]
            # targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition], sample_traj=traj_mean)

        failed_preds = []
        iteration = 0
        while not success and targets[0] != None:
            if iteration > 0 and not len(fixed_targets):
                 targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition], sample_traj=traj_mean, exclude=exclude_targets)

            iteration += 1
            if targets[0] is None:
                break

            plan = self.plans[task, targets[0].name] 
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
                self.save_free(plan)
                success = self.solver._backtrack_solve(plan, n_resamples=3, traj_mean=traj_mean, task=(self.task_list.index(task), self.obj_list.index(obj.name), self.targ_list.index(targ.name)))
                # viewer = OpenRAVEViewer._viewer if OpenRAVEViewer._viewer is not None else OpenRAVEViewer(plan.env)
                # if task == 'putdown':
                #     import ipdb; ipdb.set_trace()
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
            except Exception as e:
                traceback.print_exception(*sys.exc_info())
                self.restore_free(plan)
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

        traj = np.zeros((plan.horizon, self.dU))
        disp_traj = np.zeros((plan.horizon, self.dU))
        for t in range(plan.horizon):
            fill_vector(plan.params, plan.action_inds, traj[t], t)
            if t > 0:
                disp_traj[t-1] = traj[t] - traj[t-1]

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, disp_traj), condition, state, [task, targets[0].name, targets[1].name], noisy=False, fixed_obj=True)
        self.optimal_samples[task].append(sample)
        sample.set_ref_X(sample.get(STATE_ENUM))
        sample.set_ref_U(sample.get_U())
        return sample, failed_preds, success


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
