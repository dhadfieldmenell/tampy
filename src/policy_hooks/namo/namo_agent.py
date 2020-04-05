import copy
import sys
import time
import traceback

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import core.util_classes.common_constants as const
if const.USE_OPENRAVE:
    # import openravepy
    # from openravepy import RaveCreatePhysicsEngine
    # import ctrajoptpy
    pass
else:
    import pybullet as p


# from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

from baxter_gym.envs import MJCEnv

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
# from policy_hooks.namo.sorting_prob_4 import *
from policy_hooks.tamp_agent import TAMPAgent


MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100
GRIP_TOL = 0.
MIN_STEP = 1e-2
# LIDAR_DIST = 2.0
LIDAR_DIST = 2.5
DSAFE = 5e-1
MAX_STEP = 2


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
    def __init__(self, hyperparams):
        super(NAMOSortingAgent, self).__init__(hyperparams)

        for plan in self.plans.values():
            for t in range(plan.horizon):
                plan.params['obs0'].pose[:,t] = plan.params['obs0'].pose[:,0]

        self.robot_height = 1
        self.use_mjc = hyperparams.get('use_mjc', False)
        wall_dims = OpenRAVEBody.get_wall_dims('closet')
        config = {
            'obs_include': ['overhead_camera'],
            'include_files': [],
            'include_items': [
                {'name': 'pr2', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.6, 1.), 'rgba': (1, 1, 1, 1)},
            ],
            'view': False,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height'])
        }

        self.main_camera_id = 0
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices()
        for name in prim_options[OBJ_ENUM]:
            if name =='pr2': continue
            cur_color = colors.pop(0)
            items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.4, 1.), 'rgba': tuple(cur_color)})
        for i in range(len(wall_dims)):
            dim, next_trans = wall_dims[i]
            next_trans[0,3] -= 3.5
            next_dim = dim # [dim[1], dim[0], dim[2]]
            pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
            items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})
        self.mjc_env = MJCEnv.load_config(config)
        # self.viewer = OpenRAVEViewer(self.env)
        # import ipdb; ipdb.set_trace()
        self.in_gripper = None


    def replace_targets(self, condition=0):
        raise Exception('Deprecated method')
        new_targets = self.prob.get_end_targets(self.prob.NUM_OBJS, randomize=False)
        self.targets[condition] = new_targets
        if hasattr(self.prob, 'NUM_TARGS'):
            for i in range(self.prob.NUM_TARGS, self.prob.NUM_OBJS):
                self.targets[condition]['can{0}_end_target'.format(i)] = self.x0[condition][self.state_inds['can{0}'.format(i), 'pose']]
        target_vec = np.zeros((self.target_dim,))
        for target_name in self.targets[condition]:
            target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
        self.target_vecs[condition]= target_vec


    def sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True):
        assert not np.any(np.isnan(state))
        self.in_gripper = None
        x0 = state[self._x_data_idx[STATE_ENUM]].copy()
        task = tuple(task)
        if self.discrete_prim:
            plan = self.plans[task]
        else:
            plan = self.plans[task[0]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon
        sample = Sample(self)
        sample.init_t = 0

        prim_choices = self.prob.get_prim_choices()
        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for target_name in self.targets[condition]:
            target = plan.params[target_name]
            target.value[:,0] = self.targets[condition][target.name]
            target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        n_steps = 0
        end_state = None
        for t in range(0, self.T):
            # U_full = np.zeros((self.dU))
            noise_full = np.zeros((self.dU,))
            cur_state = np.zeros((plan.symbolic_bound))
            # fill_vector(plan.params, plan.state_inds, cur_state, t)  
            for pname, aname in self.state_inds:
                p = plan.params[pname]
                if p.is_symbol(): continue
                aval = getattr(p, aname)[:,t]
                if np.any(np.isnan(aval)):
                    print('NAN in:', pname, aname, t)
                    aval[:] = 0.
                cur_state[self.state_inds[pname, aname]] = aval

            self.fill_sample(condition, sample, cur_state, t, task, fill_obs=True)
            X = cur_state.copy()
            cur_noise = noise[t]

            U_full = policy.act(sample.get_X(t=t), sample.get_obs(t=t).copy(), t, cur_noise)
            assert not np.any(np.isnan(U_full))
            while False: # np.all(np.abs(cur_state - X) < MIN_STEP):
                self.fill_sample(condition, sample, X, t, task, fill_obs=True)

                noise_full += cur_noise
                sample.set(NOISE_ENUM, cur_noise, t)

                if use_prim_obs:
                    obs = sample.get_prim_obs(t=t)
                else:
                    obs = sample.get_obs(t=t)

                U = policy.act(sample.get_X(t=t), obs.copy(), t, cur_noise)
                U = np.clip(U, -MAX_STEP, MAX_STEP)
                # U[U > 0.5] = 0.5
                # U[U < -0.5] = -0.5
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


                '''
                if np.any(np.isnan(U)):
                    print 'Replacing nan in action.'
                    U[np.isnan(U)] = 0.0
                if np.any(np.isinf(U)):
                    print 'Replacing inf in action.'
                    U[np.isinf(U)] = 0.0
                '''

                # if np.all(np.abs(U[:2]) < 1e-4) and U[2] < 1e-4:
                #     sample.use_ts[t] = 0
                # robot_start = X[plan.state_inds['pr2', 'pose']]
                # robot_vec = U[plan.action_inds['pr2', 'pose']] - robot_start
                # if np.sum(np.abs(robot_vec)) != 0 and np.linalg.norm(robot_vec) < 0.005:
                #     U[plan.action_inds['pr2', 'pose']] = robot_start + 0.1 * robot_vec / np.linalg.norm(robot_vec)
                sample.set(ACTION_ENUM, U.copy(), t)
                    # import ipdb; ipdb.set_trace()
                # self.traj_hist.append(U)
                # while len(self.traj_hist) > self.hist_len:
                #     self.traj_hist.pop(0)


                self.run_policy_step(U, X, self.plans[task], t, None)
                # if np.any(np.abs(U) > 1e10):
                #     import ipdb; ipdb.set_trace()
                
                n_steps += 1
                U_full += U
                U_full = np.clip(U_full, -MAX_STEP, MAX_STEP)
                fill_vector(plan.params, plan.state_inds, X, np.minimum(t+1, plan.horizon-1))
                cur_noise = np.zeros((self.dU))

                if n_steps >= sample.T:
                    sample.use_ts[t+1:] = 0.
                    break

            fill_vector(plan.params, plan.state_inds, cur_state, t)
            self.fill_sample(condition, sample, cur_state, t, task, fill_obs=True)

            sample.set(NOISE_ENUM, noise_full, t)

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U_full = np.clip(U_full, -MAX_STEP, MAX_STEP)
            assert not np.any(np.isnan(U_full))
            sample.set(ACTION_ENUM, U_full, t)
            self.run_policy_step(U_full, cur_state, self.plans[task], t, None)

            new_state = np.zeros((plan.symbolic_bound))
            # fill_vector(plan.params, plan.state_inds, cur_state, t)  
            for pname, aname in self.state_inds:
                p = plan.params[pname]
                if p.is_symbol(): continue
                aval = getattr(p, aname)[:,min(t+1, sample.T-1)]
                if np.any(np.isnan(aval)):
                    print('NAN in:', pname, aname, t+1)
                    aval[:] = 0.
                new_state[self.state_inds[pname, aname]] = aval


            if np.all(np.abs(cur_state - new_state) < 1e-2):
                sample.use_ts[t] = 0
         
            if n_steps == sample.T:
                end_state = sample.get_X(t=t)

        if policy not in self.n_policy_calls:
            self.n_policy_calls[policy] = 1
        else:
            self.n_policy_calls[policy] += 1
        # print 'Called policy {0} times.'.format(self.n_policy_calls[policy])
        # X = np.zeros((plan.symbolic_bound))
        # fill_vector(plan.params, plan.state_inds, X, plan.horizon-1)
        sample.end_state = end_state if end_state is not None else sample.get_X(t=self.T-1)
        sample.task_cost = self.goal_f(condition, sample.end_state)
        if sample.T == plan.horizon:
            sample.use_ts[-1] = 0

        # if np.linalg.norm(sample.get(EE_ENUM, t=0) - sample.get(EE_ENUM, t=sample.T)) < 5e-1:
        #     sample.use_ts[:] = 0

        cost = self.cost_f(sample.end_state, task, condition, active_ts=(sample.T-1, sample.T-1), targets=sample.targets)
        sample.post_cost = cost
        sample.use_ts = np.array(sample.use_ts)
        return sample


    def dist_obs(self, plan, t, n_dirs=-1, ignore=[], return_rays=False):
        if n_dirs <= 0:
            n_dirs = self.n_dirs
        pr2 = plan.params['pr2']
        obs = 1e1*np.ones(n_dirs)
        angles = 2 * np.pi * np.array(range(n_dirs), dtype='float32') / n_dirs
        rays = np.zeros((n_dirs, 6))
        rays[:, 2] = 0.01
        for i in range(n_dirs):
            a = angles[i]
            ray = np.array([np.cos(a), np.sin(a)])
            # rays[i, :2] = pr2.pose[:,t] + (pr2.geom.radius+0.01)*ray
            rays[i, :2] = pr2.pose[:,t]
            rays[i, 3:5] = LIDAR_DIST * ray
        # pr2.openrave_body.set_pose(pr2.pose[:,t])
        pr2.openrave_body.set_pose([20, 20]) # Get this out of the way

        for p_name in plan.params:
            p = plan.params[p_name]
            if p is pr2: continue

            if p.is_symbol():
                if hasattr(p, 'openrave_body') and p.openrave_body is not None:
                    p.openrave_body.set_pose([20, 20])
            elif (p_name, 'pose') in self.state_inds:
                p.openrave_body.set_pose(p.pose[:,t])
            else:
                p.openrave_body.set_pose(p.pose[:,0])

        for name in ignore:
            plan.params[name].openrave_body.set_pose([20, 20])

        if self.in_gripper is not None:
            self.in_gripper.openrave_body.set_pose([20,20])

        if const.USE_OPENRAVE:
            is_hits, hits = self.env.CheckCollisionRays(rays, None)
            dists = np.linalg.norm(hits[:,:2]-rays[:,:2], axis=1)
            for i in range(len(is_hits)):
                dists[i] = dists[i] if is_hits[i] else LIDAR_DIST
        else:
            p.stepSimulation()
            _, _, hit_frac, hit_pos, hit_normal = p.rayBatchTest(rays[:,:3], rays[:,:3]+rays[:,3:])
            dists = LIDAR_DIST * hit_frac

        # dists[np.abs(dists) > LIDAR_DIST] = LIDAR_DIST
        # dists[not np.array(is_hits)] = LIDAR_DIST
        if return_rays:
            return dists, rays

        return dists


    def trajopt_dist_obs(self, plan, t):
        pr2 = plan.params['pr2']
        n_dirs = self.n_dirs
        obs = 1e1*np.ones(n_dirs)
        angles = 2 * np.pi * np.array(range(n_dirs), dtype='float32') / n_dirs
        pr2.openrave_body.set_pose(pr2.pose[:,t])

        for p_name in plan.params:
            p = plan.params[p_name]
            if p.is_symbol() or p is pr2: continue
            if (p_name, 'pose') in self.state_inds:
                p.openrave_body.set_pose(p.pose[:,t])
            else:
                p.openrave_body.set_pose(p.pose[:,0])
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

        old_state = x.copy()
        old_pose = plan.params['pr2'].pose[:, t].copy()
        in_gripper = []

        grasp = plan.params['grasp0'].value[:,0]
        if t < plan.horizon - 1:
            for param, attr in u_inds:
                if attr == 'pose':
                    getattr(plan.params[param], attr)[:, t] = x[x_inds[param, attr]] + u[u_inds[param, attr]]
                # elif attr == 'gripper':
                #     getattr(plan.params[param], attr)[:, t] = u[u_inds[param, attr]]
                elif attr == 'acc':
                    old_vel = x[x_inds[param, 'vel']]
                    new_vel = old_vel + u[u_inds[param, 'acc']]
                    old_pos = x[x_inds[param, 'pos']]
                    new_pos = old_pos + new_vel
                    getattr(plan.params[param], 'pos')[:, t+1] = new_pos
                    getattr(plan.params[param], 'vel')[:, t+1] = new_vel
                # else:
                #     getattr(plan.params[param], attr)[:, t] = x[x_inds[param, attr]] + u[u_inds[param, attr]]
 
            for param in plan.params.values():
                if param._type == 'Can':
                    disp = old_state[x_inds['pr2', 'pose']] - old_state[x_inds[param.name, 'pose']]# plan.params['pr2'].pose[:, t] - param.pose[:, t]
                    new_disp = plan.params['pr2'].pose[:, t] - param.pose[:, t]
                    pr2_disp = plan.params['pr2'].pose[:, t] - old_state[x_inds['pr2', 'pose']]
                    dist = np.linalg.norm(disp)
                    if grasp[1] < 0: grasp_check = disp[1] < 0
                    if grasp[1] > 0: grasp_check = disp[1] > 0
                    if grasp[0] < 0: grasp_check = disp[0] < 0
                    if grasp[0] > 0: grasp_check = disp[0] > 0
                    radius1 = param.geom.radius
                    radius2 = plan.params['pr2'].geom.radius
                    #if u[u_inds['pr2', 'gripper']][0] > GRIP_TOL and dist >= radius1 + radius2 - 0.5 * DSAFE and dist <= radius1 + radius2 + DSAFE and (obj is None or param.name == obj):
                    if plan.params['pr2'].gripper[0,t] > GRIP_TOL and dist >= radius1 + radius2 and dist <= radius1 + radius2 + DSAFE and np.abs(disp[0]) < DSAFE and grasp_check:
                    # if u[u_inds['pr2', 'gripper']][0] > GRIP_TOL and dist >= radius1 + radius2 - 0.5 * DSAFE and dist <= radius1 + radius2 + DSAFE and np.abs(disp[0]) < 0.5 * DSAFE and disp[1] < 0:
                    #if u[u_inds['pr2', 'gripper']][0] > GRIP_TOL and np.abs(disp[0]) < 0.5 * DSAFE and disp[1] <= -(radius1 + radius2 - 0.5 * DSAFE) and disp[1] >= -(radius1 + radius2 + DSAFE):
                        # in_gripper.append((param.name, disp))
                        if self.in_gripper is None:
                            self.in_gripper = param
                        if self.in_gripper is param:
                            param.pose[:, t] = plan.params['pr2'].pose[:, t] - disp

                    elif plan.params['pr2'].gripper[0,t] <= GRIP_TOL:
                        self.in_gripper = None

                    elastic = 1e-2 # max(1e-2, 2e-1 * np.linalg.norm(pr2_disp))
                    if param is not self.in_gripper and np.linalg.norm(new_disp) <= radius1 + radius2:
                        dx, dy = -1e1 * pr2_disp
                        zx, zy = param.pose[:,t]
                        x1, y1 = plan.params['pr2'].pose[:,t] - [0.5*dx, 0.5*dy] - [zx, zy]
                        # zx, zy = plan.params['pr2'].pose[:,t]
                        # x1, y1 = param.pose[:,t] - [0.5*dx, 0.5*dy] - [zx, zy]
                        # dx, dy = 1e1 * pr2_disp
                        x2, y2 = x1 + dx, y1 + dy
                        dr = np.sqrt(dx**2 + dy**2)
                        D = x1 * y2 - x2 * y1
                        r = radius1 + radius2 + elastic # + 2e-2
                        sy = -1. if dy < 0 else 1.

                        if dx >= 0 and dy >= 0:
                            x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx <= 0 and dy <= 0:
                            x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx >= 0 and dy <= 0:
                            x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx <= 0 and dy >= 0:
                            x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)

                        # param.pose[:, t] = [zx + x, zy + y]
                        plan.params['pr2'].pose[:, t] = [zx + x, zy + y]
                        if self.in_gripper is not None:
                            self.in_gripper.pose[:, t] = plan.params['pr2'].pose[:, t] - (old_state[x_inds['pr2', 'pose']] - old_state[x_inds[self.in_gripper.name, 'pose']])
                    
                    if self.in_gripper is not None and self.in_gripper is not param and np.linalg.norm(self.in_gripper.pose[:,t] - param.pose[:,t]) < radius2 + radius2:
                        dx, dy = -1e1 * pr2_disp
                        zx, zy = param.pose[:,t]
                        x1, y1 = self.in_gripper.pose[:,t] - [0.5*dx, 0.5*dy] - [zx, zy]
                        # zx, zy = plan.params['pr2'].pose[:,t]
                        # x1, y1 = param.pose[:,t] - [0.5*dx, 0.5*dy] - [zx, zy]
                        # dx, dy = 1e1 * pr2_disp
                        x2, y2 = x1 + dx, y1 + dy
                        dr = np.sqrt(dx**2 + dy**2)
                        D = x1 * y2 - x2 * y1
                        r = radius1 + radius2 + elastic # + 2e-2
                        sy = -1. if dy < 0 else 1.

                        if dx >= 0 and dy >= 0:
                            x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx <= 0 and dy <= 0:
                            x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx >= 0 and dy <= 0:
                            x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                        elif dx <= 0 and dy >= 0:
                            x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
                            y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)

                        # param.pose[:, t] = [zx + x, zy + y]
                        self.in_gripper.pose[:, t] = [zx + x, zy + y]
                        plan.params['pr2'].pose[:,t] = self.in_gripper.pose[:,t] - (old_state[x_inds[self.in_gripper.name, 'pose']] - old_state[x_inds['pr2', 'pose']])
                    else:
                        pass
                        # param.pose[:, t] = param.pose[:, t]
                    # dist = plan.params['pr2'].pose[:, t] - param.pose[:, t]
                    # radius1 = param.geom.radius
                    # radius2 = plan.params['pr2'].geom.radius
                    # grip_dist = radius1 + radius2 + dsafe
                    # if plan.params['pr2'].gripper[0, t] > 0.2 and np.abs(dist[0]) < 0.1 and np.abs(grip_dist + dist[1]) < 0.1 and (obj is None or param.name == obj):
                    #     param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + [0, grip_dist+dsafe]
                    # elif param._type == 'Can':
                    #     param.pose[:, t+1] = param.pose[:, t]

                elif param._type == 'Box':
                    disp = plan.params['pr2'].pose[:, t] - param.pose[:, t]
                    dist = np.linalg.norm(disp)
                    box_w, box_l = param.geom.width,  param.geom.length
                    radius = plan.params['pr2'].geom.radius
                    grip_dist_w = box_w + radius2 + dsafe
                    grip_dist_l = box_l + radius2 + dsafe
                    if plan.params['pr2'].gripper[0, t] < GRIP_TOL or (obj is not None and param.name != obj):
                        param.pose[:, t+1] = param.pose[:, t]
                    elif disp[0] < 0 and np.abs(disp[1]) < box_w and -disp[0] >= box_l + radius and -disp[0] <= box_l + radius + DSAFE:
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + np.array([0, box_l+radius+2*dsafe])
                    elif disp[0] >= 0 and np.abs(disp[1]) < box_w and disp[0] >= box_l + radius and disp[0] <= box_l + radius + DSAFE:
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + np.array([0, -box_l-radius-2*dsafe])
                    elif disp[1] < 0 and np.abs(disp[1]) < box_l and -disp[1] >= box_w + radius and -disp[1] <= box_w + radius + DSAFE:
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + np.array([0, box_w+radius+2*dsafe])
                    elif disp[1] >= 0 and np.abs(disp[1]) < box_l and disp[1] >= box_w + radius and disp[1] <= box_w + radius + DSAFE:
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + np.array([0, -box_w-radius-2*dsafe])
                    else:
                        param.pose[:, t+1] = param.pose[:, t]

            ignore = []
            #dist, rays = self.dist_obs(plan, t, 8, ignore=ignore, return_rays=True)
            if False: #self.check_col(plan, t): # np.any(np.abs(dist) < plan.params['pr2'].geom.radius - 0.5 * DSAFE):
                for pname, aname in self.state_inds:
                    if plan.params[pname].is_symbol(): continue
                    getattr(plan.params[pname], aname)[:,t+1] = old_state[self.state_inds[pname, aname]]
            else:
                for pname, aname in self.state_inds:
                    if plan.params[pname].is_symbol(): continue
                    getattr(plan.params[pname], aname)[:,t+1] = getattr(plan.params[pname], aname)[:,t].copy()
                plan.params['pr2'].gripper[:,t+1] = u[u_inds['pr2', 'gripper']]
            for pname, aname in self.state_inds:
                if plan.params[pname].is_symbol(): continue
                getattr(plan.params[pname], aname)[:,t] = old_state[self.state_inds[pname, aname]]

        return True


    def check_col(self, plan, t, ignore=[]):
        pr2 = plan.params['pr2']
        pr2.openrave_body.set_pose(pr2.pose[:,t])
        col = False
        for p in plan.params:
            param = plan.params[p]
            if param.is_symbol() or param._type == 'Can' or param is self.in_gripper: continue
            param.openrave_body.set_pose(param.pose[:,t])
            next_col = self.env.CheckCollision(pr2.openrave_body.env_body, param.openrave_body.env_body)
            if next_col: return True

        return col


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0, targets=[], x_only=False, t_limit=60, n_resamples=10, out_coeff=None, smoothing=False, attr_dict=None):
        success = False
        old_targets = self.target_vecs[condition]
        if not len(targets):
            targets = self.target_vecs[condition]
        else:
            self.target_vecs[condition] = targets.copy()
            for tname, attr in self.target_inds:
                self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
        
        x0 = state[self._x_data_idx[STATE_ENUM]]

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task] 
        prim_choices = self.prob.get_prim_choices()
        # obj_name = prim_choices[OBJ_ENUM][task[1]]
        # targ_name = prim_choices[TARG_ENUM][task[2]]
        set_params_attrs(plan.params, plan.state_inds, x0, 0)

        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                param.pose[:, 0] = x0[self.state_inds[param_name, 'pose']]
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = param.pose[:,0]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        grasp = np.array([0, -0.601])
        if GRASP_ENUM in prim_choices:
            grasp = self.set_grasp(grasp, task[3])

        plan.params['grasp0'].value[:,0] = grasp
        plan.params['pr2'].pose[:, 0] = x0[self.state_inds['pr2', 'pose']]
        plan.params['obs0'].pose[:] = plan.params['obs0'].pose[:,:1]

        run_solve = True
        '''
        for param_name in plan.params:
            if param_name == 'pr2': continue
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if np.all(np.abs(param.pose[:,0] - self.targets[condition][targ_name]) < 0.01):
                run_solve = False
                break
        '''

        
        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]

        prim_vals = self.get_prim_value(condition, state, task)
        old_out_coeff = self.solver.strong_transfer_coeff
        if out_coeff is not None:
            self.solver.strong_transfer_coeff = out_coeff
        try:
            if smoothing:
                success = self.solver.quick_solve(plan, n_resamples=n_resamples, traj_mean=traj_mean, attr_dict=attr_dict)
            elif run_solve:
                success = self.solver._backtrack_solve(plan, n_resamples=n_resamples, traj_mean=traj_mean, inf_f=inf_f, task=task, time_limit=t_limit)
            else:
                success = False
        except Exception as e:
            print(e)
            traceback.print_exception(*sys.exc_info())
            success = False

        self.solver.strong_transfer_coeff = old_out_coeff

        '''
        print('Planning succeeded' if success else 'Planning failed')
        print('Problem:', plan.actions)
        # print(['{0}: {1}\n'.format(p.name, p.pose[:,0]) for p in plan.params.values() if not p.is_symbol()])
        # print(['{0}: {1}\n'.format(p.name, p.value[:,0]) for p in plan.params.values() if p.is_symbol()])
        '''

        '''
        if not success:
            for action in plan.actions:
                try:
                    print('Solve failed on action: {0}'.format(action))
                    print(action, plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps))
                    print(['{0}: {1}\n'.format(p.name, p.pose[:,0]) for p in plan.params.values() if not p.is_symbol()])
                    print('\n')
                except:
                    pass
            print('\n\n')
        '''
        try:
            if not len(failed_preds):
                for action in plan.actions:
                    failed_preds += [(pred, t) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
        except:
            failed_preds += ['Nan in pred check for {0}'.format(action)]
        #if not success and not smoothing:
        #    sample = Sample(self)
        #    self.fill_sample(condition, sample, x0, 0, task, fill_obs=True)
        #    return sample, failed_preds, success

        traj = np.zeros((plan.horizon, self.symbolic_bound))
        for pname, aname in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            inds = self.state_inds[pname, aname]
            for t in range(plan.horizon):
                traj[t][inds] = getattr(plan.params[pname], aname)[:,t]

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    for param, attr in plan.action_inds:
                        if attr == 'pose':
                            # U[plan.action_inds[param, attr]] = getattr(plan.params[param], attr)[:, t+1] - getattr(plan.params[param], attr)[:, t]
                            U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] - traj[t][plan.state_inds[param, attr]] 
                        elif attr == 'gripper':
                            # U[plan.action_inds[param, attr]] = getattr(plan.params[param], attr)[:, t]
                            U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] 
                        else:
                            raise NotImplementedError
                    # U[plan.action_inds['pr2', 'pose']] = plan.params['pr2'].pose[:, t+1] - plan.params['pr2'].pose[:, t]
                    # U[plan.action_inds['pr2', 'gripper']] = plan.params['pr2'].gripper[:, t+1]
                if np.any(np.isnan(U)):
                    print('NAN in {0} plan act'.format(success))
                    U[:] = 0.
                return U
        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)

        for t in range(sample.T):
            if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 5e-3: sample.use_ts[t] = 0.

        traj = sample.get(STATE_ENUM)
        for param_name, attr in self.state_inds:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            diff = traj[:, self.state_inds[param_name, attr]].T - getattr(param, attr)
            if np.any(np.abs(diff) > 1e-3): print(diff, param_name, attr, 'ERROR IN OPT ROLLOUT')

        # self.optimal_samples[task].append(sample)
        # print(sample.get_X())
        if not smoothing:
            if not success:
                sample.use_ts[:] = 0.
                print('Failed to plan for: {0} {1} smoothing? {2} {3}'.format(task, failed_preds, smoothing, state))
                print('FAILED PLAN')
            else:
                print('SUCCESSFUL PLAN for {0}'.format(task))
        # else:
        #     print('Plan success for {0} {1}'.format(task, state))
        return sample, failed_preds, success

    # def get_sample_constr_cost(self, sample):
    #     obj = self.plans.values()[0].params[self.obj_list[np.argmax(sample.get(OBJ_ENUM, t=0))]]
    #     targ = self.plans.values()[0].params[self.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]]
    #     targets = [obj, targ]
    #     # targets = get_next_target(self.plans.values()[0], sample.get(STATE_ENUM, t=0), sample.task, self.targets[sample.condition])
    #     plan = self.plans[sample.task, targets[0].name]
    #     for t in range(sample.T):
    #         set_params_attrs(plan.params, plan.state_inds, sample.get(STATE_ENUM, t=t), t)

    #     for param_name in plan.params:
    #         param = plan.params[param_name]
    #         if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
    #             plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

    #     for target in targets:
    #         if target.name in self.targets[sample.condition]:
    #             plan.params[target.name].value[:,0] = self.targets[sample.condition][target.name]

    #     plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
    #     dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
    #     plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]

    #     return check_constr_violation(plan)


    def retime_sample(self, sample):
        cur_t = 0
        sample.use_ts[:] = 0
        for t in range(sample.T):
            u = np.zeros((self.dU,)) #sample.get(ACTION_ENUM, cur_t)
            while np.linalg.norm(u[self.action_inds['pr2', 'pose']]) < 0.05 and cur_t < sample.T:
                u += sample.get(ACTION_ENUM, cur_t)
                cur_t += 1
            if cur_t-1 >= sample.T: break
            u[self.action_inds['pr2', 'gripper']] = sample.get(ACTION_ENUM, cur_t-1)[self.action_inds['pr2', 'gripper']]
            sample.set_obs(sample.get_obs(t=cur_t-1), t=t)
            sample.set_X(sample.get_X(t=cur_t-1), t=t)
            sample.set(ACTION_ENUM, u, t=t)
            sample.use_ts[t] = 1.
            if cur_t >= sample.T: break
        return sample


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        mp_state = mp_state.copy()
        plan = self.plans[task]
        ee_pose = mp_state[self.state_inds['pr2', 'pose']]
        if targets is None:
            targets = self.target_vecs[cond]

        sample.set(EE_ENUM, ee_pose, t)
        sample.set(STATE_ENUM, mp_state, t)
        sample.set(GRIPPER_ENUM, mp_state[self.state_inds['pr2', 'gripper']], t)

        task_ind = task[0]
        obj_ind = task[1]
        targ_ind = task[2]
        prim_choices = self.prob.get_prim_choices()

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.task_ind = task[0]
        sample.set(TASK_ENUM, task_vec, t)
        
        grasp = np.array([0, -0.601])
        if self.discrete_prim:
            if GRASP_ENUM in prim_choices:
                grasp = self.set_grasp(grasp, task[3])
                grasp_vec = np.zeros(self._hyperparams['sensor_dims'][GRASP_ENUM])
                grasp_vec[task[3]] = 1.
                sample.set(GRASP_ENUM, grasp_vec, t)
                plan.params['grasp0'].value[:,0] = grasp
            
            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            if task[0] == 0:
                obj_vec[task[1]] = 1.
                targ_vec[0] = 1.
            elif task[0] == 1:
                obj_vec[0] = 1.
                targ_vec[task[2]] = 1.
            sample.obj_ind = task[1]
            sample.targ_ind = task[2]
            sample.set(OBJ_ENUM, obj_vec, t)
            sample.set(TARG_ENUM, targ_vec, t)

            obj_name = list(prim_choices[OBJ_ENUM])[obj_ind]
            targ_name = list(prim_choices[TARG_ENUM])[targ_ind]
            obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[self.state_inds['pr2', 'pose']]
            targ_pose = self.targets[cond][targ_name] - mp_state[self.state_inds['pr2', 'pose']]
        else:
            obj_pose = label[1] - mp_state[self.state_inds['pr2', 'pose']]
            targ_pose = label[1] - mp_state[self.state_inds['pr2', 'pose']]
        if task[0] == 1:
            obj_pose = np.zeros_like(obj_pose)
        sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
        if t == 0:
            sample.set(INIT_OBJ_POSE_ENUM, obj_pose, t)
        else:
            init_obj_pos = sample.get(INIT_OBJ_POSE_ENUM, 0)
            sample.set(INIT_OBJ_POSE_ENUM, init_obj_pos, t)

        if task[0] == 0:
            targ_pose = np.zeros_like(targ_pose)
        sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)

        sample.task = task
        sample.obj = task[1]
        sample.targ = task[2]
        sample.condition = cond
        sample.task_name = self.task_list[task[0]]
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(GOAL_ENUM, np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]]), t)
        sample.targets = targets.copy()

        if task[0] == 0:
            sample.set(END_POSE_ENUM, obj_pose + grasp, t)
        if task[0] == 1:
            sample.set(END_POSE_ENUM, targ_pose + grasp, t)
        
        if fill_obs:
            if LIDAR_ENUM in self._hyperparams['obs_include']:
                plan = self.plans.values()[0]
                set_params_attrs(plan.params, plan.state_inds, mp_state, t)
                lidar = self.dist_obs(plan, t)
                sample.set(LIDAR_ENUM, lidar.flatten(), t)

            if IM_ENUM in self._hyperparams['obs_include']:
                self.reset_to_state(sample.get_X(t=t))
                im = self.mjc_env.render(height=self.image_height, width=self.image_width)
                sample.set(IM_ENUM, im.flatten(), t)


    def fill_action(self, sample):
        act = np.zeros((sample.T, self.dU))
        X = sample.get(STATE_ENUM)
        pos_inds = self.action_inds['pr2', 'pose']
        act[:-1, self.action_inds['pr2', 'pose']] = X[1:, pos_inds] - X[:-1, pos_inds]
        act[:-1, self.action_inds['pr2', 'gripper']] = X[1:, self.action_inds['pr2', 'gripper']] 
        act[np.abs(act) > 2] = 2.
        sample.set(ACTION_ENUM, act)


    def get_prim_options(self, cond, state):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        outs = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = get_prim_choices()
        plan = self.plans.values()[0]
        for enum in self.prim_dims:
            if enum == TASK_ENUM: continue
            out[enum] = []
            for item in options[enum]:
                if item in plan.params:
                    param = plan.params[item]
                    if param.is_symbol():
                        if param.name in self.targets[cond]:
                            out[enum] = self.targets[cond][param.name].copy()
                        else:
                            out[enum].append(param.value[:,0].copy())
                    else:
                        out[enum].append(mp_state[self.state_inds[item, 'pose']].copy())

            out[enum] = np.array(out[enum])
        return outs


    def get_prim_value(self, cond, state, task):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = self.task_list[task[0]]
        plan = self.plans[task]
        options = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
            item = options[enum][task[i]]
            if item in plan.params:
                param = plan.params[item]
                if param.is_symbol():
                    if param.name in self.targets[cond]:
                        out[enum] = self.targets[cond][param.name].copy()
                    else:
                        out[enum] = param.value[:,0]
                else:
                    out[enum] = mp_state[self.state_inds[item, 'pose']]

        return out


    def get_prim_index(self, enum, name):
        prim_options = self.prob.get_prim_choices()
        return prim_options[enum].index(name)


    def get_prim_indices(self, names):
        task = [self.task_list.index(names[0])]
        for i in range(1, len(names)):
            task.append(self.get_prim_index(self.prim_dims.keys()[i-1], names[i]))
        return tuple(task)


    def goal_f(self, condition, state, targets=None):
        if targets is None:
            targets = self.target_vecs[condition]
        cost = self.prob.NUM_OBJS
        plan = self.plans.values()[0]
        for param in plan.params.values():
            if param._type == 'Can':
                val = targets[self.target_inds['{0}_end_target'.format(param.name), 'value']]
                dist = np.linalg.norm(state[self.state_inds[param.name, 'pose']] - val)
                # np.sum((state[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                # cost -= 1 if dist < 0.3 else 0
                cost -= 1 if dist < 0.5 else 0

        # return cost / float(self.prob.NUM_OBJS)
        return 1. if cost > 0 else 0.

    '''
    def goal_f(self, condition, state):
        cost = 0
        plan = self.plans.values()[0]
        for param in plan.params.values():
            if param._type == 'Can':
                dist = np.sum((state[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                cost += 1 if dist > 0.01 else 0

        return cost
    '''

    def set_grasp(self, grasp, rot):
        prim_choices = self.prob.get_prim_choices()
        if GRASP_ENUM not in prim_choices:
            return grasp

        theta = rot * (2. * np.pi) / self._hyperparams['sensor_dims'][GRASP_ENUM]
        mat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        return np.matmul(mat, grasp)

    def _failed_preds(self, Xs, task, condition, active_ts=None, debug=False, targets=[]):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
        plan = self.plans[task]
        if not len(targets):
            targets = self.target_vecs[condition]
        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]
        tol = 1e-3
        obs = plan.params['obs0']
        for t in range(plan.horizon):
            obs.pose[:,t] = obs.pose[:,0]

        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)

        if active_ts == None:
            active_ts = (1, plan.horizon-1)

        prim_choices = self.prob.get_prim_choices()
        grasp = np.array([0, -0.601])
        assert GRASP_ENUM in prim_choices
        if GRASP_ENUM in prim_choices:
            grasp = self.set_grasp(grasp, task[3])
        plan.params['grasp0'].value[:,0] = grasp
 
        if active_ts[0] == 0:
            prim_choices = self.prob.get_prim_choices()
            obj = prim_choices[OBJ_ENUM][task[1]]
            targ = prim_choices[TARG_ENUM][task[2]]
            grasp = np.array([0, -0.601])
            if GRASP_ENUM in prim_choices:
                grasp = self.set_grasp(grasp, task[3])
            loc = Xs[0][self.state_inds[obj, 'pose']] if task[0] == 0 else targets[self.target_inds[targ, 'value']]
            plan.params['robot_end_pose'].value[:,0] = loc + grasp
        if active_ts[1] == plan.horizon-1:
            plan.params['robot_end_pose'].value[:,0] = Xs[-1][self.state_inds['pr2', 'pose']]
        prim_choices = self.prob.get_prim_choices()
        # obj_name = prim_choices[OBJ_ENUM][task[1]]
        # targ_name = prim_choices[TARG_ENUM][task[2]]

        # for t in range(active_ts[0], active_ts[1]+1):
        #     set_params_attrs(plan.params, plan.state_inds, Xs[t-active_ts[0]], t)

        plan.params['pr2'].pose[:,active_ts[0]] = Xs[0][self.state_inds['pr2', 'pose']]
        for param_name in plan.params:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                inds = self.state_inds[param_name, 'pose']
                param.pose[:,active_ts[0]:active_ts[1]+1] = Xs[:active_ts[1]+1-active_ts[0]][:,inds].T
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]
        # for target in self.targets[condition]:
        #     plan.params[target].value[:,0] = self.targets[condition][target]

        # if targ_name in self.targets[condition]:
        #     plan.params['{0}_end_target'.format(obj_name)].value[:,0] = self.targets[condition][targ_name]

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        plan.params['robot_end_pose'].value[:,0] = plan.params['pr2'].pose[:,-1]

        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        if debug:
            print(failed_preds)

        return failed_preds


    def cost_f(self, Xs, task, condition, active_ts=None, debug=False, targets=[]):
        if active_ts == None:
            active_ts = (1, plan.horizon-1)
        failed_preds = self._failed_preds(Xs, task, condition, active_ts=active_ts, debug=debug, targets=[])

        cost = 0
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=1e-3)
                    if viol is not None:
                        cost += np.max(viol)
                except Exception as e:
                    print('Exception in cost check')
                    print(e)
                    cost += 1e1

        if len(failed_preds) and cost == 0:
            cost += 1

        return cost


    def postcond_cost(self, sample):
        return self.cost_f(sample.get_X(sample.T-1), sample.task, sample.condition, active_ts=(sample.T-1, sample.T-1), targets=sample.targets)


    def add_viewer(self):
        self.mjc_env.add_viewer()



    # def cost_f(self, Xs, task, condition, active_ts=None, debug=False):
    #     if len(Xs.shape) == 1:
    #         Xs = Xs.reshape(1, Xs.shape[0])
    #     Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
    #     plan = self.plans[task]
    #     tol = 1e-3
    #     targets = self.targets[condition]

    #     if len(Xs.shape) == 1:
    #         Xs = Xs.reshape(1, -1)

    #     if active_ts == None:
    #         active_ts = (1, plan.horizon-1)

    #     for t in range(active_ts[0], active_ts[1]+1):
    #         set_params_attrs(plan.params, plan.state_inds, Xs[t-active_ts[0]], t)

    #     for param in plan.params:
    #         if plan.params[param]._type == 'Can':
    #             plan.params['{0}_init_target'.format(param)].value[:,0] = plan.params[param].pose[:,0]
    #             plan.params['{0}_end_target'.format(param)].value[:,0] = targets['{0}_end_target'.format(param)]

    #     plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
    #     plan.params['robot_end_pose'].value[:,0] = plan.params['pr2'].pose[:,-1]
    #     plan.params['{0}_init_target'.format(params[0].name)].value[:,0] = plan.params[params[0].name].pose[:,0]
    #     plan.params['{0}_end_target'.format(params[0].name)].value[:,0] = targets['{0}_end_target'.format(params[0].name)]

    #     failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
    #     if debug:
    #         print failed_preds

    #     cost = 0
    #     print plan.actions, failed_preds
    #     for failed in failed_preds:
    #         for t in range(active_ts[0], active_ts[1]+1):
    #             if t + failed[1].active_range[1] > active_ts[1]:
    #                 break

    #             try:
    #                 viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
    #                 if viol is not None:
    #                     cost += np.max(viol)
    #             except:
    #                 pass

    #     return cost
    

    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        for param_name, attr in self.state_inds:
            if attr == 'pose':
                pos = mp_state[self.state_inds[param_name, 'pose']].copy()
                self.mjc_env.set_item_pos(param_name, np.r_[pos, 0.5], mujoco_frame=False, forward=False)
        self.mjc_env.physics.forward()


    def set_to_targets(self, condition=0):
        prim_choices = self.prob.get_prim_choices()
        objs = prim_choices[OBJ_ENUM]
        for obj_name in objs:
            self.mjc_env.set_item_pos(obj_name, np.r_[self.targets[condition]['{0}_end_target'.format(obj_name)], 0], forward=False)
        self.mjc_env.physics.forward()


    def check_targets(self, x, condition=0):
        mp_state = x[self._x_data_idx]
        prim_choices = self.prob.get_prim_choices()
        objs = prim_choices[OBJ_ENUM]
        correct = 0
        for obj_name in objs:
            target = self.targets[condition]['{0}_end_target'.format(obj_name)]
            obj_pos = mp_state[self.state_inds[obj_name, 'pose']]
            if np.linalg.norm(obj_pos - target) < 0.6:
                correct += 1
        return correct


    def get_image(self, x, depth=False):
        self.reset_to_state(x)
        im = self.mjc_env.render(camera_id=0, depth=depth, view=False)
        return im


    def get_mjc_obs(self, x):
        self.reset_to_state(x)
        # return self.mjc_env.get_obs(view=False)
        return self.mjc_env.render()


    '''
    def perturb_solve(self, sample, perturb_var=0.05, inf_f=None):
        state = sample.get_X(t=0)
        condition = sample.get(condition)
        task = sample.task
        out = self.solve_sample_opt_traj(state, task, condition, traj_mean=sample.get_U(), inf_f=inf_f, mp_var=perturb_var)
        return out
    '''


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], targets=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean, targets=targets)
        if not len(targets):
            old_targets = self.target_vecs[condition]
        else:
            old_targets = self.target_vecs[condition]
            for tname, attr in self.target_inds:
                self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
            self.target_vecs[condition] = targets
        
        exclude_targets = []
        plan = self.plans[task]
        act_traj = np.zeros((plan.horizon, self.dU))
        for t in range(plan.horizon-1):
            pos_traj = opt_traj[:, self.state_inds['pr2', 'pose']]
            grip_traj = opt_traj[:, self.state_inds['pr2', 'gripper']]
            act_traj[t, self.action_inds['pr2', 'pose']] = pos_traj[t+1] - pos_traj[t]
            act_traj[t, self.action_inds['pr2', 'gripper']] = grip_traj[t+1]
        act_traj[-1] = act_traj[-2]
        # print(act_traj, '<-- traj in sampleopt')
        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, act_traj), condition, state, task, noisy=False)
        # print(sample.get(STATE_ENUM), '<--- traj after rollout')
        # self.optimal_samples[task].append(sample)
        sample.set_ref_X(opt_traj)
        sample.set_ref_U(sample.get_U())
     
        for t in range(sample.T):
            if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 5e-3: sample.use_ts[t] = 0.

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
        return sample


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        targets = get_prim_choices[TARG_ENUM]
        state = state[self._x_data_idx[STATE_ENUM]]
        params = self.plans.values()[0].params

        return hl_plan_for_state(state, targets, plan_id, params, self.state_inds, failed_preds)


    def cost_info(self, Xs, task, cond, active_ts=(0,0)):
        failed_preds = self._failed_preds(Xs, task, cond, active_ts=active_ts)
        plan = self.plans[task]
        if active_ts[0] == -1:
            active_ts = (plan.horizon-1, plan.horizon-1)
        cost_info = [str(plan.actions)]
        tol=1e-3
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    if viol is not None:
                        cost_info.append((t, failed[1].get_type(), viol.flatten()))
                    else:
                        cost_info.append((t, failed[1].get_type(), "Violation returned null"))
                except Exception as e:
                    cost_info.append((t, failed[1].get_type(), 'Error on evaluating: {0}'.format(e)))

        return cost_info


    def dummy_transition(self, X, task, cond):
        prim_choice = self.prob.get_prim_choices()
        task_name = self.task_list[task[0]]
        obj = prim_choice[OBJ_ENUM][task[1]]
        targ = prim_choice[TARG_ENUM][task[2]]
        new_x = X.copy()
        if task_name.find('movetograsp') > -1:
            new_x[self.state_inds['pr2', 'pose']] = X[self.state_inds[obj, 'pose']] + [0, -0.75]
        elif task_name.find('place_at') > -1:
            new_x[self.state_inds['pr2', 'pose']] = self.targets[cond][targ] + [0, -0.65]
            new_x[self.state_inds[obj, 'pose']] = self.targets[cond][targ]

        s = Sample(self)
        self.fill_sample(cond, s, new_x, 0, task, True)
        return s


    def relabel_traj(self, sample):
        task = sample.task
        attr_dict = {}
        plan = self.plans[task]
        init = copy.deepcopy(sample)
        if task[0] == 0:
            pname = self.prob.get_prim_choices()[OBJ_ENUM][task[1]]
            tname = self.prob.get_prim_choices()[TARG_ENUM][task[2]]
            x = sample.get(STATE_ENUM, sample.T-1)
            end_ee = x[self.state_inds['pr2', 'pose']]
            for t in range(sample.T):
                x = sample.get(STATE_ENUM, t)
                ee = x[self.state_inds['pr2', 'pose']]
                sample.set(END_POSE_ENUM, end_ee - ee, t)
            x = sample.get(STATE_ENUM, sample.T-1)
            g = plan.params['grasp0'].value[:,0]
            goal_ee = x[self.state_inds[pname, 'pose']] + g
            attr_dict[('robot_end_pose', 'value')] = goal_ee 
        elif task[0] == 1:
            plan = self.plans[task]
            pname = self.prob.get_prim_choices()[OBJ_ENUM][task[1]]
            tname = self.prob.get_prim_choices()[TARG_ENUM][task[2]]
            g = plan.params['grasp0'].value[:,0]
            x = sample.get(STATE_ENUM, sample.T-1)
            end_ee = x[self.state_inds[pname, 'pose']] + g
            attr_dict[(tname, 'value')] = end_ee - g
            for t in range(sample.T):
                x = sample.get(STATE_ENUM, t)
                ee = x[self.state_inds['pr2', 'pose']]
                sample.set(END_POSE_ENUM, end_ee - ee, t)
        return attr_dict


    def relabel_goal(self, sample):
        X = sample.get_X(sample.T-1)
        plan = self.plans.values()[0]
        goal = self.target_vecs[0].copy()
        for pname, aname in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            attr = X[self.state_inds[pname, aname]]
            if ('{0}_end_target'.format(pname), 'value') in self.target_inds:
                goal[self.target_inds[('{0}_end_target'.format(pname), 'value')]] = attr
        prim_choices = self.prob.get_prim_choices()
        only_goal = np.concatenate([goal[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]])
        return goal, only_goal


    def replace_cond(self, cond, curric_step=-1):
        self.init_vecs[cond], self.targets[cond] = self.prob.get_random_initial_state_vec(self.config, self.targets, self.dX, self.state_inds, 1)
        self.init_vecs[cond], self.targets[cond] = self.init_vecs[cond][0], self.targets[cond][0]
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        prim_choices = self.prob.get_prim_choices()
        if OBJ_ENUM in prim_choices and curric_step > 0:
            i = 0
            step = (curric_step + 1) // 2
            inds = np.random.permutation(range(len(prim_choices[OBJ_ENUM])))
            for j in inds:
                obj = prim_choices[OBJ_ENUM][j]
                if '{0}_end_target'.format(obj) not in self.targets[cond]: continue
                if i >= len(prim_choices[OBJ_ENUM]) - step: break
                self.x0[cond][self.state_inds[obj, 'pose']] = self.targets[cond]['{0}_end_target'.format(obj)]
                i += 1
            if curric_step % 2 and step < len(prim_choices[OBJ_ENUM]):
                self.x0[cond][self.state_inds['pr2', 'pose']] = self.x0[cond][self.state_inds['can{0}'.format(inds[step]), 'pose']] - np.array([0, 0.601])


        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]



