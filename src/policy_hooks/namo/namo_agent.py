import copy
import sys
import time
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

from baxter_gym.envs import MJCEnv

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
    def __init__(self, hyperparams):
        super(NAMOSortingAgent, self).__init__(hyperparams)

        self.robot_height = 1
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


    def sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True):
        x0 = state[self._x_data_idx[STATE_ENUM]]
        task = tuple(task)
        plan = self.plans[task]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon - 1
        sample = Sample(self)
        sample.init_t = 0
        prim_choices = self.prob.get_prim_choices()
        obj = prim_choices[OBJ_ENUM][task[1]]
        targ = prim_choices[TARG_ENUM][task[2]]
        obj_param = plan.params[obj]
        targ_param = plan.params[targ]

        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for target_name in self.targets[condition]:
            target = plan.params[target_name]
            target.value[:,0] = self.targets[condition][target.name]
            target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = 5e0 * generate_noise(self.T, self.dU, self._hyperparams)
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
            task_vec[task[0]] = 1.
            sample.task_ind = task[0]
            sample.set(TASK_ENUM, task_vec, t)
            sample.set(TARGETS_ENUM, target_vec.copy(), t)

            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            targ_vec[task[2]] = 1.
            sample.obj_ind = task[1]
            sample.targ_ind = task[2]
            sample.set(OBJ_ENUM, obj_vec, t)
            sample.set(TARG_ENUM, targ_vec, t)

            ee_pose = X[self.state_inds['pr2', 'pose']]
            obj_pose = X[self.state_inds[obj, 'pose']] - X[self.state_inds['pr2', 'pose']]
            targ_pose = self.targets[condition][targ] - X[self.state_inds['pr2', 'pose']]
            sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
            sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)
            sample.set(EE_ENUM, ee_pose, t)
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
                print 'Replacing nan in action.'
                U[np.isnan(U)] = 0.0
            if np.any(np.abs(U) == np.inf):
                print 'Replacing inf in action.'
                U[np.abs(U) == np.inf] = 0.0
            if np.all(np.abs(U) < 0.001):
                sample.use_ts[t] = 0
            # robot_start = X[plan.state_inds['pr2', 'pose']]
            # robot_vec = U[plan.action_inds['pr2', 'pose']] - robot_start
            # if np.sum(np.abs(robot_vec)) != 0 and np.linalg.norm(robot_vec) < 0.005:
            #     U[plan.action_inds['pr2', 'pose']] = robot_start + 0.1 * robot_vec / np.linalg.norm(robot_vec)
            sample.set(ACTION_ENUM, U.copy(), t)
                # import ipdb; ipdb.set_trace()
            
            # self.traj_hist.append(U)
            # while len(self.traj_hist) > self.hist_len:
            #     self.traj_hist.pop(0)


            self.run_policy_step(U, X, self.plans[task], t, obj)
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
        angles = 2 * np.pi * np.array(range(n_dirs), dtype='float32') / n_dirs
        rays = np.zeros((n_dirs, 6))
        for i in range(n_dirs):
            a = angles[i]
            ray = np.array([np.cos(a), np.sin(a)])
            rays[i, :2] = pr2.pose[:,t] + (pr2.geom.radius+0.01)*ray
            rays[i, 3:5] = 100 * ray
        pr2.openrave_body.set_pose(pr2.pose[:,t])

        for p_name in plan.params:
            p = plan.params[p_name]
            if p.is_symbol() or p is pr2: continue
            if (p_name, 'pose') in self.state_inds:
                p.openrave_body.set_pose(p.pose[:,t])
            else:
                p.openrave_body.set_pose(p.pose[:,0])
        is_hits, hits = self.env.CheckCollisionRays(rays, None)
        dists = np.linalg.norm(hits[:,:3]-rays[:,:3], axis=1)
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


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0):
        success = False
        x0 = state[self._x_data_idx[STATE_ENUM]]

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task] 
        prim_choices = get_prim_choices()
        targets = [prim_choices[OBJ_ENUM][task[1]], prim_choices[TARG_ENUM][task[2]]]
        set_params_attrs(plan.params, plan.state_inds, x0, 0)

        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        for target in self.targets[condition]:
            plan.params[target].value[:,0] = self.targets[condition][target]

        if targets[1] in self.targets[condition]:
            plan.params['{0}_end_target'.format(targets[0])].value[:,0] = self.targets[condition][targets[1]]

        run_solve = True
        for param_name in plan.params:
            if param_name == 'pr2': continue
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if np.all(np.abs(param.pose[:,0] - self.targets[condition][targets[1]]) < 0.01):
                run_solve = False
                break

        if task == 'grasp':
            plan.params[targets[1]].value[:,0] = plan.params[targets[0]].pose[:,0]
        
        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        dist = plan.params['pr2'].geom.radius + plan.params[targets[0]].geom.radius + dsafe
        if task == 'putdown':
            plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1]].value[:,0] - [0, dist]
        if task == 'grasp':
            plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1]].value[:,0] - [0, dist+0.2]


        prim_vals = self.get_prim_value(condition, state, task) 

        try:
            if run_solve:
                success = self.solver._backtrack_solve(plan, n_resamples=4, traj_mean=traj_mean, inf_f=inf_f, time_limit=120)
            else:
                success = False
        except Exception as e:
            print e
            traceback.print_exception(*sys.exc_info())
            success = False

        print 'Planning succeeded' if success else 'Planning failed'

        # if not success:
        #     print state
        #     # for action in plan.actions:
        #         # try:
        #         #     print plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)
        #         # except:
        #         #     pass
        #     print '\n\n'

        try:
            if not len(failed_preds):
                for action in plan.actions:
                    failed_preds += [(pred, targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
        except:
            pass


        if not success:
            sample = Sample(self)
            for i in range(len(self.prim_dims.keys())):
                enum = self.prim_dims.keys()[i]
                vec = np.zeros((self.prim_dims[enum]))
                vec[task[i]] = 1.
                sample.set(enum, vec, 0)

            sample.set(STATE_ENUM, x0.copy(), 0)
            sample.set(OBJ_POSE_ENUM, x0[self.state_inds[targets[0], 'pose']], 0)
            sample.set(TARG_POSE_ENUM, self.targets[condition][targets[1]], 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), 0)
            sample.set(TARGETS_ENUM, self.target_vecs[condition].copy(), 0)
            sample.condition = condition
            sample.task = task
            return sample, failed_preds, success

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    for param, attr in plan.action_inds:
                        U[plan.action_inds[param, attr]] = getattr(plan.params[param], attr)[:, t+1] - getattr(plan.params[param], attr)[:, t]
                    # U[plan.action_inds['pr2', 'pose']] = plan.params['pr2'].pose[:, t+1] - plan.params['pr2'].pose[:, t]
                    # U[plan.action_inds['pr2', 'gripper']] = plan.params['pr2'].gripper[:, t+1]

                return U

        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)
        # self.optimal_samples[task].append(sample)
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


    def fill_sample(self, sample, state, t, task_ind):
        plan = self.plans[task_ind]
        ee_pose = state[self.state_inds['pr2', 'pose']]
        sample.set(EE_ENUM, ee_pose.copy(), t)
        sample.set(STATE_ENUM, state, t)

        if LIDAR_ENUM in self._hyperparams['obs_include']:
            plan = self.plans.values()[0]
            set_params_attrs(plan.params, plan.state_inds, state, t)
            lidar = self.dist_obs(plan, 0)
            sample.set(LIDAR_ENUM, lidar.flatten(), t)


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
        options = get_prim_choices()
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


    def goal_f(self, condition, state):
        cost = 0
        plan = self.plans.values()[0]
        for param in plan.params.values():
            if param._type == 'Can':
                dist = np.sum((X[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                cost += dist if dist > 0.01 else 0

        return cost


    def cost_f(self, Xs, task, condition, active_ts=None, debug=False):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
        plan = self.plans[task]
        tol = 1e-3
        targets = self.targets[condition]

        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)

        if active_ts == None:
            active_ts = (1, plan.horizon-1)

        for t in range(active_ts[0], active_ts[1]+1):
            set_params_attrs(plan.params, plan.state_inds, Xs[t-active_ts[0]], t)

        for param in plan.params:
            if plan.params[param]._type == 'Can':
                plan.params['{0}_init_target'.format(param)].value[:,0] = plan.params[param].pose[:,0]
                plan.params['{0}_end_target'.format(param)].value[:,0] = targets['{0}_end_target'.format(param)]

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        plan.params['robot_end_pose'].value[:,0] = plan.params['pr2'].pose[:,-1]
        plan.params['{0}_init_target'.format(params[0].name)].value[:,0] = plan.params[params[0].name].pose[:,0]
        plan.params['{0}_end_target'.format(params[0].name)].value[:,0] = targets['{0}_end_target'.format(params[0].name)]

        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        if debug:
            print failed_preds

        cost = 0
        print plan.actions, failed_preds
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    if viol is not None:
                        cost += np.max(viol)
                except:
                    pass

        return cost
    

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


    def perturb_solve(self, sample, perturb_var=0.05, inf_f=None):
        state = sample.get_X(t=0)
        condition = sample.get(condition)
        task = sample.task
        out = self.solve_sample_opt_traj(state, task, condition, traj_mean=sample.get_U(), inf_f=inf_f, mp_var=perturb_var)
        return out


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean)

        exclude_targets = []
        plan = self.plans[task]
        act_traj = np.zeros((plan.horizon, self.dU))
        pr2 = plan.params['pr2']
        for t in range(plan.horizon-1):
            act_traj[t, self.action_inds['pr2', 'pose']] = pr2.pose[:,t+1] - pr2.pose[:,t]
            act_traj[t, self.action_inds['pr2', 'gripper']] = pr2.gripper[:,t+1]
        act_traj[-1] = act_traj[-2]

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, act_traj), condition, state, task, noisy=False,)
        # self.optimal_samples[task].append(sample)
        sample.set_ref_X(opt_traj)
        sample.set_ref_U(sample.get_U())
        return sample


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        targets = get_prim_choices[TARG_ENUM]
        state = state[self._x_data_idx[STATE_ENUM]]
        params = self.plans.values()[0].params

        return hl_plan_for_state(state, targets, plan_id, params, self.state_inds, failed_preds)
