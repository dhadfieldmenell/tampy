import copy
import sys
import time
import traceback

import pickle as pickle

import ctypes

import numpy as np
import scipy.interpolate

import xml.etree.ElementTree as xml

from sco.expr import *

import core.util_classes.common_constants as const
if const.USE_OPENRAVE:
    pass
else:
    import pybullet as P


from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from policy_hooks.sample_list import SampleList

import baxter_gym
from baxter_gym.envs import MJCEnv

import core.util_classes.items as items
from core.util_classes.namo_grip_predicates import dsafe, NEAR_TOL, dmove, HLGraspFailed, HLTransferFailed, HLPlaceFailed
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer

from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.tamp_eval_funcs import *
# from policy_hooks.namo.sorting_prob_4 import *
from policy_hooks.namo.namo_agent import NAMOSortingAgent


MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100
GRIP_TOL = 0.
MIN_STEP = 1e-2
LIDAR_DIST = 2.
# LIDAR_DIST = 1.5
DSAFE = 5e-1
MAX_STEP = max(1.5*dmove, 1)
LOCAL_FRAME = True
NAMO_XML = baxter_gym.__path__[0] + '/robot_info/lidar_namo.xml'


class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise):
        u = np.zeros(self.dU)
        if t < len(self.opt_traj) - 1:
            for param, attr in self.action_inds:
                if attr == 'gripper':
                    u[self.action_inds[param, attr]] = self.opt_traj[t, self.state_inds[param, attr]]
                elif attr == 'pose':
                    x, y = self.opt_traj[t+1, self.state_inds['pr2', 'pose']]
                    curx, cury = X[self.state_inds['pr2', 'pose']]
                    theta = -self.opt_traj[t, self.state_inds['pr2', 'theta']][0]
                    if LOCAL_FRAME:
                        relpos = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).dot([x-curx, y-cury])
                    else:
                        relpos = [x-curx, y-cury]
                    u[self.action_inds['pr2', 'pose']] = relpos
                elif attr == 'vel':
                    vel = self.opt_traj[t+1, self.state_inds['pr2', 'vel']] # np.linalg.norm(self.opt_traj[t+1, inds]-X[inds])
                    vel = np.linalg.norm(self.opt_traj[t+1, self.state_inds['pr2', 'pose']] - X[self.state_inds['pr2', 'pose']])
                    if self.opt_traj[t+1, self.state_inds['pr2', 'vel']] < 0:
                        vel *= -1
                    u[self.action_inds[param, attr]] = vel
                elif attr == 'theta':
                    u[self.action_inds[param, attr]] = self.opt_traj[t+1, self.state_inds[param, attr]] - X[self.state_inds[param, attr]]
                else:
                    u[self.action_inds[param, attr]] = self.opt_traj[t+1, self.state_inds[param, attr]] - X[self.state_inds[param, attr]]
        else:
            u[self.action_inds['pr2', 'gripper']] = self.opt_traj[-1, self.state_inds['pr2', 'gripper']]
        if np.any(np.isnan(u)):
            u[np.isnan(u)] = 0.
        return u


class NAMOGripAgent(NAMOSortingAgent):
    def __init__(self, hyperparams):
        super(NAMOSortingAgent, self).__init__(hyperparams)

        self.optimal_pol_cls = optimal_pol
        for plan in list(self.plans.values()):
            for t in range(plan.horizon):
                plan.params['obs0'].pose[:,t] = plan.params['obs0'].pose[:,0]

        self.check_col = hyperparams['master_config'].get('check_col', True)
        self.robot_height = 1
        self.use_mjc = hyperparams.get('use_mjc', False)
        self.vel_rat = 0.05
        wall_dims = OpenRAVEBody.get_wall_dims('closet')
        config = {
            'obs_include': ['can{0}'.format(i) for i in range(hyperparams['num_objs'])],
            'include_files': [NAMO_XML],
            'include_items': [],
            'view': False,
            'sim_freq': 50,
            'timestep': 0.002,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height']),
            'step_mult': 5e0,
            'act_jnts': ['robot_x', 'robot_y', 'robot_theta', 'right_finger_joint', 'left_finger_joint']
        }

        self.main_camera_id = 0
        # colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]
        colors = [[0.9, 0, 0, 1], [0, 0.9, 0, 1], [0, 0, 0.9, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices(self.task_list)
        for name in prim_options[OBJ_ENUM]:
            if name =='pr2': continue
            cur_color = colors.pop(0)
            # items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.4), 'rgba': tuple(cur_color), 'mass': 10.})
            items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 10.})
            items.append({'name': '{0}_end_target'.format(name), 'type': 'box', 'is_fixed': True, 'pos': (0, 0, 1.5), 'dimensions': (NEAR_TOL, NEAR_TOL, 0.05), 'rgba': tuple(cur_color), 'mass': 1.})
        for i in range(len(wall_dims)):
            dim, next_trans = wall_dims[i]
            next_trans[0,3] -= 3.5
            next_dim = dim # [dim[1], dim[0], dim[2]]
            pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
            items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

        no = self._hyperparams['num_objs']
        nt = self._hyperparams['num_targs']
        config['load_render'] = hyperparams['master_config'].get('load_render', False)
        config['xmlid'] = '{0}_{1}_{2}_{3}'.format(self.process_id, self.rank, no, nt)
        self.mjc_env = MJCEnv.load_config(config)
        self.targ_labels = {i: np.array(self.prob.END_TARGETS[i]) for i in range(len(self.prob.END_TARGETS))}
        self.targ_labels.update({i: self.targets[0]['aux_target_{0}'.format(i-no)] for i in range(no, no+self.prob.n_aux)})


    def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None):
        assert not np.any(np.isnan(state))
        start_t = time.time()
        # self.reset_to_state(state)
        x0 = state[self._x_data_idx[STATE_ENUM]].copy()
        task = tuple(task)
        if self.discrete_prim:
            plan = self.plans[task]
        else:
            plan = self.plans[task[0]]
        if hor is None:
            if task_f is None:
                self.T = plan.horizon
            else:
                self.T = max([p.horizon for p in list(self.plans.values())])
        else:
            self.T = hor
        sample = Sample(self)
        sample.init_t = 0
        col_ts = np.zeros(self.T)

        prim_choices = self.prob.get_prim_choices(self.task_list)
        target_vec = np.zeros((self.target_dim,))

        n_steps = 0
        end_state = None
        cur_state = self.get_state() # x0
        sample.task = task
        for t in range(0, self.T):
            noise_full = np.zeros((self.dU,))
            self.fill_sample(condition, sample, cur_state, t, task, fill_obs=True)
            if task_f is not None:
                prev_task = task
                task = task_f(sample, t, task)
                if task not in self.plans:
                    task = self.task_to_onehot[task[0]]
                self.fill_sample(condition, sample, cur_state, t, task, fill_obs=False)

            grasp = np.array([0, -0.601])
            if GRASP_ENUM in prim_choices and self.discrete_prim:
                grasp = self.set_grasp(grasp, task[3])

            X = cur_state.copy()
            U_full = policy.act(X, sample.get_obs(t=t).copy(), t, noise_full)
            U_nogrip = U_full.copy()
            U_nogrip[self.action_inds['pr2', 'gripper']] = 0.
            if np.all(np.abs(U_nogrip)) < 1e-2:
                self._noops += 1
                self.eta_scale = 1. / np.log(self._noops+2)
            else:
                self._noops = 0
                self.eta_scale = 1.
            if len(self._prev_U): self._prev_U = np.r_[self._prev_U[1:], [U_full]]
            sample.set(NOISE_ENUM, noise_full, t)
            # U_full = np.clip(U_full, -MAX_STEP, MAX_STEP)
            suc, col = self.run_policy_step(U_full, cur_state)
            objname = prim_choices[OBJ_ENUM][task[1]]
            tname = prim_choices[TASK_ENUM][task[0]]
            if tname.find('transfer') and len(self._col) == 1 and self._col[0] == objanme:
                col = 0
            col_ts[t] = col

            if self.master_config['local_retime']:
                dist = np.linalg.norm(U_full[self.action_inds['pr2', 'pose']])
                if dist > self.master_config['velocity']:
                    U_old = U_full
                    U_full *= self.master_config['velocity'] / dist
                    U_full[self.action_inds['pr2', 'gripper']] = U_old[self.action_inds['pr2', 'gripper']]

            sample.set(ACTION_ENUM, U_full, t)

            new_state = self.get_state()
            if len(self._x_delta)-1: self._x_delta = np.r_[self._x_delta[1:], [new_state]]
            if len(self._prev_task)-1:
                self._prev_task = np.r_[self._prev_task[1:], [sample.get_prim_out(t=t)]]

            if np.all(np.abs(cur_state - new_state) < 1e-3):
                sample.use_ts[t] = 0

            cur_state = new_state

        sample.end_state = new_state # end_state if end_state is not None else sample.get_X(t=self.T-1)
        sample.task_cost = self.goal_f(condition, sample.end_state)
        sample.prim_use_ts[:] = sample.use_ts[:]
        sample.col_ts = col_ts
        return sample


    def dist_obs(self, plan, t, n_dirs=-1, ignore=[], return_rays=False, extra_rays=[]):
        if n_dirs <= 0:
            n_dirs = self.n_dirs
        n_dirs = n_dirs // 2
        pr2 = plan.params['pr2']
        obs = 1e1*np.ones(n_dirs)
        angles = 2 * np.pi * np.array(list(range(n_dirs)), dtype='float32') / n_dirs
        rays = np.zeros((n_dirs, 6))
        rays[:, 2] = 0.4
        for i in range(n_dirs):
            a = angles[i]
            ray = np.array([np.cos(a), np.sin(a)])
            rays[i, :2] = pr2.pose[:,t]
            rays[i, 3:5] = LIDAR_DIST * ray

        rot = plan.params['pr2'].theta[0,t]
        rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        far_pt = rot_mat.dot([0, 1.])
        far_rays = rays.copy()
        far_rays[:, :2] = (pr2.pose[:,t] + far_pt).reshape((1,2))
        rays = np.r_[rays, far_rays]
        if len(extra_rays):
            rays = np.concatenate([rays, extra_rays], axis=0)

        for params in [plan.params]:
            for p_name in params:
                p = params[p_name]

                if p.is_symbol():
                    if hasattr(p, 'openrave_body') and p.openrave_body is not None:
                        p.openrave_body.set_pose([0, 0, -5])
                elif (p_name, 'pose') in self.state_inds:
                    p.openrave_body.set_pose(plan.params[p_name].pose[:,t])
                else:
                    p.openrave_body.set_pose(plan.params[p_name].pose[:,0])

        pr2.openrave_body.set_pose([0, 0, -5]) # Get this out of the way
        for name in ignore:
            plan.params[name].openrave_body.set_pose([0, 0, -5])


        if const.USE_OPENRAVE:
            is_hits, hits = self.env.CheckCollisionRays(rays, None)
            dists = np.linalg.norm(hits[:,:2]-rays[:,:2], axis=1)
            for i in range(len(is_hits)):
                dists[i] = dists[i] if is_hits[i] else LIDAR_DIST
        else:
            P.stepSimulation()
            # _, _, hit_frac, hit_pos, hit_normal = P.rayTestBatch(rays[:,:3], rays[:,:3]+rays[:,3:])
            hits = P.rayTestBatch(rays[:,:3], rays[:,:3]+rays[:,3:])
            dists = LIDAR_DIST * np.array([h[2] for h in hits])

        # dists[np.abs(dists) > LIDAR_DIST] = LIDAR_DIST
        # dists[not np.array(is_hits)] = LIDAR_DIST
        if return_rays:
            return dists, rays

        return dists


    #def run_policy_step(self, u, x, plan, t, obj, grasp=None):
    def run_policy_step(self, u, x):
        self._col = []
        poses = {}
        for pname, aname in self.state_inds:
            if aname != 'pose' or pname.find('can') < 0: continue
            poses[pname] = self.mjc_env.get_item_pos(pname)
        cmd_theta = u[self.action_inds['pr2', 'theta']][0]
        if ('pr2', 'pose') not in self.action_inds:
            cmd_vel = u[self.action_inds['pr2', 'vel']]
            self.mjc_env.set_user_data('vel', cmd_vel)
            cur_theta = x[self.state_inds['pr2', 'theta']][0]
            cmd_x, cmd_y = -cmd_vel*np.sin(cur_theta), cmd_vel*np.cos(cur_theta)
        else:
            cur_theta = x[self.state_inds['pr2', 'theta']][0]
            rel_x, rel_y = u[self.action_inds['pr2', 'pose']]
            if LOCAL_FRAME:
                cmd_x, cmd_y = np.array([[np.cos(cur_theta), -np.sin(cur_theta)], [np.sin(cur_theta), np.cos(cur_theta)]]).dot([rel_x, rel_y])
            else:
                cmd_x, cmd_y = rel_x, rel_y

        nsteps = int(max(abs(cmd_x), abs(cmd_y)) / self.vel_rat) + 1
        # nsteps = min(nsteps, 10)
        gripper = u[self.action_inds['pr2', 'gripper']][0]
        if gripper < 0:
            gripper = -0.1
        else:
            gripper = 0.1
        cur_x, cur_y, _ = self.mjc_env.get_item_pos('pr2') # x[self.state_inds['pr2', 'pose']]
        for n in range(nsteps+1):
            x = cur_x + float(n)/nsteps * cmd_x
            y = cur_y + float(n)/nsteps * cmd_y
            theta = cur_theta + float(n)/nsteps * cmd_theta
            ctrl_vec = np.array([x, y, theta, 5*gripper, 5*gripper])
            self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')
        self.mjc_env.step(ctrl_vec, mode='velocity')

        new_poses = {}
        for pname, aname in self.state_inds:
            if aname != 'pose' or pname.find('can') < 0: continue
            new_poses[pname] = self.mjc_env.get_item_pos(pname)
        for pname in poses:
            if np.any(np.abs(poses[pname]-new_poses[pname])) > 5e-2:
                self._col.append(pname)
        col = 1 if self(self._col > 0) else 0
        return True, col


    def get_state(self):
        x = np.zeros(self.dX)
        for pname, attr in self.state_inds:
            if attr == 'pose':
                val = self.mjc_env.get_item_pos(pname)
                x[self.state_inds[pname, attr]] = val[:2]
            elif attr == 'rotation':
                val = self.mjc_env.get_item_rot(pname)
                x[self.state_inds[pname, attr]] = val
            elif attr == 'gripper':
                vals = self.mjc_env.get_joints(['left_finger_joint','right_finger_joint'])
                val1 = vals['left_finger_joint']
                val2 = vals['right_finger_joint']
                val = (val1 + val2) / 2.
                x[self.state_inds[pname, attr]] = 0.1 if val > 0 else -0.1
            elif attr == 'theta':
                val = self.mjc_env.get_joints(['robot_theta'])
                x[self.state_inds[pname, 'theta']] = val['robot_theta']
            elif attr == 'vel':
                val = self.mjc_env.get_user_data('vel', 0.)
                x[self.state_inds[pname, 'vel']] = val

        assert not np.any(np.isnan(x))
        return x


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        mp_state = mp_state.copy()
        plan = self.plans[task]
        ee_pose = mp_state[self.state_inds['pr2', 'pose']]
        if targets is None:
            targets = self.target_vecs[cond].copy()

        sample.set(EE_ENUM, ee_pose, t)
        theta = mp_state[self.state_inds['pr2', 'theta']][0]
        sample.set(THETA_ENUM, mp_state[self.state_inds['pr2', 'theta']], t)
        dirvec = np.array([-np.sin(theta), np.cos(theta)])
        sample.set(THETA_VEC_ENUM, dirvec, t)
        sample.set(VEL_ENUM, mp_state[self.state_inds['pr2', 'vel']], t)
        sample.set(STATE_ENUM, mp_state, t)
        sample.set(GRIPPER_ENUM, mp_state[self.state_inds['pr2', 'gripper']], t)
        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)
            sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)
        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)
        onehot_task = np.zeros(self.sensor_dims[ONEHOT_TASK_ENUM])
        onehot_task[self.task_to_onehot[task]] = 1.
        sample.set(ONEHOT_TASK_ENUM, onehot_task, t)

        task_ind = task[0]
        obj_ind = task[1]
        targ_ind = task[2]
        prim_choices = self.prob.get_prim_choices(self.task_list)

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.task_ind = task[0]
        sample.set(TASK_ENUM, task_vec, t)

        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        # sample.set(DONE_ENUM, self.goal_f(cond, mp_state), t)
        grasp = np.array([0, -0.601])
        theta = mp_state[self.state_inds['pr2', 'theta']][0]
        if self.discrete_prim:
            sample.set(FACTOREDTASK_ENUM, np.array(task), t)
            if GRASP_ENUM in prim_choices:
                grasp = self.set_grasp(grasp, task[3])
                grasp_vec = np.zeros(self._hyperparams['sensor_dims'][GRASP_ENUM])
                grasp_vec[task[3]] = 1.
                sample.set(GRASP_ENUM, grasp_vec, t)

            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            targ_vec[task[2]] = 1.
            if self.task_list[task[0]] == 'moveto':
                obj_vec[task[1]] = 1.
                targ_vec[:] = 1. / len(targ_vec)
            #elif self.task_list[task[0]] == 'transfer':
            #    obj_vec[:] = 1. / len(obj_vec)
            #    targ_vec[task[2]] = 1.
            #elif self.task_list[task[0]] == 'place':
            #    obj_vec[:] = 1. / len(obj_vec)
            #    targ_vec[task[2]] = 1.
            sample.obj_ind = task[1]
            sample.targ_ind = task[2]
            sample.set(OBJ_ENUM, obj_vec, t)
            sample.set(TARG_ENUM, targ_vec, t)

            obj_name = list(prim_choices[OBJ_ENUM])[obj_ind]
            targ_name = list(prim_choices[TARG_ENUM])[targ_ind]
            obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[self.state_inds['pr2', 'pose']]
            targ_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds['pr2', 'pose']]
            targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds[obj_name, 'pose']]
        else:
            obj_pose = label[1] - mp_state[self.state_inds['pr2', 'pose']]
            targ_pose = label[1] - mp_state[self.state_inds['pr2', 'pose']]

        if LOCAL_FRAME:
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
            obj_pose = rot.dot(obj_pose)
            targ_pose = rot.dot(targ_pose)
            targ_off_pose = rot.dot(targ_off_pose)
        else:
            rot = np.eye(2)
        # if task[0] == 1:
        #     obj_pose = np.zeros_like(obj_pose)
        sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)

        # if task[0] == 0:
        #     targ_pose = np.zeros_like(targ_pose)
        sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)

        sample.task = task
        sample.obj = task[1]
        sample.targ = task[2]
        sample.condition = cond
        sample.task_name = self.task_list[task[0]]
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(GOAL_ENUM, np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]]), t)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
            sample.set(ONEHOT_GOAL_ENUM, self.onehot_encode_goal(sample.get(GOAL_ENUM, t)), t)
        sample.targets = targets.copy()

        if self.task_list[task[0]] == 'moveto':
            sample.set(END_POSE_ENUM, obj_pose, t)
            #sample.set(END_POSE_ENUM, obj_pose.copy(), t)
        if self.task_list[task[0]] == 'transfer':
            sample.set(END_POSE_ENUM, targ_pose, t)
            #sample.set(END_POSE_ENUM, targ_pose.copy(), t)
        if self.task_list[task[0]] == 'place':
            sample.set(END_POSE_ENUM, targ_pose, t)
            #sample.set(END_POSE_ENUM, targ_pose.copy(), t)
        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose, t)
            sample.set(TARG_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

        if INGRASP_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - mp_state[self.state_inds['pr2', 'pose']] - grasp) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(INGRASP_ENUM, vec, t=t)

        if ATGOAL_ENUM in self._hyperparams['sensor_dims']:
            vec = np.zeros(len(prim_choices[OBJ_ENUM]))
            for i, o in enumerate(prim_choices[OBJ_ENUM]):
                if np.all(np.abs(mp_state[self.state_inds[o, 'pose']] - targets[self.target_inds['{0}_end_target'.format(o), 'value']]) < NEAR_TOL):
                    vec[i] = 1.
            sample.set(ATGOAL_ENUM, vec, t=t)

        if fill_obs:
            if LIDAR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                set_params_attrs(plan.params, plan.state_inds, mp_state, 0)
                lidar = self.dist_obs(plan, 1)
                sample.set(LIDAR_ENUM, lidar.flatten(), t)

            if MJC_SENSOR_ENUM in self._hyperparams['obs_include']:
                plan = list(self.plans.values())[0]
                sample.set(MJC_SENSOR_ENUM, self.mjc_env.get_sensors(), t)

            if IM_ENUM in self._hyperparams['obs_include']:
                im = self.mjc_env.render(height=self.image_height, width=self.image_width)
                sample.set(IM_ENUM, im.flatten(), t)


    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._x_delta[:] = x.reshape((1,-1))
        self.eta_scale = 1.
        self._noops = 0
        self.mjc_env.reset()
        xval, yval = mp_state[self.state_inds['pr2', 'pose']]
        grip = x[self.state_inds['pr2', 'gripper']][0]
        theta = x[self.state_inds['pr2', 'theta']][0]
        self.mjc_env.set_user_data('vel', 0.)
        self.mjc_env.set_joints({'robot_x': xval, 'robot_y': yval, 'left_finger_joint': grip, 'right_finger_joint': grip, 'robot_theta': theta}, forward=False)
        for param_name, attr in self.state_inds:
            if param_name == 'pr2': continue
            if attr == 'pose':
                pos = mp_state[self.state_inds[param_name, 'pose']].copy()
                targ = self.target_vecs[0][self.target_inds['{0}_end_target'.format(param_name), 'value']]
                self.mjc_env.set_item_pos(param_name, np.r_[pos, 0.5], forward=False)
                self.mjc_env.set_item_pos('{0}_end_target'.format(param_name), np.r_[targ, 1.5], forward=False)
        self.mjc_env.physics.forward()


    def set_to_targets(self, condition=0):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        objs = prim_choices[OBJ_ENUM]
        for obj_name in objs:
            self.mjc_env.set_item_pos(obj_name, np.r_[self.targets[condition]['{0}_end_target'.format(obj_name)], 0], forward=False)
        self.mjc_env.physics.forward()


    def get_mjc_obs(self, x):
        self.reset_to_state(x)
        # return self.mjc_env.get_obs(view=False)
        return self.mjc_env.render()


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], targets=[], run_traj=True):
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
        if run_traj:
            sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_traj), condition, state, task, noisy=False, skip_opt=True)
        else:
            self.T = plan.horizon
            sample = Sample(self)
            for t in range(len(opt_traj)-1):
                pos = opt_traj[t][self.state_inds['pr2', 'pose']]
                pos_2 = opt_traj[t+1][self.state_inds['pr2', 'pose']]
                theta = opt_traj[t][self.state_inds['pr2', 'theta']]
                theta_2 = opt_traj[t+1][self.state_inds['pr2', 'theta']]
                vel = opt_traj[t+1][self.state_inds['pr2', 'vel']]
                grip = opt_traj[t][self.state_inds['pr2', 'gripper']]
                U = np.zeros(self.dU)
                # U[self.action_inds['pr2', 'pose']] = pos_2 - pos
                U[self.action_inds['pr2', 'vel']] = vel
                U[self.action_inds['pr2', 'theta']] = theta_2 - theta
                U[self.action_inds['pr2', 'gripper']] = grip
                sample.set(ACTION_ENUM, U, t=t)
                self.reset_to_state(opt_traj[t])
                self.fill_sample(condition, sample, opt_traj[t], t, task, fill_obs=True, targets=targets)
            if len(opt_traj)-1 < sample.T:
                for j in range(len(opt_traj)-1, sample.T):
                    sample.set(ACTION_ENUM, np.zeros_like(U), t=j)
                    self.reset_to_state(opt_traj[-1])
                    self.fill_sample(condition, sample, opt_traj[-1], j, task, fill_obs=True, targets=targets)
            sample.use_ts[-1] = 0.
            sample.prim_use_ts[-1] = 0.
            sample.prim_use_ts[len(opt_traj)-1:] = 0.
            sample.use_ts[len(opt_traj)-1:] = 0.
            sample.end_state = opt_traj[-1].copy()
            sample.set(NOISE_ENUM, np.zeros((sample.T, self.dU)))
            sample.task_cost = self.goal_f(condition, sample.end_state)
            sample.col_ts = np.zeros(sample.T)
        sample.set_ref_X(sample.get_X())
        sample.set_ref_U(sample.get_U())

        # for t in range(sample.T):
        #     if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 1e-3:
        #         sample.use_ts[t] = 0.

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
        # self.optimal_samples[self.task_list[task[0]]].append(sample)
        return sample


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0, targets=[], x_only=False, t_limit=60, n_resamples=5, out_coeff=None, smoothing=False, attr_dict=None):
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
        prim_choices = self.prob.get_prim_choices(self.task_list)
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

        plan.params['pr2'].pose[:, 0] = x0[self.state_inds['pr2', 'pose']]
        plan.params['pr2'].gripper[:, 0] = x0[self.state_inds['pr2', 'gripper']]
        plan.params['obs0'].pose[:] = plan.params['obs0'].pose[:,:1]

        run_solve = True

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        for param in list(plan.params.values()):
            for attr in param._free_attrs:
                if np.any(np.isnan(getattr(param, attr)[:,0])):
                    getattr(param, attr)[:,0] = 0

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
            # traceback.print_exception(*sys.exc_info())
            success = False

        self.solver.strong_transfer_coeff = old_out_coeff

        try:
            if not len(failed_preds):
                for action in plan.actions:
                    failed_preds += [(pred, t) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
        except:
            failed_preds += ['Nan in pred check for {0}'.format(action)]

        traj = np.zeros((plan.horizon, self.symbolic_bound))
        for pname, aname in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            inds = self.state_inds[pname, aname]
            for t in range(plan.horizon):
                traj[t][inds] = getattr(plan.params[pname], aname)[:,t]

        class _optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < len(traj)-1:
                    for param, attr in plan.action_inds:
                        if attr == 'pose':
                            U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] - X[plan.state_inds[param, attr]]
                        elif attr == 'gripper':
                            U[plan.action_inds[param, attr]] = traj[t][plan.state_inds[param, attr]]
                        elif attr == 'theta':
                            U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]] - traj[t][plan.state_inds[param, attr]]
                        elif attr == 'vel':
                            U[plan.action_inds[param, attr]] = traj[t+1][plan.state_inds[param, attr]]
                        else:
                            raise NotImplementedError
                if np.any(np.isnan(U)):
                    if success: print(('NAN in {0} plan act'.format(success)))
                    U[:] = 0.
                return U
        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, traj), condition, state, task, noisy=False, skip_opt=True)
        # sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False, skip_opt=True)

        # for t in range(sample.T):
        #     if np.all(np.abs(sample.get(ACTION_ENUM, t=t))) < 1e-3: sample.use_ts[t] = 0.

        traj = sample.get(STATE_ENUM)
        for param_name, attr in self.state_inds:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            diff = traj[:, self.state_inds[param_name, attr]].T - getattr(param, attr)
            # if np.any(np.abs(diff) > 1e-3): print(diff, param_name, attr, 'ERROR IN OPT ROLLOUT')

        # self.optimal_samples[self.task_list[task[0]]].append(sample)
        # print(sample.get_X())
        if not smoothing and self.debug:
            if not success:
                sample.use_ts[:] = 0.
                print(('Failed to plan for: {0} {1} smoothing? {2} {3}'.format(task, failed_preds, smoothing, state)))
                print('FAILED PLAN')
            else:
                print(('SUCCESSFUL PLAN for {0}'.format(task)))
        # else:
        #     print('Plan success for {0} {1}'.format(task, state))
        return sample, failed_preds, success

    def retime_traj(self, traj, vel=0.3, inds=None, minpts=10):
        new_traj = []
        if len(np.shape(traj)) == 2:
            traj = [traj]
        for step in traj:
            xpts = []
            fpts = []
            grippts= []
            d = 0
            if inds is None:
                #inds = np.r_[self.state_inds['pr2', 'vel'], \
                #             self.state_inds['pr2', 'pose']]
                inds = self.state_inds['pr2', 'pose']
            for t in range(len(step)):
                xpts.append(d)
                fpts.append(step[t])
                grippts.append(step[t][self.state_inds['pr2', 'gripper']])
                if t < len(step) - 1:
                    disp = np.linalg.norm(step[t+1][inds] - step[t][inds])
                    d += disp
            assert not np.any(np.isnan(xpts))
            assert not np.any(np.isnan(fpts))
            interp = scipy.interpolate.interp1d(xpts, fpts, axis=0, fill_value='extrapolate')
            grip_interp = scipy.interpolate.interp1d(np.array(xpts), grippts, kind='previous', bounds_error=False, axis=0)

            fix_pts = []
            if type(vel) is float:
                # x = np.arange(0, d+vel/2, vel)
                # npts = max(int(d/vel), minpts)
                # x = np.linspace(0, d, npts)

                x = []
                for i, d in enumerate(xpts):
                    if i == 0:
                        x.append(0)
                        fix_pts.append((len(x)-1, fpts[i]))
                    # elif xpts[i] - xpts[i-1] <= 1e-6:
                    #     continue
                    elif xpts[i] - xpts[i-1] <= vel:
                        x.append(x[-1] + xpts[i] - xpts[i-1])
                        fix_pts.append((len(x)-1, fpts[i]))
                    else:
                        n = max(2, int((xpts[i]-xpts[i-1])//vel))
                        for _ in range(n):
                            x.append(x[-1] + (xpts[i]-xpts[i-1])/float(n))
                        x[-1] = d
                        fix_pts.append((len(x)-1, fpts[i]))
                # x = np.cumsum(x)
            elif type(vel) is list:
                x = np.r_[0, np.cumsum(vel)]
            else:
                raise NotImplementedError('Velocity undefined')
            out = interp(x)
            grip_out = grip_interp(x)
            out[:, self.state_inds['pr2', 'gripper']] = grip_out
            out[0] = step[0]
            out[-1] = step[-1]
            for pt, val in fix_pts:
                out[pt] = val
            out = np.r_[out, [out[-1]]]
            if len(new_traj):
                new_traj = np.r_[new_traj, out]
            else:
                new_traj = out
            if np.any(np.isnan(out)): print(('NAN in out', out, x))
        return new_traj


    def set_symbols(self, plan, task, anum=0, cond=0):
        st, et = plan.actions[anum].active_timesteps
        targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]] == 'moveto':
            params[3].value[:,0] = params[0].pose[:,st]
            params[2].value[:,0] = params[1].pose[:,st]
        elif self.task_list[task[0]] == 'transfer':
            params[1].value[:,0] = params[0].pose[:,st]
            params[6].value[:,0] = params[3].pose[:,st]
        elif self.task_list[task[0]] == 'place':
            params[1].value[:,0] = params[0].pose[:,st]
            params[6].value[:,0] = params[3].pose[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,0]


    def goal(self, cond, targets=None):
        if self.goal_type == 'moveto':
            assert ('can1', 'pose') not in self.state_inds
            return '(NearGraspAngle  pr2 can0) '
        if targets is None:
            targets = self.target_vecs[cond]
        prim_choices = self.prob.get_prim_choices(self.task_list)
        goal = ''
        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            for ind in self.targ_labels:
                if np.all(np.abs(targ - self.targ_labels[ind]) < NEAR_TOL):
                    goal += '(Near {0} end_target_{1}) '.format(obj, ind)
                    break
        return goal


    def backtrack_solve(self, plan, anum=0, n_resamples=5):
        if self.hl_pol:
            prim_opts = self.prob.get_prim_choices(self.task_list)
            start = anum
            plan.state_inds = self.state_inds
            plan.action_inds = self.action_inds
            plan.dX = self.symbolic_bound
            plan.dU = self.dU
            success = False
            hl_success = True
            targets = self.target_vecs[0]
            for a in range(anum, len(plan.actions)):
                x0 = np.zeros_like(self.x0[0])
                st, et = plan.actions[a].active_timesteps
                fill_vector(plan.params, self.state_inds, x0, st)
                task = tuple(self.encode_action(plan.actions[a]))

                traj = []
                success = False
                policy = self.policies[self.task_list[task[0]]]
                path = []
                x = x0
                for i in range(3):
                    sample = self.sample_task(policy, 0, x.copy(), task, skip_opt=True)
                    path.append(sample)
                    x = sample.get_X(sample.T-1)
                    postcost = self.postcond_cost(sample, task, sample.T-1)
                    if postcost < 1e-3: break
                postcost = self.postcond_cost(sample, task, sample.T-1)
                if postcost > 0:
                    taskname = self.task_list[task[0]]
                    objname = prim_opts[OBJ_ENUM][task[1]]
                    targname = prim_opts[TARG_ENUM][task[2]]
                    graspname = prim_opts[GRASP_ENUM][task[3]]
                    obj = plan.params[objname]
                    targ = plan.params[targname]
                    grasp = plan.params[graspname]
                    if taskname.find('moveto') >= 0:
                        pred = HLGraspFailed('hlgraspfailed', [obj], ['Can'])
                    elif taskname.find('transfer') >= 0:
                        pred = HLTransferFailed('hltransferfailed', [obj, targ], ['Can', 'Target'])
                    elif taskname.find('place') >= 0:
                        pred = HLTransferFailed('hlplacefailed', [targ], ['Target'])
                    plan.hl_preds.append(pred)
                    hl_success = False
                    sucess = False
                    print('POSTCOND FAIL', plan.hl_preds)
                else:
                    print('POSTCOND SUCCESS')

                fill_vector(plan.params, self.state_inds, x0, st)
                self.set_symbols(plan, task, anum=a)
                try:
                    success = self.ll_solver._backtrack_solve(plan, anum=a, amax=a, n_resamples=n_resamples, init_traj=traj)
                except Exception as e:
                    traceback.print_exception(*sys.exc_info())
                    print(('Exception in full solve for', x0, task, plan.actions[a]))
                    success = False
                self.n_opt[task] = self.n_opt.get(task, 0) + 1

                if not success:
                    failed = plan.get_failed_preds((0, et))
                    if not len(failed):
                        continue
                    print(('Graph failed solve on', x0, task, plan.actions[a], 'up to {0}'.format(et), failed, self.process_id))
                    self.n_fail_opt[task] = self.n_fail_opt.get(task, 0) + 1
                    return False
                self.run_plan(plan, targets, amin=a, amax=a, record=False)
                if not hl_success: return False
                plan.hl_preds = []
            
            print('SUCCESS WITH LL POL + PR GRAPH')
            return True
        return super(NAMOSortingAgent, self).backtrack_solve(plan, anum, n_resamples)


