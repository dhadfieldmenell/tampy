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
from policy_hooks.namo.grip_agent import NAMOGripAgent


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


class NAMODoorAgent(NAMOGripAgent):
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
        colors = [[0.9, 0, 0, 1], [0, 0.9, 0, 1], [0, 0, 0.9, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices(self.task_list)
        for name in prim_options[OBJ_ENUM]:
            if name =='pr2': continue
            cur_color = colors.pop(0)
            items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 10.})
            items.append({'name': '{0}_end_target'.format(name), 'type': 'box', 'is_fixed': True, 'pos': (0, 0, 1.5), 'dimensions': (NEAR_TOL, NEAR_TOL, 0.05), 'rgba': tuple(cur_color), 'mass': 1.})
        for i in range(len(wall_dims)):
            dim, next_trans = wall_dims[i]
            next_trans[0,3] -= 3.5
            next_dim = dim # [dim[1], dim[0], dim[2]]
            pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
            items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

        items.append({'name': 'door', 'type': 'door_2d', 'handle_dims': (0.325, 0.4), 'door_dims': (0.75, 0.1, 0.4), 'hinge_pos': (-1., 3., 0.5), 'is_fixed': True})

        no = self._hyperparams['num_objs']
        nt = self._hyperparams['num_targs']
        config['load_render'] = hyperparams['master_config'].get('load_render', False)
        config['xmlid'] = '{0}_{1}_{2}_{3}'.format(self.process_id, self.rank, no, nt)
        self.mjc_env = MJCEnv.load_config(config)
        self.targ_labels = {i: np.array(self.prob.END_TARGETS[i]) for i in range(len(self.prob.END_TARGETS))}
        self.targ_labels.update({i: self.targets[0]['aux_target_{0}'.format(i-no)] for i in range(no, no+self.prob.n_aux)})

    def get_state(self):
        x = np.zeros(self.dX)
        for pname, attr in self.state_inds:
            if attr == 'pose':
                if pname.find('door') >= 0:
                    val = self.mjc_env.get_item_pos('door_base')
                    x[self.state_inds[pname, attr]] = val[:2]
                else:
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
                if pname.find('door') >= 0:
                    val = self.mjc_env.get_joints(['door_hinge'])
                    x[self.state_inds[pname, 'theta']] = val['door_hinge']
                else:
                    val = self.mjc_env.get_joints(['robot_theta'])
                    x[self.state_inds[pname, 'theta']] = val['robot_theta']
            elif attr == 'vel':
                val = self.mjc_env.get_user_data('vel', 0.)
                x[self.state_inds[pname, 'vel']] = val

        assert not np.any(np.isnan(x))
        return x


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        if task is None:
            task = list(self.plans.keys())[0]
        mp_state = mp_state.copy()
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task]
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
        sample.set(DOOR_ENUM, mp_state[self.state_inds['door', 'theta']], t)
        sample.set(GRIPPER_ENUM, mp_state[self.state_inds['pr2', 'gripper']], t)
        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)
            sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)
        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)
        
        prim_choices = self.prob.get_prim_choices(self.task_list)

        task_ind = task[0]
        task_name = self.task_list[task_ind]
        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.task_ind = task[0]
        sample.set(TASK_ENUM, task_vec, t)

        #post_cost = self.cost_f(sample.get_X(t=t), task, cond, active_ts=(sample.T-1, sample.T-1), targets=targets)
        #done = np.ones(1) if post_cost == 0 else np.zeros(1)
        #sample.set(DONE_ENUM, done, t)
        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        grasp = np.array([0, -0.601])
        onehottask = tuple([val for val in task if np.isscalar(val)])
        sample.set(FACTOREDTASK_ENUM, np.array(onehottask), t)

        theta = mp_state[self.state_inds['pr2', 'theta']][0]
        if LOCAL_FRAME:
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
        else:
            rot = np.eye(2)

        if OBJ_ENUM in prim_choices:
            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            if task_name.find('door') >= 0:
                obj_vec[:] = 1. / len(obj_vec)

            sample.obj_ind = task[1]
            obj_ind = task[1]
            obj_name = list(prim_choices[OBJ_ENUM])[obj_ind]
            sample.set(OBJ_ENUM, obj_vec, t)
            obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[self.state_inds['pr2', 'pose']]
            base_pos = obj_pose
            obj_pose = rot.dot(obj_pose)
            sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
            sample.obj = task[1]
            if self.task_list[task[0]].find('move') >= 0:
                sample.set(END_POSE_ENUM, obj_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, mp_state[self.state_inds[obj_name, 'pose']], t)

        if TARG_ENUM in prim_choices:
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            targ_vec[task[2]] = 1.
            if self.task_list[task[0]].find('door') >= 0 or self.task_list[task[0]].find('move') >= 0:
                targ_vec[:] = 1. / len(targ_vec)
            sample.targ_ind = task[2]
            targ_ind = task[2]
            targ_name = list(prim_choices[TARG_ENUM])[targ_ind]
            sample.set(TARG_ENUM, targ_vec, t)
            targ_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds['pr2', 'pose']]
            targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds[obj_name, 'pose']]
            base_pos = targ_pose
            targ_pose = rot.dot(targ_pose)
            targ_off_pose = rot.dot(targ_off_pose)
            sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)
            sample.targ = task[2]
            if self.task_list[task[0]].find('put') >= 0 or self.task_list[task[0]].find('leave') >= 0:
                sample.set(END_POSE_ENUM, targ_pose, t)
                sample.set(REL_POSE_ENUM, base_pos, t)
                sample.set(ABS_POSE_ENUM, targets[self.target_inds[targ_name, 'value']], t)

        if task_name.find('close_door') >= 0 or task_name.find('open_door') >= 0:
            theta = mp_sate[self.state_inds['door', 'theta']]
            handle_pos = mp_state[self.state_inds['door', 'pose']] + [1.5*np.sin(theta), 1.5*np.cos(theta)]
            targ_pose = handle_pos - mp_state[self.state_inds['pr2', 'pose']]
            sample.set(END_POSE_ENUM, rot.dot(targ_pose), t)
            sample.set(REL_POSE_ENUM, base_pos, t)
            sample.set(ABS_POSE_ENUM, handle_pos[:2], t)

        if ABS_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(ABS_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(ABS_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind] - mp_state[self.state_inds['pr2', 'pose']]), t)

        if REL_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(REL_POSE_ENUM)
            if ind < len(task) and not np.isscalar(task[ind]):
                sample.set(REL_POSE_ENUM, task[ind], t)
                sample.set(END_POSE_ENUM, rot.dot(task[ind]), t)

        if END_POSE_ENUM in prim_choices:
            ind = list(prim_choices.keys()).index(END_POSE_ENUM)
            if ind < len(task) and type(task[ind]) is not int:
                sample.set(END_POSE_ENUM, task[ind], t)

        sample.task = task
        sample.condition = cond
        sample.task_name = self.task_list[task[0]]
        sample.set(TARGETS_ENUM, targets.copy(), t)
        sample.set(GOAL_ENUM, np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]]), t)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
            sample.set(ONEHOT_GOAL_ENUM, self.onehot_encode_goal(sample.get(GOAL_ENUM, t)), t)
        sample.targets = targets.copy()

        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose, t)
            sample.set(TARG_ENUMS[i], targ, t)
            sample.set(TARG_DELTA_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

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

            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include']:
                im = self.mjc_env.render(height=self.image_height, width=self.image_width, view=self.view)
                im = (im - 128.) / 128.
                sample.set(IM_ENUM, im.flatten(), t)


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
        door_theta = x[self.state_inds['door', 'theta']][0]
        self.mjc_env.set_user_data('vel', 0.)
        self.mjc_env.set_joints({'robot_x': xval, 'robot_y': yval, 'left_finger_joint': grip, 'right_finger_joint': grip, 'robot_theta': theta, 'door_hinge': door_theta}, forward=False)
        for param_name, attr in self.state_inds:
            if param_name == 'pr2': continue
            if attr == 'pose':
                pos = mp_state[self.state_inds[param_name, 'pose']].copy()
                targ = self.target_vecs[0][self.target_inds['{0}_end_target'.format(param_name), 'value']]
                self.mjc_env.set_item_pos(param_name, np.r_[pos, 0.5], forward=False)
                self.mjc_env.set_item_pos('{0}_end_target'.format(param_name), np.r_[targ, 1.5], forward=False)
        self.mjc_env.physics.forward()


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
        onehot_task = tuple([val for val in task if np.isscalar(val)])
        plan = self.plans[onehot_task]
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


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None):
        st, et = plan.actions[anum].active_timesteps
        if targets is None:
            targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]] == 'moveto':
            params[3].value[:,0] = params[0].pose[:,st]
            params[2].value[:,0] = params[1].pose[:,st]
        elif self.task_list[task[0]] == 'put_in_closet':
            params[1].value[:,0] = params[0].pose[:,st]
            params[5].value[:,0] = params[3].pose[:,st]
        elif self.task_list[task[0]] == 'leave_closet':
            params[1].value[:,0] = params[0].pose[:,st]
            params[5].value[:,0] = params[3].pose[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,0]


    def goal(self, cond, targets=None):
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
        goal += '(DoorClosed Door)'
        return goal


    def backtrack_solve(self, plan, anum=0, n_resamples=5, rollout=False):
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
                    #graspname = prim_opts[GRASP_ENUM][task[3]]
                    obj = plan.params[objname]
                    targ = plan.params[targname]
                    #grasp = plan.params[graspname]
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
        return super(NAMOSortingAgent, self).backtrack_solve(plan, anum, n_resamples, rollout)


    def encode_plan(self, plan, permute=False):
        encoded = []
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for a in plan.actions:
            encoded.append(self.encode_action(a))

        encoded = [tuple(l) for l in encoded]
        return encoded


    def encode_action(self, action):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        keys = list(prim_choices.keys())
        astr = str(action).lower()
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower().find(task) >= 0:
                l[0] = i
                break

        for enum in prim_choices:
            if enum is TASK_ENUM or np.isscalar(prim_choices[enum]): continue
            l.append(0)
            if hasattr(prim_choices[enum], '__len__'):
                for i, opt in enumerate(prim_choices[enum]):
                    if opt in [p.name for p in action.params]:
                        l[-1] = i
                        break
            else:
                raise Exception('THIS SHOUDLN"T HAPPEN?', enum, prim_choices[enum])
                if self.task_list[l[0]].find('move') >= 0:
                    l[-1] = tuple(action.params[1].pose[:,action.active_timesteps[0]])
                elif self.task_list[l[0]].find('place') >= 0 or self.task_list[l[0]].find('transfer') >= 0:
                    l[-1] = tuple(action.params[4].value[:,0])
                else:
                    l = l[:-1]

        if self.task_list[l[0]].find('move') >= 0 and TARG_ENUM in prim_choices:
            l[keys.index(TARG_ENUM)] = np.random.randint(len(prim_choices[TARG_ENUM]))
        if self.task_list[l[0]].find('door') >= 0 and OBJ_ENUM in prim_choices:
            l[keys.index(OBJ_ENUM)] = np.random.randint(len(prim_choices[OBJ_ENUM]))
        if self.task_list[l[0]].find('door') >= 0 and TARG_ENUM in prim_choices:
            l[keys.index(TARG_ENUM)] = np.random.randint(len(prim_choices[TARG_ENUM]))
        return l # tuple(l)



