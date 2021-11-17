import copy
import ctypes
import pickle as pickle
import sys
from threading import Thread
import time
from tkinter import TclError
import traceback
import xml.etree.ElementTree as xml

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation

import pybullet as P
import robodesk

import core.util_classes.common_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
import core.util_classes.transform_utils as T
from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.tamp_agent import TAMPAgent


const.NEAR_GRIP_COEFF = 1e-1
const.GRASP_DIST = 0.15
const.APPROACH_DIST = 0.015
const.EEREACHABLE_ROT_COEFF = 8e-3

SHELF_ROT = [1.57, 1.57, 0]
SHELF_OFFSET = [-0.3, -0.07, 1.005]
DRAWER_OFFSET = [0., -0.36, 0.01]

STEP = 0.1
NEAR_TOL = 0.05
LOCAL_NEAR_TOL = 0.12 # 0.3

def theta_error(cur_quat, next_quat):
    sign1 = np.sign(cur_quat[np.argmax(np.abs(cur_quat))])
    sign2 = np.sign(next_quat[np.argmax(np.abs(next_quat))])
    next_quat = np.array(next_quat)
    cur_quat = np.array(cur_quat)
    angle = -(sign1 * sign2) * robo_T.get_orientation_error(sign1 * next_quat, sign2 * cur_quat)
    return angle

class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise=None):
        u = np.zeros(self.dU)
        if t < len(self.opt_traj) - 1:
            for param, attr in self.action_inds:
                cur_val = X[self.state_inds[param, attr]] if (param, attr) in self.state_inds else None
                if attr.find('grip') >= 0:
                    val = self.opt_traj[t+1, self.state_inds[param, attr]][0]
                    val = 0.1 if val <= 0. else -0.1
                    u[self.action_inds[param, attr]] = val 
                elif attr.find('ee_pos') >= 0:
                    cur_ee = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds[param, attr]]
                    next_ee = self.opt_traj[t+1, self.state_inds[param, attr]]
                    u[self.action_inds[param, attr]] = next_ee - cur_ee
                elif attr.find('ee_rot') >= 0:
                    cur_ee = cur_val
                    cur_quat = np.array(T.euler_to_quaternion(cur_ee, 'xyzw'))
                    next_ee = self.opt_traj[t+1, self.state_inds[param, attr]]
                    next_quat = np.array(T.euler_to_quaternion(next_ee, 'xyzw'))
                    currot = Rotation.from_quat(cur_quat)
                    targrot = Rotation.from_quat(next_quat)
                    act = (targrot * currot.inv()).as_rotvec()
                    u[self.action_inds[param, attr]] = act
                else:
                    cur_attr = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds[param, attr]]
                    next_attr = self.opt_traj[t+1, self.state_inds[param, attr]]
                    u[self.action_inds[param, attr]] = next_attr - cur_attr
        else:
            for param, attr in self.action_inds:
                if attr.find('grip') >= 0:
                    val = self.opt_traj[-1, self.state_inds[param, attr]][0]
                    val = 0.1 if val <= 0. else -0.1
                    u[self.action_inds[param, attr]] = val 
        if np.any(np.isnan(u)):
            u[np.isnan(u)] = 0.
        return u


class EnvWrapper():
    def __init__(self, env, robot, mode='ee_pos'):
        self.env = env
        self.physics = env.physics
        self.robot = robot
        self.geom = robot.geom
        self._type_cache = {}
        self.physics = env.physics
        self.model = self.physics.model
        self.mode = mode
        self.upright_rot = Rotation.from_euler('xyz', [1.57, 0., 0.])
        self.upright_rot_inv = self.upright_rot.inv()


    def get_task():
        return self.env.task


    def set_task(self, taks):
        self.env.task = task
    

    def get_attr(self, obj, attr, euler=False):
        if attr.find('ee_pos') >= 0:
            obj = 'end_effector'
            ind = self.env.physics.model.name2id(obj, 'site')
            return self.env.physics.data.site_xpos[ind]

        if attr.find('ee_rot') >= 0:
            obj = 'end_effector'
            ind = self.env.physics.model.name2id(obj, 'site')
            mat = self.env.physics.data.site_xmat[ind].reshape((3,3))
            rot = T.mat2quat(mat)
            if euler:
                rot = T.quaternion_to_euler(rot)
            return rot

        if attr in self.geom.jnt_names:
            jnts = self.geom.jnt_names[attr]
            if attr in self.geom.arms:
                vals = self.get_joints(jnts)
                return vals
            else:
                #cv, ov = self.geom.get_gripper_closed_val(), self.geom.get_gripper_open_val()
                vals = self.get_joints(jnts)
                return vals

        if attr.find('hinge') >= 0:
            if obj.find('drawer') >= 0:
                jnts = ['drawer_joint']
            elif obj.find('shelf') >= 0:
                jnts = ['slide_joint']
            return self.get_joints(jnts)

        if obj == 'panda':
            obj = 'panda0_link0'

        if attr == 'pose' or attr == 'pos':
            return self.get_item_pose(obj)[0]

        if attr.find('rot') >= 0 or attr.find('quat') >= 0:
            return self.get_item_pose(obj, euler=euler)[1]

    def set_attr(self, obj, attr, val, euler=False, forward=False):
        if attr in self.geom.jnt_names:
            jnts = self.geom.jnt_names[attr]
            return self.set_joints(jnts, val, forward=forward)

        if attr.find('hinge') >= 0:
            if obj.find('drawer') >= 0:
                jnts = ['drawer_joint']
            elif obj.find('shelf') >= 0:
                jnts = ['slide_joint']
            return self.set_joints(jnts, val, forward=forward)

        if attr.find('ee_pos') >= 0 or attr.find('ee_rot') >= 0:
            return

        if attr == 'pose' or attr == 'pos':
            return self.set_item_pose(obj, val, forward=forward)

        if attr.find('rot') >= 0 or attr.find('quat') >= 0:
            return self.set_item_pose(obj, quat=val, euler=euler, forward=forward)


    def get_handle_pose(self, handle_name, order='xyzw', euler=False):
        if item_name.find('shelf') >= 0:
            quat = T.euler_to_quaternion(SHELF_HANDLE_ROT, 'xyzw')
        else:
            quat = [0., 0., 0., 1.]

        door = item_name.split('_handle')[0]
        if door.find('shelf') >= 0:
            jnt = 'slide_joint'
            val = self.env.physics.named.data.qpos[jnt]
            door_pos = self.env.physics.named.data.xpos[door]
            pos = door_pos + SHELF_OFFSET + np.array([val, 0., 0.])

        if door.find('drawer') >= 0:
            jnt = 'drawer_joint'
            val = self.env.physics.named.data.qpos[jnt]
            door_pos = self.env.physics.named.data.xpos[door]
            pos = door_pos + DRAWER_OFFSET + np.array([0, val, 0.])

        rot = quat
        if euler:
            rot = T.quaternion_to_euler(quat, 'xyzw')
        return pos, rot
   

    def get_item_pose(self, item_name, order='xyzw', euler=False):
        pos, quat = None, None
        if item_name.find('handle') >= 0:
            return self.get_handle_pose(item_name, order, euler)

        if item_name.find('drawer') >= 0: item_name = 'drawer'
        if item_name.find('shelf') >= 0: item_name = 'slide'

        pos = self.env.physics.named.data.xpos[item_name]
        quat = self.env.physics.named.data.xquat[item_name]
        #try:
        #    pose = self.env.physics.named.data.qpos[item_name]
        #    pos, quat = pose[:3], pose[3:7]
        #except Exception as e:
        #    pos = self.env.physics.named.data.xpos[item_name]
        #    quat = self.env.physics.named.data.xquat[item_name]

        if item_name.find('ball') >= 0:
            quat = [1., 0., 0., 0.]

        if order != 'xyzw':
            raise Exception()

        quat = [quat[1], quat[2], quat[3], quat[0]]
        if item_name.find('upright') >= 0:
            base_rot = Rotation.from_quat(quat)
            quat = (base_rot * self.upright_rot).as_quat()

        rot = quat
        if euler:
            rot = T.quaternion_to_euler(quat, 'xyzw')
        return np.array(pos), np.array(rot)


    def set_item_pose(self, item_name, pos=None, quat=None, forward=False, order='xyzw', euler=False):
        if item_name == 'panda': return
        if item.name.find('desk'): return
        if item.name.find('handle'): return
        if item.name.find('button'): return

        if quat is not None and len(quat) == 3:
            quat = T.euler_to_quaternion(quat, order)

        if item_name.find('upright') >= 0:
            base_rot = Rotation.from_quat(quat)
            quat = (self.upright_rot_inv * base_rot).as_quat()

        if quat is not None and order != 'wxyz':
            quat = [quat[3], quat[0], quat[1], quat[2]]

        try:
            if pos is not None:
                self.env.physics.named.data.qpos[item_name][:3] = pos[item_name]
            if quat is not None:
                self.env.physics.named.data.qpos[item_name][3:] = quat
        except Exception as e:
            if pos is not None:
                self.env.physics.named.data.xpos[item_name][:3] = pos[item_name]
            if quat is not None:
                self.env.physics.named.data.xpos[item_name][3:] = quat

        if forward:
            self.forward()

    def get_joints(self, jnt_names):
        vals = []
        for jnt in jnt_names:
            ind = self.env.physics.model.name2id(jnt, 'joint')
            adr = self.env.physics.model.jnt_qposadr[ind]
            vals.append(self.env.physics.data.qpos[adr])
        return np.array(vals)

    def set_joints(self, jnt_names, jnt_vals, forward=False):
        if len(jnt_vals) != len(jnt_names):
            print(jnt_names, jnt_vals, 'MAKE SURE JNTS MATCH')

        for jnt, val in zip(jnt_names, jnt_vals):
            ind = self.env.physics.model.name2id(jnt, 'joint')
            adr = self.env.physics.model.jnt_qposadr[ind]
            self.env.physics.data.qpos[adr] = val

        if forward:
            self.forward()

    def step(self, action):
        if self.mode.find('joint') >= 0:
            joint_position = action
            for t in range(self.env.action_repeat):
                for _ in range(10):
                    self.env.physics.data.ctrl[0:9] = joint_position[0:9]
                    # Ensure gravity compensation stays enabled.
                    self.env.physics.data.qfrc_applied[0:9] = self.physics.data.qfrc_bias[0:9]
                    self.env.physics.step()
                    self.env.physics_copy.data.qpos[:] = self.physics.data.qpos[:]

                    if self.env.reward == 'dense':
                        total_reward += self.env._get_task_reward(self.env.task, 'dense_reward')
                    elif self.env.reward == 'sparse':
                        total_reward += float(self.env._get_task_reward(self.env.task, 'success'))
                    elif self.env.reward == 'success':
                        if self.env.success:
                            total_reward += 0  # Only give reward once in case episode continues.
                        else:
                            self.env.success = self.env._get_task_reward(self.env.task, 'success')
                            total_reward += float(self.env.success)
                    else:
                        raise ValueError(self.env.reward)

                self.env.num_steps += self.env.action_repeat
                if self.env.episode_length and self.env.num_steps >= self.env.episode_length:
                    done = True
                else:
                    done = False

            return self.env._get_obs(), total_reward, done, {'discount': 1.0}

        return self.env.step(action)

    def zero(self):
        self.env.physics.data.time = 0.0
        self.env.physics.data.qvel[:] = 0
        self.env.physics.data.qacc[:] = 0
        self.env.physics.data.qfrc_bias[:] = 0
        self.env.physics.data.qacc_warmstart[:] = 0
        self.env.physics.data.ctrl[:] = 0
        self.env.physics.data.qfrc_applied[:] = 0
        self.env.physics.data.xfrc_applied[:] = 0

    def forward(self):
        #self.zero()
        self.env.physics.forward()

    def reset(self):
        obs = self.env.reset()
        #cur_pos = self.get_attr('panda', 'right_ee_pos')
        #cur_jnts = self.get_attr('panda', 'right')
        #dim = 9 if self.mode.find('joint') >= 0 else 8
        #for _ in range(40):
        #    self.step(np.zeros(dim))
        #    self.set_attr('panda', 'right', cur_jnts)
        #    self.forward()

        #self.forward()
        return obs


    def close(self):
        self.env.close()


class RobotAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super(RobotAgent, self).__init__(hyperparams)

        self.optimal_pol_cls =  optimal_pol
        self.load_render = hyperparams['master_config'].get('load_render', False)
        self.ctrl_mode = 'joint' if ('panda', 'right') in self.action_inds else 'ee_pos'

        freq = 25
        self.base_env = robodesk.RoboDesk(task='lift_ball', \
                                          reward='success', \
                                          action_repeat=freq, \
                                          episode_length=None, \
                                          image_size=64)

        prim_options = self.prob.get_prim_choices(self.task_list)
        self.obj_list = prim_options[OBJ_ENUM]
        self.panda = list(self.plans.values())[0].params['panda']
        self.mjc_env = EnvWrapper(self.base_env, self.panda, self.ctrl_mode)
        self.check_col = hyperparams['master_config'].get('check_col', True)
        self.use_glew = False
        self._viewer = None
        self.camera_id = 1
        self.main_camera_id = 0
        no = self._hyperparams['num_objs']
        self.targ_labels = {}
        self.cur_obs = self.mjc_env.reset()
        self.replace_cond(0)

    def get_annotated_image(self, s, t, cam_id=None):
        x = s.get_X(t=t)
        self.reset_to_state(x, full=False)
        task = [int(val) for val in s.get(FACTOREDTASK_ENUM, t=t)]
        pos = s.get(END_POSE_ENUM, t=t)
        precost = round(self.precond_cost(s, tuple(task), t), 5)
        postcost = round(self.postcond_cost(s, tuple(task), t), 5)

        precost = str(precost)[1:]
        postcost = str(postcost)[1:]

        gripcmd = round(s.get_U(t=t)[self.action_inds['sawyer', 'right_gripper']][0], 2)

        for ctxt in self.base_env.sim.render_contexts:
            ctxt._overlay[mj_const.GRID_TOPLEFT] = ['{}'.format(task), '']
            ctxt._overlay[mj_const.GRID_BOTTOMLEFT] = ['{0: <7} {1: <7} {2}'.format(precost, postcost, gripcmd), '']
        #return self.base_env.sim.render(height=self.image_height, width=self.image_width, camera_name="frontview")
        im = self.base_env.sim.render(height=192, width=192, camera_name="frontview")
        im = np.flip(im, axis=0)
        for ctxt in self.base_env.sim.render_contexts:
            for key in list(ctxt._overlay.keys()):
                del ctxt._overlay[key]
        return im

    def get_image(self, x, depth=False, cam_id=None):
        self.reset_to_state(x, full=False)
        #return self.base_env.sim.render(height=self.image_height, width=self.image_width, camera_name="frontview")
        im = self.base_env.sim.render(height=192, width=192, camera_name="frontview")
        im = np.flip(im, axis=0)
        return im

    #def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None, policies=None):
    #    assert not np.any(np.isnan(state))
    #    start_t = time.time()
    #    x0 = self.get_state() # state[self._x_data_idx[STATE_ENUM]].copy()
    #    task = tuple(task)
    #    if self.discrete_prim:
    #        plan = self.plans[task]
    #    else:
    #        plan = self.plans[task[0]]

    #    base_t = 0
    #    self.T = plan.horizon if hor is None else hor
    #    sample = Sample(self)
    #    sample.init_t = 0
    #    col_ts = np.zeros(self.T)

    #    prim_choices = self.prob.get_prim_choices(self.task_list)
    #    target_vec = np.zeros((self.target_dim,))

    #    set_params_attrs(plan.params, plan.state_inds, x0, 0)
    #    for target_name in self.targets[condition]:
    #        target = plan.params[target_name]
    #        target.value[:,0] = self.targets[condition][target.name]
    #        target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

    #    cur_state = self.get_state()
    #    noise = np.zeros((self.T, self.dU))
    #    n_steps = 0
    #    end_state = None
    #    for t in range(0, self.T):
    #        noise_full = np.zeros((self.dU,))
    #        self.fill_sample(condition, sample, cur_state, t, task, fill_obs=True)
    #        sample.env_state[t] = self.base_env.sim.get_state()
    #        prev_task = task
    #        if task_f is not None:
    #            sample.task = task
    #            task = task_f(sample, t, task)
    #            if task not in self.plans:
    #                task = self.task_to_onehot[task[0]]
    #            self.fill_sample(condition, sample, cur_state, t, task, fill_obs=False)
    #            taskname = self.task_list[task[0]]
    #            if policies is not None: policy = policies[taskname]

    #        X = cur_state.copy()
    #        cur_noise = noise[t]

    #        U_full = policy.act(X, sample.get_obs(t=t).copy(), t, cur_noise)
    #        U_nogrip = U_full.copy()
    #        U_nogrip[self.action_inds['sawyer', 'right_gripper']] = 0.
    #        if np.all(np.abs(U_nogrip)) < 1e-3:
    #            self._noops += 1
    #            self.eta_scale = 1. / np.log(self._noops+2)
    #        else:
    #            self._noops = 0
    #            self.eta_scale = 1.
    #        assert not np.any(np.isnan(U_full))
    #        sample.set(NOISE_ENUM, noise_full, t)

    #        obs = sample.get_obs(t=t)
    #        #U_full = np.clip(U_full, -MAX_STEP, MAX_STEP)
    #        assert not np.any(np.isnan(U_full))
    #        sample.set(ACTION_ENUM, U_full, t)
    #        obj = self.prob.get_prim_choices(self.task_list)[OBJ_ENUM][task[1]]
    #        suc, col = self.run_policy_step(U_full, cur_state)
    #        col_ts[t] = col
    #        new_state = self.get_state()
    #        if len(self._prev_U): self._prev_U = np.r_[self._prev_U[1:], [U_nogrip]]
    #        if len(self._x_delta)-1: self._x_delta = np.r_[self._x_delta[1:], [new_state]]
    #        if len(self._prev_task): self._prev_task = np.r_[self._prev_task[1:], [sample.get_prim_out(t=t)]]


    #        #if np.all(np.abs(cur_state - new_state) < 1e-4):
    #        #    sample.use_ts[t] = 0

    #        if n_steps == sample.T:
    #            end_state = sample.get_X(t=t)

    #        cur_state = new_state

    #    if policy not in self.n_policy_calls:
    #        self.n_policy_calls[policy] = 1
    #    else:
    #        self.n_policy_calls[policy] += 1
    #    sample.end_state = new_state # end_state if end_state is not None else sample.get_X(t=self.T-1)
    #    sample.task_cost = self.goal_f(condition, sample.end_state)
    #    sample.use_ts[-1] = 0
    #    sample.prim_use_ts[:] = sample.use_ts[:]
    #    for t in range(sample.T-1):
    #        if np.mean(np.abs(sample.get_X(t=t) - sample.get_X(t=t+1))) < 1e-3:
    #            sample.prim_use_ts[t] = 0.
    #    sample.col_ts = col_ts
    #    return sample


    def run_policy_step(self, u, x):
        self._col = []
        ctrl = {attr: u[inds] for (param_name, attr), inds in self.action_inds.items()}
        cur_grip = x[self.state_inds['panda', 'right_gripper']][0]

        panda = list(self.plans.values())[0].params['panda']
        true_lb, true_ub = panda.geom.get_joint_limits('right')
        factor = (np.array(true_ub) - np.array(true_lb)) / 5
        if 'right_ee_pos' in ctrl:
            targ_pos = self.mjc_env.get_attr('panda', 'right_ee_pos') + ctrl['right_ee_pos']
            rotoff = Rotation.from_rotvec(ctrl['right_ee_rot'])
            curquat = self.mjc_env.get_attr('panda', 'right_ee_rot', euler=False)
            targrot = (rotoff * Rotation.from_quat(curquat)).as_quat()
            for n in range(n_steps+1):
                curquat = self.mjc_env.get_attr('panda', 'right_ee_rot', euler=False)
                pos_ctrl = targ_pos - self.mjc_env.get_attr('panda', 'right_ee_pos')
                sign1 = np.sign(targrot[np.argmax(np.abs(targrot))])
                sign2 = np.sign(curquat[np.argmax(np.abs(curquat))])
                rot_ctrl = -sign1 * sign2 * robo_T.get_orientation_error(sign1*targrot, sign2*curquat)
                self.cur_obs, rew, done, _ = self.base_env.step(np.r_[pos_ctrl, rot_ctrl, [gripper]])

        if 'right' in ctrl:
            targ_pos = self.mjc_env.get_attr('panda', 'right') + ctrl['right']
            ctrl = np.r_[targ_pos, gripper]
            self.cur_obs, rew, done, _ = self.base_env.step(ctrl)

        rew = self.base_env.reward()
        self._rew = rew
        self._ret += rew
        col = 0 # 1 if len(self._col) > 0 else 0
        return True, col


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None, st=0):
        act_st, et = plan.actions[anum].active_timesteps
        st = max(act_st, st)
        if targets is None:
            targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]].find('grasp') >= 0:
            params[2].value[:,0] = params[1].pose[:,st]
            params[2].rotation[:,0] = params[1].rotation[:,st]
        #params[3].value[:,0] = params[0].pose[:,st]
        #for arm in params[0].geom.arms:
        #    getattr(params[3], arm)[:,0] = getattr(params[0], arm)[:,st]
        #    gripper = params[0].geom.get_gripper(arm)
        #    getattr(params[3], gripper)[:,0] = getattr(params[0], gripper)[:,st]
        #    ee_attr = '{}_ee_pos'.format(arm)
        #    rot_ee_attr = '{}_ee_rot'.format(arm)
        #    if hasattr(params[0], ee_attr):
        #        getattr(params[3], ee_attr)[:,0] = getattr(params[0], ee_attr)[:,st]
        #    if hasattr(params[0], rot_ee_attr):
        #        getattr(params[3], rot_ee_attr)[:,0] = getattr(params[0], rot_ee_attr)[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,st]
                if hasattr(plan.params[pname], 'rotation'):
                    plan.params['{0}_init_target'.format(pname)].rotation[:,0] = plan.params[pname].rotation[:,st]


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0, targets=[], x_only=False, t_limit=60, n_resamples=10, out_coeff=None, smoothing=False, attr_dict=None):
        success = False
        old_targets = self.target_vecs[condition].copy()
        if not len(targets):
            targets = self.target_vecs[condition].copy()
        else:
            self.target_vecs[condition] = targets.copy()
            for tname, attr in self.target_inds:
                if attr == 'value':
                    self.targets[condition][tname] = targets[self.target_inds[tname, attr]]

        x0 = state[self._x_data_idx[STATE_ENUM]]

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task]
        prim_choices = self.prob.get_prim_choices(self.task_list)
        set_params_attrs(plan.params, plan.state_inds, x0, 0)

        for param_name in plan.params:
            param = plan.params[param_name]
            if '{0}_init_target'.format(param_name) in plan.params:
                param.pose[:, 0] = x0[self.state_inds[param_name, 'pose']]
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = param.pose[:,0]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for param in plan.params.values():
            if (param.name, 'pose') in self.state_inds:
                param.pose[:, 0] = x0[self.state_inds[param.name, 'pose']]
            if 'Robot' in param.get_type(True):
                for arm in param.geom.arms:
                    gripper = param.geom.get_gripper(arm)
                    ee_attr = '{}_ee_pos'.format(arm)
                    if (param.name, arm) in self.state_inds:
                        getattr(param, arm)[:,0] = x0[self.state_inds[param.name, arm]]
                    if (param.name, gripper) in self.state_inds:
                        getattr(param, gripper)[:,0] = x0[self.state_inds[param.name, gripper]]
                    if (param.name, ee_attr) in self.state_inds:
                        getattr(param, ee_attr)[:,0] = x0[self.state_inds[param.name, ee_attr]]

        run_solve = True
        for param in list(plan.params.values()):
            for attr in param._free_attrs:
                if np.any(np.isnan(getattr(param, attr)[:,0])):
                    getattr(param, attr)[:,0] = 0

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

        #self.solver.strong_transfer_coeff = old_out_coeff

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

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, traj), condition, state, task, noisy=False, skip_opt=True)

        traj = sample.get(STATE_ENUM)
        for param_name, attr in self.state_inds:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            diff = traj[:, self.state_inds[param_name, attr]].T - getattr(param, attr)
        self.target_vecs[condition] = old_targets 
        return sample, failed_preds, success


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        mp_state = mp_state.copy()
        if targets is None:
            targets = self.target_vecs[cond].copy()

        for (pname, aname), inds in self.state_inds.items():
            if aname == 'left_ee_pos':
                sample.set(LEFT_EE_POS_ENUM, mp_state[inds], t)
                ee_pose = mp_state[inds]
                ee_rot = mp_state[self.state_inds[pname, 'left_ee_rot']]
            elif aname == 'right_ee_pos':
                sample.set(RIGHT_EE_POS_ENUM, mp_state[inds], t)
                ee_pose = mp_state[inds]
                ee_rot = mp_state[self.state_inds[pname, 'right_ee_rot']]
            elif aname.find('ee_pos') >= 0:
                sample.set(EE_ENUM, mp_state[inds], t)
                ee_pose = mp_state[inds]
                ee_rot = mp_state[self.state_inds[pname, 'ee_rot']]

        ee_spat = Rotation.from_euler('xyz', ee_rot)
        ee_quat = T.euler_to_quaternion(ee_rot, 'xyzw')
        ee_mat = T.quat2mat(ee_quat)
        sample.set(STATE_ENUM, mp_state, t)
        if self.hist_len > 0:
            sample.set(TRAJ_HIST_ENUM, self._prev_U.flatten(), t)
            x_delta = self._x_delta[1:] - self._x_delta[:1]
            sample.set(STATE_DELTA_ENUM, x_delta.flatten(), t)
        sample.set(STATE_HIST_ENUM, self._x_delta.flatten(), t)
        if self.task_hist_len > 0:
            sample.set(TASK_HIST_ENUM, self._prev_task.flatten(), t)
        sample.set(DONE_ENUM, np.zeros(1), t)
        sample.set(TASK_DONE_ENUM, np.array([1, 0]), t)
        robot = 'panda'
        if RIGHT_ENUM in self.sensor_dims:
            sample.set(RIGHT_ENUM, mp_state[self.state_inds[robot, 'right']], t)
        if LEFT_ENUM in self.sensor_dims:
            sample.set(LEFT_ENUM, mp_state[self.state_inds[robot, 'left']], t)
        if LEFT_GRIPPER_ENUM in self.sensor_dims:
            sample.set(LEFT_GRIPPER_ENUM, mp_state[self.state_inds[robot, 'left_gripper']], t)
        if RIGHT_GRIPPER_ENUM in self.sensor_dims:
            sample.set(RIGHT_GRIPPER_ENUM, mp_state[self.state_inds[robot, 'right_gripper']], t)

        prim_choices = self.prob.get_prim_choices(self.task_list)
        if task is not None:
            task_ind = task[0]
            obj_ind = task[1]
            targ_ind = task[2]
            door_ind = task[3]

            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[task[0]] = 1.
            sample.task_ind = task[0]
            sample.set(TASK_ENUM, task_vec, t)
            for ind, enum in enumerate(prim_choices):
                if hasattr(prim_choices[enum], '__len__'):
                    vec = np.zeros((len(prim_choices[enum])), dtype='float32')
                    vec[task[ind]] = 1.
                else:
                    vec = np.array(task[ind])
                sample.set(enum, vec, t)

            sample.set(FACTOREDTASK_ENUM, np.array(task), t)
            obj_name = list(prim_choices[OBJ_ENUM])[task[1]]
            targ_name = list(prim_choices[TARG_ENUM])[task[2]]
            for (pname, aname), inds in self.state_inds.items():
                if aname.find('right_ee_pos') >= 0:
                    obj_pose = mp_state[self.state_inds[obj_name, 'pose']] - mp_state[inds]
                    targ_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[inds]
                    break
            targ_off_pose = targets[self.target_inds[targ_name, 'value']] - mp_state[self.state_inds[obj_name, 'pose']]
            obj_quat = T.euler_to_quaternion(mp_state[self.state_inds[obj_name, 'rotation']], 'xyzw')
            targ_quat = T.euler_to_quaternion(targets[self.target_inds[targ_name, 'rotation']], 'xyzw')
            sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
            sample.set(TARG_POSE_ENUM, targ_pose.copy(), t)
            sample.task = task
            sample.obj = task[1]
            sample.targ = task[2]
            sample.task_name = self.task_list[task[0]]

            grasp_pt = list(self.plans.values())[0].params[obj_name].geom.grasp_point
            task_name = self.task_list[task[0]].lower()

            if task_name.find('door') < 0:
                door_vec = np.zeros(len(prim_choices[DOOR_ENUM]))
                door_vec[:] = 1. / len(door_vec)
                sample.set(DOOR_ENUM, door_vec, t)
            else:
                door_vec = np.zeros(len(prim_choices[DOOR_ENUM]))
                door_vec[task[2]] = 1.
                sample.set(DOOR_ENUM, door_vec, t)

            if task_name.lower() in ['move_to_grasp', 'lift', 'hold'] or \
                task_name.find('slide') >= 0:
                obj_mat = T.quat2mat(obj_quat)
                goal_quat = T.mat2quat(obj_mat.dot(ee_mat))
                rot_off = theta_error(ee_quat, goal_quat)
                sample.set(END_POSE_ENUM, obj_pose+grasp_pt, t)
                sample.set(END_ROT_ENUM, mp_state[self.state_inds[obj_name, 'rotation']], t)
                targ_vec = np.zeros(len(prim_choices[TARG_ENUM]))
                targ_vec[:] = 1. / len(targ_vec)
                sample.set(TARG_ENUM, targ_vec, t)

            elif self.task_list[task[0]].find('place') >= 0:
                targ_mat = T.quat2mat(targ_quat)
                goal_quat = T.mat2quat(targ_mat.dot(ee_mat))
                rot_off = theta_error(ee_quat, targ_quat)
                sample.set(END_POSE_ENUM, targ_off_pose, t)
                sample.set(END_ROT_ENUM, targets[self.target_inds[targ_name, 'rotation']], t)

            else:
                raise NotImplementedError()

        sample.condition = cond
        sample.set(TARGETS_ENUM, targets.copy(), t)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
            sample.set(ONEHOT_GOAL_ENUM, targets, t)
        sample.targets = targets.copy()

        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            grasp_pt = list(self.plans.values())[0].params[obj].geom.grasp_point
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose+grasp_pt, t)
            sample.set(TARG_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

            obj_spat = Rotation.from_euler('xyz', mp_state[self.state_inds[obj, 'rotation']])
            obj_quat = T.euler_to_quaternion(mp_state[self.state_inds[obj, 'rotation']], 'xyzw')
            obj_mat = T.quat2mat(obj_quat)
            goal_quat = T.mat2quat(obj_mat.dot(ee_mat))
            rot_off = theta_error(ee_quat, goal_quat)
            #sample.set(OBJ_ROTDELTA_ENUMS[i], rot_off, t)
            sample.set(OBJ_ROTDELTA_ENUMS[i], (obj_spat.inv() * ee_spat).as_rotvec(), t)
            targ_rot_off = theta_error(ee_quat, [0, 0, 0, 1])
            targ_spat = Rotation.from_euler('xyz', [0., 0., 0.])
            #sample.set(TARG_ROTDELTA_ENUMS[i], targ_rot_off, t)
            sample.set(TARG_ROTDELTA_ENUMS[i], (targ_spat.inv() * ee_spat).as_rotvec(), t)

        if fill_obs:
            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include']:
                im = self.base_env.render(height=self.image_height, width=self.image_width, view=self.view)
                im = (im - 128.) / 128.
                sample.set(IM_ENUM, im.flatten(), t)


    def goal_f(self, condition, state, targets=None, cont=False, anywhere=False, tol=LOCAL_NEAR_TOL, verbose=False):
        if targets is None:
            targets = self.target_vecs[condition]
        objs = self.prob.get_prim_choices(self.task_list)[OBJ_ENUM]
        cost = len(objs)
        alldisp = 0
        plan = list(self.plans.values())[0]
        no = self._hyperparams['num_objs']
        if len(np.shape(state)) < 2:
            state = [state]

        for param_name in objs:
            param = plan.params[param_name]
            if 'Item' in param.get_type(True) and ('{0}_end_target'.format(param.name), 'value') in self.target_inds:
                if anywhere:
                    vals = [targets[self.target_inds[key, 'value']] for key, _ in self.target_inds if key.find('end_target') >= 0]
                else:
                    vals = [targets[self.target_inds['{0}_end_target'.format(param.name), 'value']]]
                dist = np.inf
                disp = None
                for x in state:
                    for val in vals:
                        curdisp = x[self.state_inds[param.name, 'pose']] - val
                        curdist = np.linalg.norm(curdisp)
                        if curdist < dist:
                            disp = curdisp
                            dist = curdist
                # np.sum((state[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                # cost -= 1 if dist < 0.3 else 0
                alldisp += dist # np.linalg.norm(disp)
                cost -= 1 if np.all(np.abs(disp) < tol) else 0

        if cont: return alldisp / float(no)
        # return cost / float(self.prob.NUM_OBJS)
        return 1. if cost > 0 else 0.


    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x, full=True):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._ret = 0.
        self._rew = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self.eta_scale = 1.
        self._noops = 0
        self._x_delta[:] = x.reshape((1,-1))
        self._prev_task = np.zeros((self.task_hist_len, self.dPrimOut))
        self.cur_state = x.copy()
        if full: self.base_env.reset()
        self.base_env.physics.reset()
        for (pname, aname), inds in self.state_inds.items():
            if aname.find('ee_pos') >= 0 or aname.find('ee_rot') >= 0: continue
            val = x[inds]
            self.mjc_env.set_attr(pname, aname, val, forward=False)
        self.base_env.physics.forward()


    def get_state(self, clip=False):
        x = np.zeros(self.dX)
        for (pname, aname), inds in self.state_inds.items():
            val = self.mjc_env.get_attr(pname, aname, euler=True)

            if clip:
                if aname in ['left', 'right']:
                    lb, ub = self.mjc_env.geom.get_joint_limits(aname)
                    val = np.maximum(np.minimum(val, ub), lb)
                elif aname.find('gripper') >= 0:
                    cv, ov = self.panda.geom.get_gripper_closed_val(), self.panda.geom.get_gripper_open_val()
                    val = ov if np.max(np.abs(val-cv)) > np.max(np.abs(val-ov)) else cv

            if len(inds) != len(val):
                raise Exception('Bad state retrieval for', pname, aname, 'expected', len(inds), 'but got', len(val))

            x[inds] = val

        return x


    def reset_mjc_env(self, x, targets=None, draw_targets=True):
        pass


    def set_to_targets(self, condition=0):
        return


    def check_targets(self, x, condition=0):
        mp_state = x[self._x_data_idx]
        prim_choices = self.prob.get_prim_choices(self.task_list)
        objs = prim_choices[OBJ_ENUM]
        correct = 0
        for obj_name in objs:
            target = self.targets[condition]['{0}_end_target'.format(obj_name)]
            obj_pos = mp_state[self.state_inds[obj_name, 'pose']]
            if np.linalg.norm(obj_pos - target) < 0.05:
                correct += 1
        return correct


    def get_mjc_obs(self, x):
        return self.mjc_env.render()


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], targets=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean, targets=targets)
        if not len(targets):
            old_targets = self.target_vecs[condition].copy()
        else:
            old_targets = self.target_vecs[condition].copy()
            for tname, attr in self.target_inds:
                if attr == 'value':
                    self.targets[condition][tname] = targets[self.target_inds[tname, attr]]
            self.target_vecs[condition] = targets

        exclude_targets = []
        plan = self.plans[task]
        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_traj), condition, state, task, noisy=False, skip_opt=True, hor=len(opt_traj))
        sample.set_ref_X(sample.get_X())
        sample.set_ref_U(sample.get_U())

        self.target_vecs[condition] = old_targets
        for tname, attr in self.target_inds:
            if attr == 'value':
                self.targets[condition][tname] = old_targets[self.target_inds[tname, attr]]
        return sample


    def get_random_initial_state_vec(self, config, plans, dX, state_inds, ncond):
        self.cur_obs = self.mjc_env.reset()
        #for ind, obj in enumerate(self.obj_list):
        #    if ind >= config['num_objs'] and (obj, 'pose') in self.state_inds:
        #        self.set_to_target(obj)

        x = np.zeros(self.dX)
        for pname, aname in self.state_inds:
            inds = self.state_inds[pname, aname]
            val = self.mjc_env.get_attr(pname, aname, euler=True)
            if len(inds) == 1: val = np.mean(val)
            x[inds] = val

        targets = {}
        plan = list(self.plans.values())[0]
        for targ in ['shelf_target', 'off_desk_target', 'bin_target']:
            targets[targ] = plan.params[targ].value[:,0].copy()
        return [x], [targets] 
   

    def set_to_target(self, obj, targets=None):
        pass

    
    def replace_cond(self, cond, curric_step=-1):
        self.cur_obs = self.mjc_env.reset()
        x, targets = self.get_random_initial_state_vec(self.config, self.plans, self.dX, self.state_inds, 1)
        x, targets = x[0], targets[0]
        self.init_vecs[cond] = x
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        self.targets[cond] = targets

        prim_choices = self.prob.get_prim_choices(self.task_list)
        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]


    def goal(self, cond, targets=None):
        if targets is None:
            targets = self.target_vecs[cond]
        prim_choices = self.prob.get_prim_choices(self.task_list)
        goal = ''
        if self.goal_type == 'moveto':
            return '(NearApproachRight sawyer cereal)'
        
        if self.goal_type == 'grasp':
            return '(NearGripperRight sawyer cereal)'

        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            goal += '(Near {0} {0}_end_target) '.format(obj)
        return goal


    def check_target(self, targ):
        return
        vec = np.zeros(len(list(self.targ_labels.keys())))
        for ind in self.targ_labels:
            if np.all(np.abs(targ - self.targ_labels[ind]) < NEAR_TOL):
                vec[ind] = 1.
                break
        return vec


    def onehot_encode_goal(self, targets, descr=None, debug=False):
        vecs = []
        for i in range(0, len(targets), 3):
            targ = targets[i:i+3]
            vec = self.check_target(targ)
            vecs.append(vec)
        return np.concatenate(vecs)


    def get_mask(self, sample, enum):
        mask = np.ones((sample.T, 1))
        return mask


    def permute_hl_data(self, hl_mu, hl_obs, hl_wt, hl_prc, aux):
        return hl_mu, hl_obs, hl_wt, hl_prc


    def permute_tasks(self, tasks, targets, plan=None, x=None):
        encoded = [list(l) for l in tasks]
        no = self._hyperparams['num_objs']
        perm = np.random.permutation(range(no))
        for l in encoded:
            l[1] = perm[l[1]]
        prim_opts = self.prob.get_prim_choices(self.task_list)
        objs = prim_opts[OBJ_ENUM]
        encoded = [tuple(l) for l in encoded]
        target_vec = targets.copy()
        param_map = {}
        old_values = {}
        perm_map = {}
        for n in range(no):
            obj1 = objs[n]
            obj2 = objs[perm[n]]
            inds = self.target_inds['{0}_end_target'.format(obj1), 'value']
            inds2 = self.target_inds['{0}_end_target'.format(obj2), 'value']
            target_vec[inds2] = targets[inds]
            if plan is None:
                old_values[obj1] = x[self.state_inds[obj1, 'pose']]
            else:
                old_values[obj1] = plan.params[obj1].pose.copy()
            perm_map[obj1] = obj2
        return encoded, target_vec, perm_map


    def encode_plan(self, plan, permute=False):
        encoded = []
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for a in plan.actions:
            encoded.append(self.encode_action(a))
        encoded = [tuple(l) for l in encoded]
        return encoded


    def encode_action(self, action):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        astr = str(action).lower()
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower() == task:
                l[0] = i
                break

        for enum in prim_choices:
            if enum is TASK_ENUM: continue
            l.append(0)
            if hasattr(prim_choices[enum], '__len__'):
                for i, opt in enumerate(prim_choices[enum]):
                    if opt in [p.name for p in action.params]:
                        l[-1] = i
                        break
            else:
                param = action.params[1]
                l[-1] = param.value[:,0] if param.is_symbol() else param.pose[:,action.active_timesteps[0]]
        return l # tuple(l)


    def retime_traj(self, traj, vel=0.01, inds=None, minpts=10):
        new_traj = []
        if len(np.shape(traj)) == 2:
            traj = [traj]
        for step in traj:
            xpts = []
            fpts = []
            grippts= []
            d = 0
            rotinds = self.state_inds['panda', 'right_ee_rot']
            rotpts =[]
            if inds is None:
                if ('panda', 'right_ee_pos') in self.action_inds:
                    inds = self.state_inds['panda', 'right_ee_pos']
                elif ('panda', 'right') in self.action_inds:
                    inds = self.state_inds['panda', 'right']

            for t in range(len(step)):
                xpts.append(d)
                fpts.append(step[t])
                grippts.append(step[t][self.state_inds['panda', 'right_gripper']])
                rotpts.append(step[t][self.state_inds['panda', 'right_ee_rot']])
                if t < len(step) - 1:
                    disp = np.linalg.norm(step[t+1][inds] - step[t][inds])
                    d += disp
            assert not np.any(np.isnan(xpts))
            assert not np.any(np.isnan(fpts))
            interp = scipy.interpolate.interp1d(xpts, fpts, axis=0, fill_value='extrapolate')
            grip_interp = scipy.interpolate.interp1d(np.array(xpts), grippts, kind='next', bounds_error=False, axis=0)
            rot_interp = scipy.interpolate.interp1d(np.array(xpts), rotpts, kind='next', bounds_error=False, axis=0)

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
                    elif xpts[i] - xpts[i-1] <= 1e-6:
                        continue
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
            rot_out = rot_interp(x)
            out[:, self.state_inds['panda', 'right_gripper']] = grip_out
            #out[:, self.state_inds['panda', 'right_ee_rot']] = rot_out
            for step in out:
                self.panda.openrave_body.set_pose(step[self.state_inds['panda', 'pose']])
                self.panda.openrave_body.set_dof({'right': step[self.state_inds['panda', 'right']]})
                info = self.panda.openrave_body.fwd_kinematics('right')
                out[:, self.state_inds['panda', 'right_ee_pos']] = info['pos']
                out[:, self.state_inds['panda', 'right_ee_rot']] = T.quaternion_to_euler(info['quat'], 'xyzw')
            for pt, val in fix_pts:
                out[pt] = val
            out[0] = step[0]
            out[-1] = step[-1]
            if len(new_traj):
                new_traj = np.r_[new_traj, out]
            else:
                new_traj = out
            if np.any(np.isnan(out)): print(('NAN in out', out, x))
        return new_traj


    def compare_tasks(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1]

   
    def get_inv_cov(self):
        vec = np.ones(self.dU)
        robot = 'panda'
        if ('panda', 'right') in self.action_inds:
            return np.eye(self.dU)
        elif ('panda', 'right_ee_pos') in self.action_inds and ('panda', 'right_ee_rot') in self.action_inds:
            vecs = np.array([1e1, 1e1, 1e1, 1e-2, 1e-2, 1e-2, 1e0])
        return np.diag(vec)


    def clip_state(self, x):
        x = x.copy()
        lb, ub = self.sawyer.geom.get_joint_limits('right')
        lb = np.array(lb) + 2e-3
        ub = np.array(ub) - 2e-3
        jnt_vals = x[self.state_inds['panda', 'right']]
        x[self.state_inds['panda', 'right']] = np.clip(jnt_vals, lb, ub)
        cv, ov = self.panda.geom.get_gripper_closed_val(), self.panda.geom.get_gripper_open_val()
        grip_vals = x[self.state_inds['panda', 'right_gripper']]
        grip_vals = ov if np.mean(np.abs(grip_vals-cv)) > np.mean(np.abs(grip_vals-ov)) else cv
        x[self.state_inds['panda', 'right_gripper']] = grip_vals
        return x


    def feasible_state(self, x, targets):
        return True

    
    def reward(self, x=None, targets=None, center=False):
        return self.base_env.reward()


    def add_viewer(self):
        if self._viewer is not None: return
        self.cur_im = np.zeros((self.image_height, self.image_width, 3))
        self._launch_viewer(self.image_width, self.image_height)


    def _launch_viewer(self, width, height, title='Main'):
        self._matplot_view_thread = None
        if self.use_glew:
            from dm_control.viewer import viewer
            from dm_control.viewer import views
            from dm_control.viewer import gui
            from dm_control.viewer import renderer
            self._renderer = renderer.NullRenderer()
            self._render_surface = None
            self._viewport = renderer.Viewport(width, height)
            self._window = gui.RenderWindow(width, height, title)
            self._viewer = viewer.Viewer(
                self._viewport, self._window.mouse, self._window.keyboard)
            self._viewer_layout = views.ViewportLayout()
            self._viewer.render()
        else:
            self._viewer = None
            self._matplot_im = None
            self._run_matplot_view()


    def _reload_viewer(self):
        if self._viewer is None or not self.use_glew: return

        if self._render_surface:
          self._render_surface.free()

        if self._renderer:
          self._renderer.release()

        self._render_surface = render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._viewer.zoom_to_scene()


    def render_viewer(self, pixels):
        if self.use_glew and self.viewer is not None:
            with self._window._context.make_current() as ctx:
                ctx.call(
                    self._window._update_gui_on_render_thread, self._window._context.window, pixels)
            self._window._mouse.process_events()
            self._window._keyboard.process_events()
        elif not self.use_glew and self._matplot_im is not None:
            self._matplot_im.set_data(pixels)
            plt.draw()


    def _run_matplot_view(self):
        self._matplot_view_thread = Thread(target=self._launch_matplot_view)
        self._matplot_view_thread.daemon = True
        self._matplot_view_thread.start()


    def _launch_matplot_view(self):
        try:
            self._matplot_im = plt.imshow(self.cur_im)
            plt.show()
        except TclError:
            print('\nCould not find display to launch viewer (this does not affect the ability to render images)\n')


    def distance_to_goal(self, x=None, targets=None):
        raise NotImplementedError()


    def reward(self, x=None, targets=None, center=False):
        if x is None: x = self.get_state()
        if targets is None: targets = self.target_vecs[0]
        raise NotImplementedError()

