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
import pybullet as P


# from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from policy_hooks.sample_list import SampleList

from baxter_gym.envs import MJCEnv

import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer

from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.tamp_eval_funcs import *
# from policy_hooks.namo.sorting_prob_4 import *
from policy_hooks.tamp_agent import TAMPAgent


NEAR_TOL = 0.03
LOCAL_NEAR_TOL = 0.04
MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 100
GRIP_TOL = 0.
MIN_STEP = 1e-2
LIDAR_DIST = 2.
# LIDAR_DIST = 1.5
DSAFE = 5e-1
MAX_STEP = max(1.5*dmove, 1)


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
                cur_val = X[self.state_inds[param, attr]] if (param, attr) in self.state_inds else None
                if attr.find('grip') >= 0:
                    u[self.action_inds[param, attr]] = self.opt_traj[t, self.state_inds[param, attr]]
                elif attr.find('ee_pos') >= 0:
                    cur_ee = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds['pr2', attr]]
                    next_ee = self.opt_traj[t+1, self.state_inds['pr2', attr]]
                    u[self.action_inds[param, attr]] = next_ee - cur_ee
                else:
                    cur_attr = cur_val if cur_val is not None else self.opt_traj[t, self.state_inds['pr2', attr]]
                    next_attr = self.opt_traj[t+1, self.state_inds['pr2', attr]]
                    u[self.action_inds[param, attr]] = next_attr - cur_attr
        else:
            for param, attr in self.action_inds:
                if attr.find('grip') >= 0:
                    u[self.action_inds['pr2', attr]] = self.opt_traj[-1, self.state_inds['pr2', attr]]
        if np.any(np.isnan(u)):
            u[np.isnan(u)] = 0.
        return u


class RobotAgent(TAMPAgent):
    def __init__(self, hyperparams):
        super(RobotAgent, self).__init__(hyperparams)

        self.optimal_pol_cls =  optimal_pol
        prob_env = None if not hasattr(self.prob, 'get_env_cls') else self.prob.get_env_cls()
        self.env_cls = MJCEnv if prob_env is None else prob_env
        self.check_col = hyperparams['master_config'].get('check_col', True)
        items = []
        incl_files = []
        colors = [[0.9, 0, 0, 1], [0, 0.9, 0, 1], [0, 0, 0.9, 1], [0.7, 0.7, 0.1, 1], [1., 0.1, 0.8, 1], [0.5, 0.95, 0.5, 1], [0.75, 0.4, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]
        for param in list(self.plan.values()[0])[0].params.values():
            if 'Robot' in param.get_type(True) and self.env_cls is MJCEnv:
                incl_files.append(param.geom.shape)

            if 'Item' in param.get_type(True):
                if 'Cloth' in param.get_type(True):
                    color = tuple(colors.pop())
                    items.append({'name': param.name, 'type': 'box', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.02, 0.02, 0.02), 'rgba': color})
                    items.append({'name': '{}_end_target'.format(param.name), 'type': 'box', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.04, 0.04, 0.001), 'rgba': color})
                elif 'Obstacle' in param.get_type(True):
                    color = tuple(colors.pop())
                    dims = tuple(param.geom.dim)
                    items.append({'name': param.name, 'type': 'box', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': dim, 'rgba': color})
        config = {
            'obs_include': ['forward_camera'],
            'include_files': incl_files,
            'include_items': items,
            'view': self.view,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height'])
        }

        self.main_camera_id = 0

        items = config['include_items']
        prim_options = self.prob.get_prim_choices(self.task_list)
        config['load_render'] = hyperparams['master_config'].get('load_render', False)
        self.mjc_env = MJCEnv.load_config(config)
        # self.viewer = OpenRAVEViewer(self.env)
        # import ipdb; ipdb.set_trace()
        self.in_gripper = None
        self._in_gripper = None
        no = self._hyperparams['num_objs']
        self.targ_labels = {i: np.array(self.prob.END_TARGETS[i]) for i in range(len(self.prob.END_TARGETS))}
        self.targ_labels.update({i: self.targets[0]['aux_target_{0}'.format(i-no)] for i in range(no, no+self.prob.n_aux)})

    def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None):
        assert not np.any(np.isnan(state))
        start_t = time.time()
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
        self.T = plan.horizon if hor is None else hor
        sample = Sample(self)
        sample.init_t = 0
        col_ts = np.zeros(self.T)

        prim_choices = self.prob.get_prim_choices(self.task_list)
        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for target_name in self.targets[condition]:
            target = plan.params[target_name]
            target.value[:,0] = self.targets[condition][target.name]
            target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

        cur_state = self.get_state()
        noise = np.zeros((self.T, self.dU))
        n_steps = 0
        end_state = None
        for t in range(0, self.T):
            noise_full = np.zeros((self.dU,))
            self.fill_sample(condition, sample, cur_state, t, task, fill_obs=True)
            prev_task = task
            if task_f is not None:
                sample.task = task
                task = task_f(sample, t, task)
                if task not in self.plans:
                    task = self.task_to_onehot[task[0]]
                self.fill_sample(condition, sample, cur_state, t, task, fill_obs=False)

            X = cur_state.copy()
            cur_noise = noise[t]

            U_full = policy.act(sample.get_X(t=t), sample.get_obs(t=t).copy(), t, cur_noise)
            U_nogrip = U_full.copy()
            if np.all(np.abs(U_nogrip)) < 1e-2:
                self._noops += 1
                self.eta_scale = 1. / np.log(self._noops+2)
            else:
                self._noops = 0
                self.eta_scale = 1.
            assert not np.any(np.isnan(U_full))
            sample.set(NOISE_ENUM, noise_full, t)

            obs = sample.get_obs(t=t)
            U_full = np.clip(U_full, -MAX_STEP, MAX_STEP)
            assert not np.any(np.isnan(U_full))
            sample.set(ACTION_ENUM, U_full, t)
            obj = self.prob.get_prim_choices(self.task_list)[OBJ_ENUM][task[1]]
            suc, col = self.run_policy_step(U_full, cur_state, plan, 0, obj, grasp=grasp)
            col_ts[t] = col
            new_state = self.get_state()
            if len(self._prev_U): self._prev_U = np.r_[self._prev_U[1:], [U_nogrip]]
            if len(self._x_delta)-1: self._x_delta = np.r_[self._x_delta[1:], [new_state]]
            if len(self._prev_task): self._prev_task = np.r_[self._prev_task[1:], [sample.get_prim_out(t=t)]]


            if np.all(np.abs(cur_state - new_state) < 1e-3):
                sample.use_ts[t] = 0

            if n_steps == sample.T:
                end_state = sample.get_X(t=t)

            cur_state = new_state

        if policy not in self.n_policy_calls:
            self.n_policy_calls[policy] = 1
        else:
            self.n_policy_calls[policy] += 1
        sample.end_state = new_state # end_state if end_state is not None else sample.get_X(t=self.T-1)
        sample.task_cost = self.goal_f(condition, sample.end_state)
        sample.prim_use_ts[:] = sample.use_ts[:]
        sample.col_ts = col_ts
        return sample


    def run_policy_step(self, u, x):
	self._col = []
        poses = {}
        for pname, aname in self.state_inds:
            if aname.find('pos') < 0: continue
            poses[pname] = self.mjc_env.get_item_pos(pname)
    
        ctrl = {attr: u[inds] for (param_name, attr), inds in self.action_inds.items()}
        n_steps = 5
        for n in range(nsteps+1):
            self.mjc_env.step(ctrl, mode=self.mjc_ctrl_mode, gen_obs=False)

        new_poses = {}
        for pname, aname in self.state_inds:
            if aname.find('pos') < 0: continue
            new_poses[pname] = self.mjc_env.get_item_pos(pname)

        for pname in poses:
            if np.any(np.abs(poses[pname]-new_poses[pname])) > 5e-2:
                self._col.append(pname)
	col = 1 if len(self._col) > 0 else 0
        return True, col


    def set_symbols(self, plan, task, anum=0, cond=0, targets=None):
        st, et = plan.actions[anum].active_timesteps
        if targets is None:
            targets = self.target_vecs[cond].copy()
        prim_choices = self.prob.get_prim_choices(self.task_list)
        act = plan.actions[anum]
        params = act.params
        if self.task_list[task[0]].find('grasp') >= 0:
            params[2].value[:,0] = params[1].pose[,st]
        params[3].pose[:,st] = params[0].pose[:,st]
        for arm in params[0].geom.arms:
            getattr(params[3], arm)[:,0] = getattr(params[0], arm)[:,st]
            gripper = params[0].geom.get_gripper(arm)
            getattr(params[3], gripper)[:,0] = getattr(params[0], gripper)[:,st]
            ee_attr = '{}_ee_pos'.format(arm)
            if hasattr(params[0], ee_attr):
                getattr(params[3], ee_attr)[:,0] = getattr(params[0], ee_attr)[:,st]

        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]

        for pname in plan.params:
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = plan.params[pname].pose[:,0]


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

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, traj), condition, state, task, noisy=False, skip_opt=True)

        traj = sample.get(STATE_ENUM)
        for param_name, attr in self.state_inds:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            diff = traj[:, self.state_inds[param_name, attr]].T - getattr(param, attr)
        return sample, failed_preds, success


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        mp_state = mp_state.copy()
        plan = self.plans[task]
        if targets is None:
            targets = self.target_vecs[cond].copy()

        enum_mapping = self.prob.get_enum_mapping()
        for enum, (pname, aname) in enum_mapping:
            vec = mp_state[self.state_inds[pname, aname]]
            sample.set(enum, vec, t)
        sample.set(STATE_ENUM, mp_state, t)
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
        if self.discrete_prim:
            sample.set(FACTOREDTASK_ENUM, np.array(task), t)
            obj_vec = np.zeros((len(prim_choices[OBJ_ENUM])), dtype='float32')
            targ_vec = np.zeros((len(prim_choices[TARG_ENUM])), dtype='float32')
            obj_vec[task[1]] = 1.
            targ_vec[task[2]] = 1.
            if self.task_list[task[0]].find('move') >= 0:
                obj_vec[task[1]] = 1.
                targ_vec[:] = 1. / len(targ_vec)
            #elif self.task_list[task[0]].find('transfer') >= 0:
            #    obj_vec[:] = 1. / len(obj_vec)
            #    targ_vec[task[2]] = 1.
            #obj_vec[task[1]] = 1.
            #targ_vec[task[2]] = 1.
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
        sample.set(OBJ_POSE_ENUM, obj_pose.copy(), t)
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

        if self.task_list[task[0]].find('move') >= 0:
            sample.set(END_POSE_ENUM, obj_pose + grasp, t)
            #sample.set(END_POSE_ENUM, obj_pose.copy(), t)
        if self.task_list[task[0]].find('transfer') >= 0:
            sample.set(END_POSE_ENUM, targ_pose + grasp, t)
            #sample.set(END_POSE_ENUM, targ_pose.copy(), t)
        for i, obj in enumerate(prim_choices[OBJ_ENUM]):
            sample.set(OBJ_ENUMS[i], mp_state[self.state_inds[obj, 'pose']], t)
            targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            sample.set(OBJ_DELTA_ENUMS[i], mp_state[self.state_inds[obj, 'pose']]-ee_pose, t)
            sample.set(TARG_ENUMS[i], targ-mp_state[self.state_inds[obj, 'pose']], t)

        if fill_obs:
            if IM_ENUM in self._hyperparams['obs_include'] or \
               IM_ENUM in self._hyperparams['prim_obs_include']:
                self.reset_mjc_env(sample.get_X(t=t), targets, draw_targets=True)
                im = self.mjc_env.render(height=self.image_height, width=self.image_width, view=self.view)
                im = (im - 128.) / 128.
                sample.set(IM_ENUM, im.flatten(), t)


    def goal_f(self, condition, state, targets=None, cont=False, anywhere=False, tol=LOCAL_NEAR_TOL):
        if targets is None:
            targets = self.target_vecs[condition]
        cost = self.prob.NUM_OBJS
        alldisp = 0
        plan = list(self.plans.values())[0]
        no = self._hyperparams['num_objs']
        if len(np.shape(state)) < 2:
            state = [state]
        for param in list(plan.params.values()):
            if 'Item' in param.get_type(True) and ('{0}_end_target'.format(param.name), 'value') in self.target_inds:
                if anywhere:
                    vals = [targets[self.target_inds[key, 'value']] for key, _ in self.target_inds if key.find('end_target') >= 0]
                else:
                    vals = [targets[self.target_inds['{0}_end_target'.format(param.name), 'value']]]
                dist = np.inf
                disp = None
                for x in state:
                    if self.goal_type == 'moveto':
                        vals = [x[self.state_inds['pr2', 'pose']]]
                    for val in vals:
                        curdisp = x[self.state_inds[param.name, 'pose']] - val
                        curdist = np.linalg.norm(curdisp)
                        if curdist < dist:
                            disp = curdisp
                            dist = curdist
                # np.sum((state[self.state_inds[param.name, 'pose']] - self.targets[condition]['{0}_end_target'.format(param.name)])**2)
                # cost -= 1 if dist < 0.3 else 0
                alldisp += curdist # np.linalg.norm(disp)
                cost -= 1 if np.all(np.abs(disp) < tol) else 0

        if cont: return alldisp / float(no)
        # return cost / float(self.prob.NUM_OBJS)
        return 1. if cost > 0 else 0.


    def reset_to_sample(self, sample):
        self.reset_to_state(sample.get_X(sample.T-1))


    def reset(self, m):
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        self._done = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self.eta_scale = 1.
        self._noops = 0
        self._x_delta[:] = x.reshape((1,-1))
        self._prev_task = np.zeros((self.task_hist_len, self.dPrimOut))
        self.cur_state = x.copy()
        self.mjc_env.reset()
        for (pname, aname), inds in self.state_inds.items():
            val = x[inds]
            self.mjc_env.set_attr(pname, aname, val, forward=False)
        self.mjc_env.physics.forward()


    def get_state(self):
        x = np.zeros(self.dX)
        for (pname, aname), inds in self.state_inds.items():
            x[inds] = self.mjc_env.get_attr(pname, aname)

        return x


    def reset_mjc_env(self, x, targets=None, draw_targets=True):
        pass


    def set_to_targets(self, condition=0):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        objs = prim_choices[OBJ_ENUM]
        for obj_name in objs:
            self.mjc_env.set_item_pos(obj_name, self.targets[condition]['{0}_end_target'.format(obj_name)], forward=False)
        self.mjc_env.physics.forward()


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
        # self.reset_to_state(x)
        # return self.mjc_env.get_obs(view=False)
        return self.mjc_env.render()


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
        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_traj), condition, state, task, noisy=False, skip_opt=True)
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


    def relabel_goal(self, path, debug=False):
        sample = path[-1]
        X = sample.get_X(sample.T-1)
        targets = sample.get(TARGETS_ENUM, t=sample.T-1).copy()
        assert np.sum([s.get(TARGETS_ENUM, t=2) - s.targets for s in path]) < 0.001
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for n, obj in enumerate(prim_choices[OBJ_ENUM]):
            pos = X[self.state_inds[obj, 'pose']]
            cur_targ = targets[self.target_inds['{0}_end_target'.format(obj), 'value']]
            prev_targ = cur_targ.copy()
            for opt in self.targ_labels:
                if np.all(np.abs(pos - self.targ_labels[opt]) < NEAR_TOL):
                    cur_targ = self.targ_labels[opt]
                    break
            targets[self.target_inds['{0}_end_target'.format(obj), 'value']] = cur_targ
            if TARG_ENUMS[n] in self._prim_obs_data_idx:
                for s in path:
                    new_disp = s.get(TARG_ENUMS[n]) + (cur_targ - prev_targ).reshape((1, -1))
                    s.set(TARG_ENUMS[n], new_disp)
        only_goal = np.concatenate([targets[self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]])
        onehot_goal = self.onehot_encode_goal(only_goal, debug=debug)
        for enum, val in zip([GOAL_ENUM, ONEHOT_GOAL_ENUM, TARGETS_ENUM], [only_goal, onehot_goal, targets]):
            for s in path:
                for t in range(s.T):
                    s.set(enum, val, t=t)
        for s in path: s.success = 1-self.goal_f(0, s.get(STATE_ENUM, t=s.T-1), targets=s.get(TARGETS_ENUM, t=s.T-1))
        for s in path: s.targets = targets
        return {GOAL_ENUM: only_goal, ONEHOT_GOAL_ENUM: onehot_goal, TARGETS_ENUM: targets}


    def replace_cond(self, cond, curric_step=-1):
        self.init_vecs[cond], self.targets[cond] = self.prob.get_random_initial_state_vec(self.config, self.targets, self.dX, self.state_inds, 1)
        self.init_vecs[cond], self.targets[cond] = self.init_vecs[cond][0], self.targets[cond][0]
        if self.master_config['easy']:
            self.init_vecs[cond][self.state_inds['pr2', 'pose']] = [0, -2.]
            for pname, aname in self.state_inds:
                inds = self.state_inds[pname, aname]
                if '{0}_end_target'.format(pname) in self.targets[cond]:
                    x, y = self.targets[cond]['{0}_end_target'.format(pname)]
                    if x < -5:
                        newx = x + np.random.uniform(1.5, 3.5)
                    elif x > 5:
                        newx = x - np.random.uniform(1.5, 3.5)
                    else:
                        newx = x + np.random.uniform(-2, 2)
                    if y < -5:
                        newy = y + np.random.uniform(1.5, 3.5)
                    elif y > 1:
                        newy = y - np.random.uniform(1.5, 3.5)
                    else:
                        newy = y + np.random.uniform(-2, 2)
                    self.init_vecs[cond][inds] = [newx, newy]
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        prim_choices = self.prob.get_prim_choices(self.task_list)
        if OBJ_ENUM in prim_choices and curric_step > 0:
            i = 0
            step = (curric_step + 1) // 2
            inds = np.random.permutation(list(range(len(prim_choices[OBJ_ENUM]))))
            for j in inds:
                obj = prim_choices[OBJ_ENUM][j]
                if '{0}_end_target'.format(obj) not in self.targets[cond]: continue
                if i >= len(prim_choices[OBJ_ENUM]) - step: break
                self.x0[cond][self.state_inds[obj, 'pose']] = self.targets[cond]['{0}_end_target'.format(obj)]
                i += 1
            if curric_step % 2 and step <= len(prim_choices[OBJ_ENUM]):
                grasp = np.array([0, -0.601])
                if GRASP_ENUM in prim_choices:
                    g = np.random.randint(len(prim_choices[GRASP_ENUM]))
                    grasp = self.set_grasp(grasp, g)

                self.x0[cond][self.state_inds['pr2', 'pose']] = self.x0[cond][self.state_inds['can{0}'.format(inds[len(prim_choices[OBJ_ENUM]) - step]), 'pose']] + grasp


        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]
        only_goal = np.concatenate([self.target_vecs[cond][self.target_inds['{0}_end_target'.format(o), 'value']] for o in prim_choices[OBJ_ENUM]])
        onehot_goal = self.onehot_encode_goal(only_goal)

        nt = len(prim_choices[TARG_ENUM])


    def goal(self, cond, targets=None):
        if self.goal_type == 'moveto':
            assert ('can1', 'pose') not in self.state_inds
            return '(RobotAtGrasp pr2 can0) '
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


    def check_target(self, targ):
        vec = np.zeros(len(list(self.targ_labels.keys())))
        for ind in self.targ_labels:
            if np.all(np.abs(targ - self.targ_labels[ind]) < NEAR_TOL):
                vec[ind] = 1.
                break
        return vec


    def onehot_encode_goal(self, targets, descr=None, debug=False):
        vecs = []
        for i in range(0, len(targets), 2):
            targ = targets[i:i+2]
            vec = self.check_target(targ)
            vecs.append(vec)
        if debug:
            print(('Encoded {0} as {1} {2}'.format(targets, vecs, self.prob.END_TARGETS)))
        return np.concatenate(vecs)


    def get_mask(self, sample, enum):
        mask = np.ones((sample.T, 1))
        return mask


    def permute_hl_data(self, hl_mu, hl_obs, hl_wt, hl_prc, aux):
        for enum in [IM_ENUM, OVERHEAD_IMAGE_ENUM]:
            if enum in self._prim_obs_data_idx:
                return hl_mu, hl_obs, hl_wt, hl_prc

        #print('-> Permuting data')
        assert len(hl_mu) == len(hl_obs)
        start_t = time.time()
        prim_opts = self.prob.get_prim_choices(self.task_list)
        objs = prim_opts[OBJ_ENUM]
        idx = self._prim_out_data_idx[OBJ_ENUM]
        a, b = min(idx), max(idx)+1
        no = self._hyperparams['num_objs']
        obs_idx = None
        if OBJ_ENUMS[0] in self._prim_obs_data_idx:
            obs_idx = [self._prim_obs_data_idx[OBJ_ENUMS[n]] for n in range(no)]
        obs_idx2 = None
        if OBJ_DELTA_ENUMS[0] in self._prim_obs_data_idx:
            obs_idx2 = [self._prim_obs_data_idx[OBJ_DELTA_ENUMS[n]] for n in range(no)]
        targ_idx = None
        if TARG_ENUMS[0] in self._prim_obs_data_idx:
            targ_idx = [self._prim_obs_data_idx[TARG_ENUMS[n]] for n in range(no)]
        goal_idx = self._prim_obs_data_idx[ONEHOT_GOAL_ENUM]
        hist_idx = self._prim_obs_data_idx.get(TASK_HIST_ENUM, None)
        xhist_idx = self._prim_obs_data_idx.get(STATE_DELTA_ENUM, None)

        inds = np.where(aux == 1)[0]
        save_inds = np.where(aux == 0)[0]
        new_mu = hl_mu[inds].copy()
        new_obs = hl_obs[inds].copy()
        save_mu = hl_mu[save_inds]
        save_obs = hl_obs[save_inds]
        hl_mu = hl_mu[inds]
        hl_obs = hl_obs[inds]

        old_goals = hl_obs[:,goal_idx]
        ng = len(goal_idx) // no
        order = np.random.permutation(range(no))
        rev_order = [order.tolist().index(n) for n in range(no)]
        nperm = 500
        for t in range(0, len(hl_mu), nperm):
            order = np.random.permutation(range(no))
            rev_order = [order.tolist().index(n) for n in range(no)]
            cur_inds = np.array([self.state_inds[obj, 'pose'] for obj in objs])
            new_mu[t:t+nperm][:,a:b] = hl_mu[t:t+nperm][:,a:b][:,order]
            if xhist_idx is not None:
                hist = hl_obs[t:t+nperm][:,xhist_idx].reshape((-1,self.hist_len,self.dX))
                new_hist = hist.copy()
                new_hist[:, np.r_[cur_inds]] = new_hist[:, np.r_[cur_inds[order]]]
                new_obs[t:t+nperm][:,xhist_idx] = new_hist.reshape((-1, 1, self.hist_len*self.dX))
            for n in range(no):
                if obs_idx is not None: new_obs[t:t+nperm][:,obs_idx[rev_order[n]]] = hl_obs[t:t+nperm][:,obs_idx[n]]
                if obs_idx2 is not None: new_obs[t:t+nperm][:,obs_idx2[rev_order[n]]] = hl_obs[t:t+nperm][:,obs_idx2[n]]
                if targ_idx is not None: new_obs[t:t+nperm][:,targ_idx[rev_order[n]]] = hl_obs[t:t+nperm][:,targ_idx[n]]
            new_obs[t:t+nperm][:, goal_idx] = np.concatenate([old_goals[t:t+nperm][:,order[n]*ng:(order[n]+1)*ng] for n in range(no)], axis=-1)
            if hist_idx is not None:
                hist = hl_obs[t:t+nperm][:,hist_idx].reshape((-1,self.task_hist_len,self.dPrimOut))
                new_hist = hist.copy()
                new_hist[:,a:b] = hist[:,a:b][:, order]
                new_obs[t:t+nperm][:, hist_idx] = new_hist.reshape((-1, 1, self.dPrimOut*self.task_hist_len))

        #print('Permuted with order', order, [hl_obs[-1][obs_idx2[n]] for n in range(no)], [new_obs[-1][obs_idx2[n]] for n in range(no)], hl_mu[-1, a:b], new_mu[-1, a:b])
        #print(hl_obs[-1,-1][xhist_idx].reshape((self.hist_len, -1)))
        #print(new_obs[-1,-1][xhist_idx].reshape((self.hist_len, -1)))
        #print(hl_obs[-1,-1][goal_idx])
        #print(new_obs[-1,-1][goal_idx])
        #print(hl_obs[-1,-1][hist_idx].reshape((self.task_hist_len, -1)))
        #print(new_obs[-1,-1][hist_idx].reshape((self.task_hist_len, -1)))
        new_wt = np.r_[hl_wt[save_inds], hl_wt[inds]]
        new_prc = np.r_[hl_prc[save_inds], hl_prc[inds]]
        return np.r_[save_mu, new_mu], np.r_[save_obs, new_obs], new_wt, new_prc


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
            if action.name.lower().find(task) >= 0:
                l[0] = i
                break

        for enum in prim_choices:
            if enum is TASK_ENUM: continue
            l.append(0)
            for i, opt in enumerate(prim_choices[enum]):
                if opt in [p.name for p in action.params]:
                    l[-1] = i
                    break
        return l # tuple(l)


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
            grip_interp = scipy.interpolate.interp1d(np.array(xpts), grippts, kind='next', bounds_error=False, axis=0)

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


    def compare_tasks(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1]

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
                    obj = plan.params[objname]
                    targ = plan.params[targname]
                    #if taskname.find('moveto') >= 0:
                    #    pred = HLGraspFailed('hlgraspfailed', [obj, grasp], ['Can', 'Grasp'])
                    #elif taskname.find('transfer') >= 0:
                    #    pred = HLTransferFailed('hltransferfailed', [obj, targ, grasp], ['Can', 'Target', 'Grasp'])
                    #plan.hl_preds.append(pred)
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
        return super(RobotAgent, self).backtrack_solve(plan, anum, n_resamples, rollout)


