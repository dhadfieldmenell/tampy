from abc import ABCMeta, abstractmethod
import copy
import random
import itertools
import sys
import time
import traceback

import pickle as pickle

import ctypes

import numpy as np
import scipy.interpolate

import xml.etree.ElementTree as xml

from sco.expr import *

import core.util_classes.common_constants as common_const
from pma.pr_graph import *
if common_const.USE_OPENRAVE:
    import openravepy
    from openravepy import RaveCreatePhysicsEngine

    import ctrajoptpy
else:
    import pybullet as p


# from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from policy_hooks.sample_list import SampleList

import core.util_classes.items as items
from core.util_classes.namo_predicates import dsafe
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer

from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.utils.load_task_definitions import *

MAX_SAMPLELISTS = 1000
MAX_TASK_PATHS = 1000


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


class TAMPAgent(Agent, metaclass=ABCMeta):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        # Note: All plans should contain identical sets of parameters
        self.config = hyperparams
        self.prob = hyperparams['prob']
        self.plans = self._hyperparams['plans']
        self.plans_list = list(self.plans.values())
        self.task_list = self._hyperparams['task_list']
        self.task_durations = self._hyperparams['task_durations']
        self.task_encoding = self._hyperparams['task_encoding']
        # self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self._samples = {task: [] for task in self.task_list}
        self._hl_probs = []
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        # self.dX = self._hyperparams['dX']
        self.image_width = hyperparams.get('image_width', utils.IM_W)
        self.image_height = hyperparams.get('image_height', utils.IM_H)
        self.image_channels = 3
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.rollout_seed = self._hyperparams['rollout_seed']
        self.num_objs = self._hyperparams['num_objs']
        self.init_vecs = self._hyperparams['x0']
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.targets = self._hyperparams['targets']
        self.target_dim = self._hyperparams['target_dim']
        self.target_inds = self._hyperparams['target_inds']
        self.target_vecs = []
        self.master_config = hyperparams['master_config']
        self.rank = hyperparams['master_config'].get('rank', 0)
        self.process_id = self.master_config['id']
        self.goal_type = self.master_config.get('goal_type', 'default')
        self.retime = hyperparams['master_config'].get('retime', False)
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)
        self.goals = self.target_vecs
        self.task_to_onehot = {}
        for i, task in enumerate(self.plans.keys()):
            self.task_to_onehot[i] = task
            self.task_to_onehot[task] = i
        self.sensor_dims = self._hyperparams['sensor_dims']
        self.discrete_prim = self._hyperparams.get('discrete_prim', True)
        # self.targ_list = self.targets[0].keys()
        # self.obj_list = self._hyperparams['obj_list']

        self.policies = {task: None for task in self.task_list}
        self._get_hl_plan = self._hyperparams['get_hl_plan']
        self.attr_map = self._hyperparams['attr_map']
        self.env = self._hyperparams['env']
        self.openrave_bodies = self._hyperparams['openrave_bodies']
        if self._hyperparams['viewer'] and self.env is not None:
            self.viewer = OpenRAVEViewer(self.env)
        else:
            self.viewer = None

        self._done = 0.
        self._task_done = 0.
        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        opts = self._hyperparams['prob'].get_prim_choices(self.task_list)
        self.label_options = list(itertools.product(*[list(range(len(opts[e]))) for e in opts])) # range(self.num_tasks), *[range(n) for n in self.num_prims]))
        self.hist_len = self._hyperparams['hist_len']
        self.task_hist_len = self._hyperparams.get('task_hist_len', 1)
        self.traj_hist = None
        self.reset_hist()

        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._prev_task = np.zeros((self.hist_len, self.dPrimOut))
        self.optimal_samples = {task: [] for task in self.task_list}
        self.optimal_state_traj = [[] for _ in range(len(self.plans_list))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans_list))]
        self.optimal_pol_cls = optimal_pol

        self.task_paths = []

        # self.get_plan = self._hyperparams['get_plan']
        self.move_limit = 1e-3

        self.n_policy_calls = {}
        if common_const.USE_OPENRAVE:
            self._cc = openravepy.RaveCreateCollisionChecker(self.env, "ode")
            self._cc.SetCollisionOptions(openravepy.CollisionOptions.Contacts)
            self.env.SetCollisionChecker(self._cc)
        # self._cc = ctrajoptpy.GetCollisionChecker(self.env)
        self.n_dirs = self._hyperparams['n_dirs']
        self.seed = 1234
        self.prim_dims = self._hyperparams['prim_dims']
        self.prim_dims_keys = list(self.prim_dims.keys())
        self.permute_hl = self.master_config.get('permute_hl', False)

        self.solver = self._hyperparams['mp_solver_type'](self._hyperparams)
        if 'll_solver_type' in self._hyperparams['master_config']:
            self.ll_solver = self._hyperparams['master_config']['ll_solver_type'](self._hyperparams)
        else:
            self.ll_solver = self._hyperparams['mp_solver_type'](self._hyperparams)
        self.traj_smooth = self.master_config['traj_smooth']
        self.hl_solver = get_hl_solver(self.prob.domain_file)
        self.debug = True
        self.fail_log = 'tf_saved/'+self.master_config['weight_dir']+'/agent_fail_log_{0}.txt'
        self.n_opt = {task: 0 for task in self.plans}
        self.n_fail_opt = {task: 0 for task in self.plans}
        self.n_hl_plan = 0
        self.n_hl_fail = 0
        self.n_plans_run = 0
        self.n_plans_suc_run = 0


    def get_init_state(self, condition):
        return self.x0[condition][self._x_data_idx[STATE_ENUM]].copy()


    def add_gym_env(self):
        from policy_hooks.agent_env_wrapper import AgentEnvWrapper
        self.gymenv = AgentEnvWrapper(agent=self)


    def get_goal(self, condition):
        return self.goals[condition].copy()


    def get_x0(self, condition):
        return self.x0[condition].copy()


    def add_viewer(self):
        if hasattr(self, 'mjc_env'):
            self.mjc_env.add_viewer()
        else:
            self.viewer = OpenRAVEViewer(self.env)


    def get_opt_samples(self, task=None, clear=False):
        data = []
        if task is None:
            tasks = list(self.optimal_samples.keys())
        else:
            tasks = [task]
        for task in tasks:
            data.extend(self.optimal_samples[task])
            if clear:
                self.optimal_samples[task] = []
        return data


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(batch)

        return samples


    def store_hl_problem(self, x0, initial, goal, keep_prob=0.1, max_len=20):
        if np.random.uniform() < keep_prob:
            self._hl_probs.append((x0, initial, goal))
        self._hl_probs = self._hl_probs[-max_len:]


    def sample_hl_problem(self, eta=1.):
        if not len(self._hl_probs):
            return None, None
        inds = np.array(list(range(len(self._hl_probs))))
        wt = np.exp(inds / eta)
        wt /= np.sum(eta)
        ind = np.random.choice(inds, p=wt)
        return self._hl_probs.pop(ind)


    def add_sample_batch(self, samples, task):
        if type(task) is tuple:
            task = self.task_list[task[0]]
        if not hasattr(samples[0], '__getitem__'):
            if not isinstance(samples, SampleList):
                samples = SampleList(samples)
            self._samples[task].append(samples)
        else:
            for batch in samples:
                if not isinstance(batch, SampleList):
                    batch = SampleList(batch)
                self._samples[task].append(batch)
        while len(self._samples[task]) > MAX_SAMPLELISTS:
            del self._samples[task][0]


    def clear_samples(self, keep_prob=0., keep_opt_prob=1.):
        for task in self.task_list:
            n_keep = int(keep_prob * len(self._samples[task]))
            self._samples[task] = random.sample(self._samples[task], n_keep)

            n_opt_keep = int(keep_opt_prob * len(self.optimal_samples[task]))
            self.optimal_samples[task] = random.sample(self.optimal_samples[task], n_opt_keep)


    def store_x_hist(self, x):
        self._x_delta = x.reshape((self.hist_len+1, self.dX))


    def store_act_hist(self, u):
        self._prev_U = u.reshape((self.hist_len, self.dU))


    def store_task_hist(self, task):
        self._prev_task = task.reshape((self.task_hist_len, self.dPrimOut))


    def reset_sample_refs(self):
        for task in self.task_list:
            for batch in self._samples[task]:
                for sample in batch:
                    sample.set_ref_X(np.zeros((sample.T, self.symbolic_bound)))
                    sample.set_ref_U(np.zeros((sample.T, self.dU)))


    def reset_hist(self):
        self._prev_U = np.zeros((self.hist_len, self.dU))


    def add_task_paths(self, paths):
        self.task_paths.extend(paths)
        while len(self.task_paths) > MAX_TASK_PATHS:
            del self.task_paths[0]


    def get_task_paths(self):
        return copy.copy(self.task_paths)


    def clear_task_paths(self, keep_prob=0.):
        n_keep = int(keep_prob * len(self.task_paths))
        self.task_paths = random.sample(self.task_paths, n_keep)


    def get_hist(self):
        return copy.deepcopy(self.traj_hist)


    def animate_sample(self, sample):
        if hasattr(self, 'mjc_env'):
            for t in range(sample.T):
                mp_state = sample.get(STATE_ENUM, t)
                for param_name, attr in self.state_inds:
                    if attr == 'pose':
                        self.mjc_env.set_item_pos(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False)
                    elif attr == 'rotation':
                        self.mjc_env.set_item_rot(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False, use_euler=True)
                self.mjc_env.physics.forward()
                self.mjc_env.render(view=True)
                time.sleep(0.2)

        if self.viewer is None: return
        plan = self.plans_list[0]
        for p in list(plan.params.values()):
            if p.is_symbol(): continue
            p.openrave_body.set_pose(p.pose[:,0])

        state = sample.get(STATE_ENUM)
        for t in range(sample.T):
            for p_name, a_name in self.state_inds:
                if a_name == 'rotation' or plan.params[p_name].is_symbol(): continue
                if a_name == 'pose':
                    if (p_name, 'rotation') in self.state_inds:
                        pos = state[t, self.state_inds[p_name, a_name]]
                        rot = state[t, self.state_inds[p_name, 'rotation']]
                        plan.params[p_name].openrave_body.set_pose(pos, rot)
                    else:
                        pos = state[t, self.state_inds[p_name, a_name]]
                        plan.params[p_name].openrave_body.set_pose(pos)
                else:
                    attr_val = state[t, self.state_inds[p_name, a_name]]
                    plan.params[p_name].openrave_body.set_dof({a_name: attr_val})
            time.sleep(0.2)


    def draw_sample_ts(self, sample, t):
        if self.viewer is None: return
        plan = self.plans_list[0]
        for p in list(plan.params.values()):
            if p.is_symbol(): continue
            p.openrave_body.set_pose(p.pose[:,0])
        state = sample.get(STATE_ENUM)
        for p_name, a_name in self.state_inds:
            if a_name == 'rotation' or plan.params[p_name].is_symbol(): continue
            if a_name == 'pose':
                if (p_name, 'rotation') in self.state_inds:
                    pos = state[t, self.state_inds[p_name, a_name]]
                    rot = state[t, self.state_inds[p_name, 'rotation']]
                    plan.params[p_name].openrave_body.set_pose(pos, rot)
                else:
                    pos = state[t, self.state_inds[p_name, a_name]]
                    plan.params[p_name].openrave_body.set_pose(pos)
            else:
                attr_val = state[t, self.state_inds[p_name, a_name]]
                plan.params[p_name].openrave_body.set_dof({a_name: attr_val})


    def save_free(self, plan):
        old_params_free = {}
        for p in plan.params.values():
            p_attrs = {}
            old_params_free[p.name] = p_attrs
            for attr in p._free_attrs:
                p_attrs[attr] = p._free_attrs[attr].copy()
        self.saved_params_free = old_params_free


    def restore_free(self, plan):
        for p in self.saved_params_free:
            for attr in self.saved_params_free[p]:
                plan.params[p]._free_attrs[attr] = self.saved_params_free[p][attr].copy()

    def retime_sample(self, sample, dx=1e-3):
        '''
        Collapse steps below dx separation together
        Assumes transitivity of action space; i.e. (x0, u0, x1) and (x1, u1, x2) equals (x0, u1+u2, x2)
        Mutates samples in place
        '''
        cur_step = 0
        sample.use_ts[:] = 0.
        for t in range(sample.T):
            cur_X = sample.get_X(t=cur_step)
            next_X = cur_X
            full_U = sample.get_U(t=cur_step)
            while cur_step < sample.T - 1 and np.all(np.abs(cur_X - next_X) < dx):
                cur_step += 1
                next_X = sample.get_X(t=cur_step)
                full_U += sample.geT_U(t=cur_step)
            sample.set_X(next_X, t=t)
            sample.set_U(full_U, t=t)
            sample.use_ts[t] = 1.
        return sample


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    @abstractmethod
    def _sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, hor=None):
        raise NotImplementedError

    def sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True, task_f=None, skip_opt=False):
        if not skip_opt and (policy is None or (hasattr(policy, 'scale') and policy.scale is None)): # Policy is uninitialized
            s, failed, success = self.solve_sample_opt_traj(state, task, condition)
            s.opt_strength = 1.
            s.opt_suc = success
            return s
        s = self._sample_task(policy, condition, state, task, save_global=save_global, noisy=noisy, task_f=task_f)
        s.opt_strength = 0.
        s.opt_suc = False
        return s


    def resample(self, sample_lists, policy, num_samples):
        samples = []
        for slist in sample_lists:
            if hasattr(slist, '__len__') and not len(slist): continue
            new_samples = []
            for i in range(num_samples):
                s = slist[0] if hasattr(slist, '__getitem__') else slist
                if np.any(np.isnan(s.get_X(t=0))):
                    raise Exception('Nans in resample step state.')
                # self.reset_hist(s.get(TRAJ_HIST_ENUM, t=0).reshape((self.hist_len, 3)).tolist())
                s = self.sample_task(policy, s.condition, s.get_X(t=0), s.task, noisy=True)
                new_samples.append(s)
            samples.append(new_samples)
        s = samples[0][0]
        if np.any(np.isnan(s.get_U(t=1))):
            raise Exception('Nans in resample step action.')
        return samples


    # @abstractmethod
    # def dist_obs(self, plan, t):
    #     raise NotImplementedError


    # @abstractmethod
    # def run_policy_step(self, u, x, plan, t, obj):
    #     raise NotImplementedError


    @abstractmethod
    def goal(self, cond, targets=None):
        raise NotImplementedError


    @abstractmethod
    def goal_f(self, condition, state, targets=None, cont=False):
        raise NotImplementedError


    @abstractmethod
    def set_symbols(self, plan, task, anum=0, cond=0):
        raise NotImplementedError


    @abstractmethod
    def encode_action(self, action):
        raise NotImplementedError


    @abstractmethod
    def reset(self, m):
        raise NotImplementedError


    def _reset_to_state(self, x):
        self._done = 0.
        self._prev_U = np.zeros((self.hist_len, self.dU))
        self._x_delta = np.zeros((self.hist_len+1, self.dX))
        self._x_delta[:] = x.reshape((1,-1))
        self.mjc_env.reset()


    @abstractmethod
    def reset_to_state(self, x):
        raise NotImplementedError


    @abstractmethod
    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        raise NotImplementedError


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], fixed_targets=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean, fixed_targets)

        exclude_targets = []
        opt_disp_traj = np.zeros_like(opt_traj)
        for t in range(0, len(opt_traj)-1):
            opt_disp_traj[t] = opt_traj[t+1] - opt_traj[t]

        if len(fixed_targets):
            targets = fixed_targets
            obj = fixed_targets[0]
            targ = fixed_targets[1]
        else:
            task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))
            obj = self.plans_list[0].params[self.obj_list[np.argmax(obj_distr)]]
            targ = self.plans_list[0].params[self.targ_list[np.argmax(targ_distr)]]
            targets = [obj, targ]
            # targets = get_next_target(self.plans_list[0], state, task, self.targets[condition], sample_traj=traj_mean)

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_disp_traj), condition, state, [task, targets[0].name, targets[1].name], noisy=False, fixed_obj=True)
        self.optimal_samples[task].append(sample)
        sample.set_ref_X(sample.get(STATE_ENUM))
        sample.set_ref_U(sample.get_U())
        return sample


    @abstractmethod
    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], fixed_targets=[]):
        raise NotImplementedError


    def _sample_opt_traj(self, plan, state, task, condition):
        '''
        Only call for successfully planned trajectories
        '''
        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    fill_vector(plan.params, plan.action_inds, U, t+1)
                else:
                    fill_vector(plan.params, plan.action_inds, U, t)
                return U

        state_traj = np.zeros((plan.horizon, self.symbolic_bound))
        for i in range(plan.horizon):
            fill_vector(plan.params, plan.state_inds, state_traj[i], i)
        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)
        self.optimal_samples[self.task_list[task[0]]].append(sample)
        sample.set_ref_X(state_traj)
        sample.set_ref_U(sample.get_U())
        return sample, [], True


    def perturb_solve(self, sample, perturb_var=0.02, inf_f=None):
        x0 = sample.get(STATE_ENUM, t=0)
        saved_targets = {}
        cond = sample.condition
        saved_target_vec = self.target_vecs[cond].copy()
        for obj in self.obj_list:
            obj_p = self.plans_list[0].params[obj]
            if obj_p._type == 'Robot': continue
            x0[self.state_inds[obj, 'pose']] += np.random.normal(0, perturb_var, obj_p.pose.shape[0])
        old_targets = sample.get(TARGETS_ENUM, t=0)
        for target_name in self.targets[cond]:
            old_value = old_targets[self.target_inds[target_name, 'value']]
            saved_targets[target_name] = old_value
            self.targets[cond][target_name] = old_value + np.random.normal(0, perturb_var, old_value.shape)
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]
        target_params = [self.plans_list[0].params[sample.obj], self.plans_list[0].params[sample.targ]]
        out, success, failed = self.solve_sample_opt_traj(x0, sample.task, sample.condition, sample.get_U(), target_params, inf_f=inf_f)
        for target_name in self.targets[cond]:
            self.targets[cond][target_name] = saved_targets[target_name]
        self.target_vecs[cond] = saved_target_vec
        return out, failed, success


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        return self._get_hl_plan(state, self.targets[condition], '{0}{1}'.format(condition, plan_id), self.plans_list[0].params, self.state_inds, failed_preds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets


    def replace_targets(self, condition=0):
        new_targets = self.prob.get_end_targets(self.prob.NUM_OBJS, randomize=False)
        self.targets[condition] = new_targets
        target_vec = np.zeros((self.target_dim,))
        for target_name in self.targets[condition]:
            target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
        self.target_vecs[condition]= target_vec


    # @abstractmethod
    # def get_sample_constr_cost(self, sample):
    #     raise NotImplementedError


    def randomize_init_state(self, condition=0):
        self.targets[condition] = self.prob.get_end_targets(self.num_objs)
        self.init_vecs[condition] = self.prob.get_random_initial_state_vec(self.config, self.num_objs, self.targets, self.dX, self.state_inds, 1)[0]
        self.x0[condition] = self.init_vecs[condition][:self.symbolic_bound]
        target_vec = np.zeros((self.target_dim,))
        for target_name in self.targets[condition]:
            target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
        self.target_vecs[condition] = target_vec


    '''
    def replace_conditions(self, conditions, keep=(0., 0.)):
        self.targets = []
        for i in range(conditions):
            self.targets.append(self.prob.get_end_targets(self.num_objs))
        self.init_vecs = self.prob.get_random_initial_state_vec(self.num_objs, self.targets, self.dX, self.state_inds, conditions)
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.target_vecs = []
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)

        if keep != (1., 1.):
            self.clear_samples(*keep)
    '''
    def replace_conditions(self, conditions=None, curric_step=-1):
        if conditions is None:
            conditions = list(range(len(self.x0)))
        for c in conditions:
            self.replace_cond(c, curric_step)

    '''
    def replace_cond(self, cond, curric_step=-1):
        self.targets[cond] = self.prob.get_end_targets(self.num_objs)
        self.init_vecs[cond] = self.prob.get_random_initial_state_vec(self.config, self.num_objs, self.targets, self.dX, self.state_inds, 1)[0]
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        prim_choices = self.prob.get_prim_choices()
        if OBJ_ENUM in prim_choices and curric_step > 0:
            i = 0
            inds = np.random.permutation(range(len(prim_choices[OBJ_ENUM])))
            for j in inds:
                obj = prim_choices[OBJ_ENUM][j]
                if '{0}_end_target'.format(obj) not in self.targets[cond]: continue
                if i >= len(prim_choices[OBJ_ENUM]) - curric_step: break
                self.x0[cond][self.state_inds[obj, 'pose']] = self.targets[cond]['{0}_end_target'.format(obj)]
                i += 1

        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]
    '''

    def replace_cond(self, cond, curric_step=-1):
        self.init_vecs[cond], self.targets[cond] = self.prob.get_random_initial_state_vec(self.config, self.targets, self.dX, self.state_inds, 1)
        self.init_vecs[cond], self.targets[cond] = self.init_vecs[cond][0], self.targets[cond][0]
        self.x0[cond] = self.init_vecs[cond][:self.symbolic_bound]
        self.target_vecs[cond] = np.zeros((self.target_dim,))
        prim_choices = self.prob.get_prim_choices(self.task_list)
        if OBJ_ENUM in prim_choices and curric_step > 0:
            i = 0
            inds = np.random.permutation(list(range(len(prim_choices[OBJ_ENUM]))))
            for j in inds:
                obj = prim_choices[OBJ_ENUM][j]
                if '{0}_end_target'.format(obj) not in self.targets[cond]: continue
                if i >= len(prim_choices[OBJ_ENUM]) - curric_step: break
                self.x0[cond][self.state_inds[obj, 'pose']] = self.targets[cond]['{0}_end_target'.format(obj)]
                i += 1

        for target_name in self.targets[cond]:
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]


    def perturb_conditions(self, perturb_var=0.02):
        self.perturb_init_states(perturb_var)
        self.perturb_targets(perturb_var)


    def perturb_init_states(self, perturb_var=0.02):
        for c in range(len(self.x0)):
            x0 = self.x0[c]
            for obj in self.obj_list:
                obj_p = self.plans_list[0].params[obj]
                if obj.is_symbol() or obj._type == 'Robot': continue
                x0[self.state_inds[obj, 'pose']] += np.random.normal(0, perturb_var, obj_p.pose.shape[0])


    def perturb_targets(self, perturb_var=0.02):
        for c in range(len(self.x0)):
            for target_name in self.targets[c]:
                target_p = self.plans_list[0].params[target_name]
                self.targets[c][target_name] += np.random.normal(0, perturb_var, target_p.value.shape[0])
                self.target_vecs[c][self.target_inds[target_name, 'value']] = self.targets[c][target_name]


    def get_prim_options(self, cond, state):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = self.prob.get_prim_choices(self.task_list)
        plan = self.plans_list[0]
        for enum in self.prim_dims:
            if enum == TASK_ENUM: continue
            out[enum] = []
            for item in options[enum]:
                if item in plan.params:
                    param = plan.params[item]
                    if param.is_symbol():
                        out[enum].append(param.value[:,0].copy())
                    else:
                        out[enum].append(mp_state[self.state_inds[item, 'pose']].copy())
                    continue

                # val = self.env.get_pos_from_label(item, mujoco_frame=False)
                # if val is not None:
                #     out[enum] = val
                # out[enum].append(val)
            out[enum] = np.array(out[enum])
        return out


    def get_prim_value(self, cond, state, task):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = self.task_list[task[0]]
        plan = self.plans[task]
        options = self.prob.get_prim_choices(self.task_list)
        for i in range(1, len(task)):
            enum = self.prim_dims_keys()[i-1]
            item = options[enum][task[i]]
            if item in plan.params:
                param = plan.params[item]
                if param.is_symbol():
                    out[enum] = param.value[:,0]
                else:
                    out[enum] = mp_state[self.state_inds[item, 'pose']]
                continue

            # val = self.env.get_pos_from_label(item, mujoco_frame=False)
            # if val is not None:
            #     out[enum] = val

        return out


    def get_prim_index(self, enum, name):
        prim_options = self.prob.get_prim_choices(self.task_list)
        return prim_options[enum].index(name)


    def get_prim_indices(self, names):
        task = [self.task_list.index(names[0])]
        for i in range(1, len(names)):
            task.append(self.get_prim_index(self.prim_dims_keys()[i-1], names[i]))
        return tuple(task)


    def get_target_dict(self, cond):
        info = {}
        for target_name in self.targets[cond]:
            info[target_name] = list(self.targets[cond][target_name])
        return info


    def get_trajectories(self, sample=None, mp_state=None):
        if sample is not None:
            mp_state = sample.get(STATE_ENUM)

        info = {}
        plan = list(self.plans.values())[0]
        for param_name, attr in self.state_inds:
            if plan.params[param_name].is_symbol(): continue
            info[param_name, attr] = mp_state[:, self.state_inds[param_name, attr]]
        return info


    def get_ee_trajectories(self, sample):
        info = {}
        if LEFT_EE_POS_ENUM in self.x_data_types or LEFT_EE_POS_ENUM in self.obs_data_types:
            info[LEFT_EE_POS_ENUM] = sample.get(LEFT_EE_POS_ENUM)
        if RIGHT_EE_POS_ENUM in self.x_data_types or RIGHT_EE_POS_ENUM in self.obs_data_types:
            info[LEFT_EE_POS_ENUM] = sample.get(RIGHT_EE_POS_ENUM)
        if EE_ENUM in self.x_data_types or EE_ENUM in self.obs_data_types:
            info[EE_ENUM] = sample.get(EE_ENUM)
        if EE_POS_ENUM in self.x_data_types or EE_POS_ENUM in self.obs_data_types:
            info[EE_POS_ENUM] = sample.get(EE_POS_ENUM)
        if EE_2D_ENUM in self.x_data_types or EE_2D_ENUM in self.obs_data_types:
            info[EE_2D_ENUM] = sample.get(EE_2D_ENUM)
        if GRIPPER_ENUM in self.x_data_types or GRIPPER_ENUM in self.obs_data_types:
            info[GRIPPER_ENUM] = sample.get(GRIPPER_ENUM)
        if LEFT_GRIPPER_ENUM in self.x_data_types or LEFT_GRIPPER_ENUM in self.obs_data_types:
            info[LEFT_GRIPPER_ENUM] = sample.get(LEFT_GRIPPER_ENUM)
        if RIGHT_GRIPPER_ENUM in self.x_data_types or RIGHT_GRIPPER_ENUM in self.obs_data_types:
            info[RIGHT_GRIPPER_ENUM] = sample.get(RIGHT_GRIPPER_ENUM)
        return info


    def get_obj_dict(self, mp_state, t=None):
        assert t is None and len(mp_state.shape) == 1 or t is not None and len(mp_state) == 2
        info = {}
        for param_name, attr in self.state_inds:
            if list(self.plans.values())[0].params[param_name].is_symbol(): continue
            inds = self.state_inds[param_name, attr]
            info[param_name, attr] = mp_state[inds] if t is None else mp_state[t, inds]
        return info


    def get_obj_trajectory(self, obj_name, sample=None, mp_state=None):
        if sample is not None:
            mp_state = sample.get(STATE_ENUM)

        if mp_state is None or (obj_name, 'pose') not in self.state_inds: return {obj_name: 'NO TRAJ DATA'}
        return mp_state[:, self.state_inds[obj_name, 'pose']]


    def get_obj_trajectories(self, sample=None, mp_state=None):
        if sample is not None:
            mp_state = sample.get(STATE_ENUM)

        info = {}
        mp_state = mp_state.copy()
        if mp_state is None: return info
        for param_name, attr in self.state_inds:
            if list(self.plans.values())[0].params[param_name].is_symbol(): continue
            info[param_name, attr] = self.get_obj_trajectory(param_name, mp_state=mp_state)
        return info


    def postcond_cost(self, sample, task=None, t=None):
        if t is None: t = sample.T-1
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(-1, -1), targets=sample.targets)


    def precond_cost(self, sample, task=None, t=0):
        if task is None: task = tuple(sample.get(FACTOREDTASK_ENUM, t=t))
        return self.cost_f(sample.get_X(t), task, sample.condition, active_ts=(0, 0), targets=sample.targets)


    def relabel_path(self, path):
        end = path[-1]
        start_X = path[0].get_X(0)
        end_X = end.get_X(end.T-1)
        goal = self.relabel_goal(end)
        if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims'] and np.all(goal[ONEHOT_GOAL_ENUM] == 0.):
            return []

        new_path = []
        cur_s = path[0]
        i = 0
        while self.goal_f(end.condition, start_X, goal[TARGETS_ENUM]) > 1e-2:
            new_s = Sample(self)
            new_s.targets = goal[TARGETS_ENUM]
            new_s.set_val_obs(path[i].get_val_obs())
            new_s.set_prim_obs(path[i].get_prim_obs())
            new_s.set_X(path[i].get_X())
            new_s.success = 1.
            for t in range(new_s.T):
                new_s.set(TARGETS_ENUM, goal[TARGETS_ENUM], t)
                new_s.set(GOAL_ENUM, goal[GOAL_ENUM], t)
                if ONEHOT_GOAL_ENUM in self._hyperparams['sensor_dims']:
                    new_s.set(ONEHOT_GOAL_ENUM, goal[ONEHOT_GOAL_ENUM], t)
            new_path.append(new_s)
            start_X = new_path[-1].get_X(new_path[-1].T-1)
            i += 1

        for i, s in enumerate(new_path):
            s.discount = 0.9 ** (len(new_path) - i)
        return new_path


    def get_hl_info(self, state=None, targets=None, cond=0, plan=None, act=0):
        if targets is None:
            targets = self.target_vecs[cond].copy()

        initial = ['(RobotAt pr2 robot_init_pose)']
        plans = [plan]
        if plan is None:
            plans = list(self.plans.values())

        st = plans[0].actions[act].active_timesteps[0]
        for plan in plans:
            for pname, aname in self.state_inds:
                if plan.params[pname].is_symbol(): continue
                if state is not None: getattr(plan.params[pname], aname)[:,st] = state[self.state_inds[pname, aname]]
                init_t = '{0}_init_target'.format(pname)
                if init_t in plan.params and st == 0:
                    plan.params[init_t].value[:,0] = plan.params[pname].pose[:,st]
                    near_pred = '(Near {0} {1}) '.format(pname, init_t)
                    if near_pred not in initial:
                        initial.append(near_pred)
            for pname, aname in self.target_inds:
                getattr(plan.params[pname], aname)[:,0] = targets[self.target_inds[pname, aname]]
            init_preds = parse_state(plan, [], st)
            initial.extend([p.get_rep() for p in init_preds])
        goal = self.goal(cond, targets)
        return list(set(initial)), goal


    def solve_hl(self, state, targets):
        plan = list(self.plans.values())[0]
        prob = plan.prob
        initial, goal = self.get_hl_info(state, targets)
        abs_prob = self.hl_solver.translate_problem(prob, initial, goal)
        new_plan = self.hl_solver.solve(abs_prob, plan.domain, prob, label=self.process_id)
        return new_plan


    def task_from_ff(self, state, targets):
        plan = self.solve_hl(state, targets)
        if type(plan) == str:
            return None
        return self.encode_plan(plan)


    def check_curric(self, buf, n_thresh, curr_thresh, cur_curr):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        curr_thresh *= cur_curr
        if len(buf) < n_thresh: return False
        return np.mean(buf[-n_thresh:]) < curr_thresh


    def encode_action(self, action, next_act=None):
        prim_choices = self.prob.get_prim_choices(self.task_list)
        astr = str(action).lower()
        l = [0]
        for i, task in enumerate(self.task_list):
            if action.name.lower().find(task) >= 0:
                l[0] = i
                break

        for enum in prim_choices:
            if enum is TASK_ENUM: continue
            l.append(-1)
            for act in [action, next_act]:
                for i, opt in enumerate(prim_choices[enum]):
                    if opt in [p.name for p in act.params]:
                        l[-1] = i
                        break
                if l[-1] >= 0: break
            if l[-1] < 0.: l[-1] = 0.
        return l # tuple(l)


    def encode_plan(self, plan, permute=False):
        encoded = []
        prim_choices = self.prob.get_prim_choices(self.task_list)
        for a in plan.actions:
            encoded.append(self.encode_action(a))
        encoded = [tuple(l) for l in encoded]
        return encoded


    def get_encoded_tasks(self):
        if hasattr(self, '_cached_encoded_tasks'):
            return self._cached_encoded_tasks
        opts = self.prob.get_prim_choices(self.task_list)
        nacts = np.prod([len(opts[e]) for e in opts])
        dact = np.sum([len(opts[e]) for e in opts])
        out = np.zeros((len(self.label_options), dact))
        for i, l in enumerate(self.label_options):
            cur_vec = np.zeros(0)
            for j, e in enumerate(opts.keys()):
                v = np.zeros(len(opts[e]))
                v[l[j]] = 1.
                cur_vec = np.concatenate([cur_vec, v])
            out[i, :] = cur_vec
        self._cached_encoded_tasks = out
        return out


    def _backtrack_solve(self, plan, anum=0, n_resamples=5):
        return self.backtrack_solve(plan, anum=anum, n_resamples=n_resamples)


    def backtrack_solve(self, plan, anum=0, n_resamples=5):
        # Handle to make PR Graph integration easier
        start = anum
        plan.state_inds = self.state_inds
        plan.action_inds = self.action_inds
        plan.dX = self.symbolic_bound
        plan.dU = self.dU
        success = False
        for a in range(anum, len(plan.actions)):
            x0 = np.zeros_like(self.x0[0])
            st, et = plan.actions[a].active_timesteps
            fill_vector(plan.params, self.state_inds, x0, st)
            task = tuple(self.encode_action(plan.actions[a]))

            traj = []
            success = False
            policy = self.policies[self.task_list[task[0]]]
            if self.rollout_seed and policy.scale is not None:
                sample = self.sample_task(policy, 0, x0.copy(), task)
                traj = np.zeros((plan.horizon, self.symbolic_bound))
                traj[st:et+1] = sample.get_X()
                fill_trajectory_from_sample(sample, plan, active_ts=(st+1,et-1))
                '''
                self.set_symbols(plan, x0, task, anum=a)
                free_attrs = plan.get_free_attrs()
                for param in plan.actions[a].params:
                    if not param.is_symbol():
                        for attr in param._free_attrs:
                            param._free_attrs[attr][:] = 0
                try:
                    self.solver._backtrack_solve(plan, anum=anum, amax=anum, n_resamples=1, max_priority=-2)
                    failed = plan.get_failed_preds((st,et), tol=1e-3)
                except Exception as e:
                    print(e, 'Exception to solve on', anum, plan.actions, x0)
                    failed = ['Bad solve!']
                plan.store_free_attrs(free_attrs)
                failed = list(filter(lambda p: not type(p[1].expr) is EqExpr, failed))
                success = len(failed) == 0
                '''

            if not success:
                fill_vector(plan.params, self.state_inds, x0, st)
                self.set_symbols(plan, task, anum=a)
                try:
                    success = self.ll_solver._backtrack_solve(plan, anum=a, amax=a, n_resamples=n_resamples, init_traj=traj)
                    if self.traj_smooth:
                        plan.backup_params()
                        suc = self.ll_solver.traj_smoother(plan)
                        if not suc:
                            plan.restore_params()
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
        #path = self.run_plan(plan)
        #self.add_task_paths([path])
        return success


    def run_plan(self, plan, targets, tasks=None, reset=True, permute=False):
        self.n_plans_run += 1
        path = []
        nzero = self.master_config.get('add_noop', 0)
        if tasks is None:
            tasks = self.encode_plan(plan)
        x0 = np.zeros_like(self.x0[0])
        fill_vector(plan.params, self.state_inds, x0, 0)
        perm = {}
        if permute:
            tasks, targets, perm = self.permute_tasks(tasks, targets, plan)
            for pname, aname in self.state_inds:
                if pname in perm:
                    x0[self.state_inds[perm[pname], aname]] = getattr(plan.params[pname], aname)[:,0]
        if reset:
            self.reset_to_state(x0)
        for a in range(len(plan.actions)):
            # x0 = np.zeros_like(self.x0[0])
            st, et = plan.actions[a].active_timesteps
            # fill_vector(plan.params, self.state_inds, x0, st)
            task = tasks[a]
            opt_traj = np.zeros((et-st+1, self.symbolic_bound))
            for pname, attr in self.state_inds:
                if plan.params[pname].is_symbol(): continue
                opt_traj[:,self.state_inds[perm.get(pname, pname), attr]] = getattr(plan.params[pname], attr)[:,st:et+1].T

            # self.reset_hist()
            cur_len = len(path)
            if self.retime:
                vel = self.master_config.get('velocity', 0.3)
                new_traj = self.retime_traj(opt_traj, vel=vel)
                # for _ in range(nzero):
                #     new_traj = np.r_[new_traj, new_traj[-1:]]
                T = et-st+1
                t = 0
                ind = 0
                while t < len(new_traj)-1:
                    # traj = new_traj[t:t+T+1]
                    traj = new_traj[t:t+T]
                    # if np.all(np.abs(traj[-1]-traj[0]) < 1e-5): break
                    sample = self.sample_optimal_trajectory(x0, task, 0, traj, targets=targets)
                    # self.add_sample_batch([sample], task)
                    path.append(sample)
                    sample.discount = 1.
                    sample.opt_strength = 1.
                    sample.opt_suc = True
                    sample.step = ind
                    sample.task_end = False
                    ind += 1
                    t += T - 1
                    x0 = sample.end_state # sample.get_X(t=sample.T-1)
                    sample.success = 1. - self.goal_f(0, x0, sample.targets)
                    '''
                    if np.all(np.abs(traj[-1]-traj[0]) < 1e-5):
                        sample.use_ts[:] = 0.
                        sample.use_ts[:nzero] = 1.
                        sample.prim_use_ts[:] = 0.
                    elif t >= len(new_traj)-1:
                        sample.use_ts[len(traj)-1] = 1.
                        sample.prim_use_ts[len(traj)-1] = 1.
                        if nzero > 0: sample.use_ts[-nzero:] = 1.
                    '''
            else:
                sample = self.sample_optimal_trajectory(x0, task, 0, opt_traj, targets=targets)
                #zero_sample = self.sample_optimal_trajectory(sample.end_state, task, 0, opt_traj[-1:], targets=targets)
                # self.add_sample_batch([sample], task)
                path.append(sample)
                #path.append(zero_sample)
                #zero_sample.use_ts[:] = 0.
                #zero_sample.use_ts[:nzero] = 1.
                #zero_sample.prim_use_ts[:] = 0.
                sample.discount = 1.
                sample.opt_strength = 1.
                sample.opt_suc = True
                x0 = sample.end_state # sample.get_X(sample.T-1)
                sample.success = 1. - self.goal_f(0, x0, sample.targets)
                # zero_sample.success = sample.success
                #sample.use_ts[-1] = 1.
                #sample.prim_use_ts[-1] = 1.
            path[-1].task_end = True
            path[-1].set(TASK_DONE_ENUM, np.array([0, 1]), t=path[-1].T-1)
            zero_sample = self.sample_optimal_trajectory(path[-1].end_state, task, 0, opt_traj[-1:], targets=targets)
            zero_sample.use_ts[:] = 0.
            zero_sample.use_ts[:nzero] = 1.
            zero_sample.prim_use_ts[:] = 0.
            zero_sample.success = path[-1].success
            zero_sample.set(TASK_DONE_ENUM, np.tile([0,1], (zero_sample.T, 1)))
            path.append(zero_sample)
            # path[cur_len].set(DONE_ENUM, np.ones(1), t=0)
            path[-1].task_end = True
        if path[-1].success > 0.99:
            self.add_task_paths([path])
            for s in path:
                self.optimal_samples[self.task_list[s.task[0]]].append(s)
            self.n_plans_suc_run += 1
        print(('Plans run vs. success:', self.n_plans_run, self.n_plans_suc_run, self.process_id))
        return path

    def run_pr_graph(self, state, targets=None, cond=None, reset=True):
        if targets is None:
            targets = self.target_vecs[cond]
        initial, goal = self.get_hl_info(state, targets=targets)
        domain = list(self.plans.values())[0].domain
        prob = list(self.plans.values())[0].prob
        for pname, attr in self.state_inds:
            p = prob.init_state.params[pname]
            if p.is_symbol(): continue
            getattr(p, attr)[:,0] = state[self.state_inds[pname, attr]]
        for targ, attr in self.target_inds:
            if targ in prob.init_state.params:
                p = prob.init_state.params[targ]
                getattr(p, attr)[:,0] = self.target_vecs[0][self.target_inds[targ, attr]].copy()
        try:
            plan, descr = p_mod_abs(self.hl_solver, self, domain, prob, initial=initial, goal=goal, label=self.process_id)
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            plan = None
        self.n_hl_plan += 1
        if plan is None or type(plan) is str:
            self.n_hl_fail += 1
            return []
        path = self.run_plan(plan, targets=targets, reset=reset)
        return path

    def retime_traj(self, traj, vel=0.3, inds=None):
        xpts = []
        fpts = []
        d = 0
        for t in range(len(traj)):
            xpts.append(d)
            fpts.append(traj[t])
            if t < len(traj):
                if inds is None:
                    d += np.linalg.norm(traj[t+1] - traj[t])
                else:
                    d += np.linalg.norm(traj[t+1][inds] - traj[t][inds])
        interp = scipy.interpolate.interp1d(xpts, fpts, axis=0)

        x = np.linspace(0, d, int(d / vel))
        out = interp(x)
        return out


    def project_onto_constr(self, x0, plan, targets, ts=(0,0)):
        x0 = x0.copy()
        for pname, aname in self.state_inds:
            plan.params['robot_init_pose'].value[:,0] = x0[self.state_inds['pr2', 'pose']]
            if plan.params[pname].is_symbol(): continue
            getattr(plan.params[pname], aname)[:,0] = x0[self.state_inds[pname, aname]]
            plan.params[pname]._free_attrs[aname][:,0] = 1.
            if '{0}_init_target'.format(pname) in plan.params:
                plan.params['{0}_init_target'.format(pname)].value[:,0] = x0[self.state_inds[pname, aname]]
        suc = self.solver.solve(plan, traj_mean=np.array(x0).reshape((1,-1)), active_ts=(0,0))
        for pname, aname in self.state_inds:
            if plan.params[pname].is_symbol(): continue
            x0[self.state_inds[pname, aname]] = getattr(plan.params[pname], aname)[:,0]
        return x0


    def resample_hl_plan(self, plan, targets, n=5):
        x0s, _ = self.prob.get_random_initial_state_vec(self.config, None, self.symbolic_bound, self.state_inds, n)
        nsuc = 0
        for x0 in x0s:
            for pname, aname in self.state_inds:
                plan.params['robot_init_pose'].value[:,0] = x0[self.state_inds['pr2', 'pose']]
                if plan.params[pname].is_symbol(): continue
                getattr(plan.params[pname], aname)[:,0] = x0[self.state_inds[pname, aname]]
                plan.params[pname]._free_attrs[aname][:,0] = 1.
                if '{0}_init_target'.format(pname) in plan.params:
                    plan.params['{0}_init_target'.format(pname)].value[:,0] = x0[self.state_inds[pname, aname]]
            suc = self.solver.solve(plan, traj_mean=np.array(x0).reshape((1,-1)), active_ts=(0,0))
            for pname, aname in self.state_inds:
                if plan.params[pname].is_symbol(): continue
                x0[self.state_inds[pname, aname]] = getattr(plan.params[pname], aname)[:,0]
            assert len(plan.get_failed_preds(active_ts=(0,0), tol=1e-3)) == 0
            try:
                success = self.solver._backtrack_solve(plan)
            except Exception as e:
                print(('Failed in resample of hl', e))
                success = False
            if success:
                nsuc += 1
                path = self.run_plan(plan, targets=targets)
                print(('RESAMPLED!', nsuc))
            else:
                print(('Failed to resample', plan.get_failed_preds(), x0))
        print(('Generated', x0s, 'for', plan.actions, 'success:', nsuc))
        return x0s


    #def _failed_preds(self, Xs, task, condition, active_ts=None, debug=False, targets=[]):
    #    raise NotImplementedError


    def _failed_preds(self, Xs, task, condition, active_ts=None, debug=False, targets=[]):
        if len(np.shape(Xs)) == 1:
            Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
        plan = self.plans[task]
        if targets is None or not len(targets):
            targets = self.target_vecs[condition]
        for tname, attr in self.target_inds:
            getattr(plan.params[tname], attr)[:,0] = targets[self.target_inds[tname, attr]]
        tol = 1e-3
        for t in range(active_ts[0], active_ts[1]+1):
            set_params_attrs(plan.params, self.state_inds, Xs[t-active_ts[0]], t)
        self.set_symbols(plan, task, condition)

        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)

        if active_ts == None:
            active_ts = (1, plan.horizon-1)

        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        failed_preds = [p for p in failed_preds if not type(p[1].expr) is EqExpr]
        return failed_preds


    def cost_f(self, Xs, task, condition, active_ts=None, debug=False, targets=[]):
        task = tuple(task)
        if active_ts == None:
            plan = self.plans[task]
            active_ts = (1, plan.horizon-1)
        elif active_ts[0] == -1:
            plan = self.plans[task]
            active_ts = (plan.horizon-1, plan.horizon-1)
        failed_preds = self._failed_preds(Xs, task, condition, active_ts=active_ts, debug=debug, targets=targets)
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


    def plan_to_policy(self, plan=None, opt_traj=None):
        if opt_traj is None:
            opt_traj = plan_to_traj(plan, self.state_inds, self.dX)

        pol = self.optimal_pol_cls(self.dU, self.action_inds, self.state_inds, opt_traj)
        return pol


    def get_annotated_image(self, s, t):
        x = s.get_X(t=t)
        task = s.get(FACTOREDTASK_ENUM, t=t)
        u = s.get(ACTION_ENUM, t=t)
        textover = self.mjc_env.get_text_overlay(body='Task: {0}'.format(task))
        self.reset_to_state(x)
        im = self.mjc_env.render(camera_id=0, height=self.image_height, width=self.image_width, view=False, overlays=(textover,))
        return im


    def get_image(self, x, depth=False):
        self.reset_to_state(x)
        im = self.mjc_env.render(camera_id=0, height=self.image_height, width=self.image_width, view=False, depth=depth)
        return im

    
    def compare_tasks(self, t1, t2):
        return np.all(np.array(t1) == np.array(t2))

