from abc import ABCMeta, abstractmethod
import copy
import random
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


MAX_SAMPLELISTS = 100
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


class TAMPAgent(Agent):
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        # Note: All plans should contain identical sets of parameters
        self.prob = hyperparams['prob']
        self.plans = self._hyperparams['plans']
        self.task_list = self._hyperparams['task_list']
        self.task_durations = self._hyperparams['task_durations']
        self.task_encoding = self._hyperparams['task_encoding']
        # self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self._samples = {task: [] for task in self.task_list}
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        # self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.num_objs = self._hyperparams['num_objs']
        self.init_vecs = self._hyperparams['x0']
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.targets = self._hyperparams['targets']
        self.target_dim = self._hyperparams['target_dim']
        self.target_inds = self._hyperparams['target_inds']
        self.target_vecs = []
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)
        # self.targ_list = self.targets[0].keys()
        # self.obj_list = self._hyperparams['obj_list']

        self._get_hl_plan = self._hyperparams['get_hl_plan']
        self.attr_map = self._hyperparams['attr_map']
        self.env = self._hyperparams['env']
        self.openrave_bodies = self._hyperparams['openrave_bodies']
        if self._hyperparams['viewer']:
            self.viewer = OpenRAVEViewer(self.env)
        else:
            self.viewer = None

        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        self.hist_len = self._hyperparams['hist_len']
        self.traj_hist = None
        self.reset_hist()

        self.optimal_samples = {task: [] for task in self.task_list}
        self.optimal_state_traj = [[] for _ in range(len(self.plans))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans))]

        self.task_paths = []

        # self.get_plan = self._hyperparams['get_plan']
        self.move_limit = 1e-3

        self.n_policy_calls = {}
        self._cc = openravepy.RaveCreateCollisionChecker(self.env, "ode")
        self._cc.SetCollisionOptions(openravepy.CollisionOptions.Contacts)
        self.env.SetCollisionChecker(self._cc)
        # self._cc = ctrajoptpy.GetCollisionChecker(self.env)
        self.n_dirs = self._hyperparams['n_dirs']
        self.seed = 1234
        self.prim_dims = self._hyperparams['prim_dims']

        self.solver = self._hyperparams['solver_type'](self._hyperparams)


    def get_init_state(self, condition):
        return self.x0[condition][self._x_data_idx[STATE_ENUM]].copy()


    def add_viewer(self):
        self.viewer = OpenRAVEViewer(self.env)


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(batch)

        return samples
        

    def add_sample_batch(self, samples, task):
        if type(task) is tuple:
            task = self.task_list[task[0]]
        if not hasattr(samples[0], '__getitem__'):
            if not isinstance(samples, SampleList):
                samples = SampleList(samples)
            self._samples[task].append(samples)
            # print 'Stored {0} samples for'.format(len(samples)), task
        else:
            for batch in samples:
                if not isinstance(batch, SampleList):
                    batch = SampleList(batch)
                self._samples[task].append(batch)
                # print 'Stored {0} samples for'.format(len(samples)), task
        while len(self._samples[task]) > MAX_SAMPLELISTS:
            del self._samples[task][0]


    def clear_samples(self, keep_prob=0., keep_opt_prob=1.):
        for task in self.task_list:
            n_keep = int(keep_prob * len(self._samples[task]))
            self._samples[task] = random.sample(self._samples[task], n_keep)

            n_opt_keep = int(keep_opt_prob * len(self.optimal_samples[task]))
            self.optimal_samples[task] = random.sample(self.optimal_samples[task], n_opt_keep)
        # print 'Cleared samples. Remaining per task:'
        # for task in self.task_list:
        #     print '    ', task, ': ', len(self._samples[task]), ' standard; ', len(self.optimal_samples[task]), 'optimal'


    def reset_sample_refs(self):
        for task in self.task_list:
            for batch in self._samples[task]:
                for sample in batch:
                    sample.set_ref_X(np.zeros((sample.T, self.symbolic_bound)))
                    sample.set_ref_U(np.zeros((sample.T, self.dU)))


    def add_task_paths(self, paths):
        self.task_paths.extend(paths)
        while len(self.task_paths) > MAX_TASK_PATHS:
            del self.task_paths[0]


    def get_task_paths(self):
        return copy.copy(self.task_paths)


    def clear_task_paths(self, keep_prob=0.):
        n_keep = int(keep_prob * len(self.task_paths))
        self.task_paths = random.sample(self.task_paths, n_keep)


    def reset_hist(self, hist=[]):
        if not len(hist):
            hist = np.zeros((self.hist_len, self.dU)).tolist()
        self.traj_hist = hist


    def get_hist(self):
        return copy.deepcopy(self.traj_hist)


    def animate_sample(self, sample):
        if self.viewer is None: return
        plan = self.plans.values()[0]
        for p in plan.params.values():
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
        plan = self.plans.values()[0]
        for p in plan.params.values():
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
        for p in plan.params.itervalues():
            p_attrs = {}
            old_params_free[p.name] = p_attrs
            for attr in p._free_attrs:
                p_attrs[attr] = p._free_attrs[attr].copy()
        self.saved_params_free = old_params_free


    def restore_free(self, plan):
        for p in self.saved_params_free:
            for attr in self.saved_params_free[p]:
                plan.params[p]._free_attrs[attr] = self.saved_params_free[p][attr].copy()


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    @abstractmethod
    def sample_task(self, policy, condition, x0, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True):
        raise NotImplementedError


    def resample(self, sample_lists, policy, num_samples):
        samples = []
        for slist in sample_lists:
            if hasattr(slist, '__len__') and not len(slist): continue
            samples.append([])
            for i in range(num_samples):
                s = slist[0] if hasattr(slist, '__getitem__') else slist
                # self.reset_hist(s.get(TRAJ_HIST_ENUM, t=0).reshape((self.hist_len, 3)).tolist())
                samples[-1].append(self.sample_task(policy, s.condition, s.get(STATE_ENUM, t=0), (s.task, s.obj, s.targ), noisy=True))
            samples[-1] = SampleList(samples[-1])
        return samples


    # @abstractmethod
    # def dist_obs(self, plan, t):
    #     raise NotImplementedError


    # @abstractmethod
    # def run_policy_step(self, u, x, plan, t, obj):
    #     raise NotImplementedError


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
            obj = self.plans.values()[0].params[self.obj_list[np.argmax(obj_distr)]]
            targ = self.plans.values()[0].params[self.targ_list[np.argmax(targ_distr)]]
            targets = [obj, targ]
            # targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition], sample_traj=traj_mean)

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
        sample.set_ref_U(sample.get(ACTION_ENUM))
        return sample, [], True


    def perturb_solve(self, sample, perturb_var=0.02):
        x0 = sample.get(STATE_ENUM, t=0)
        saved_targets = {}
        cond = sample.condition
        saved_target_vec = self.target_vecs[cond].copy()
        for obj in self.obj_list:
            obj_p = self.plans.values()[0].params[obj]
            if obj_p._type == 'Robot': continue
            x0[self.state_inds[obj, 'pose']] += np.random.normal(0, perturb_var, obj_p.pose.shape[0])
        old_targets = sample.get(TARGETS_ENUM, t=0)
        for target_name in self.targets[cond]:
            old_value = old_targets[self.target_inds[target_name, 'value']]
            saved_targets[target_name] = old_value
            self.targets[cond][target_name] = old_value + np.random.normal(0, perturb_var, old_value.shape)
            self.target_vecs[cond][self.target_inds[target_name, 'value']] = self.targets[cond][target_name]
        target_params = [self.plans.values()[0].params[sample.obj], self.plans.values()[0].params[sample.targ]]
        out = self.solve_sample_opt_traj(x0, sample.task, sample.condition, sample.get_U(), target_params)
        for target_name in self.targets[cond]:
            self.targets[cond][target_name] = saved_targets[target_name]
        self.target_vecs[cond] = saved_target_vec
        return out


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        return self._get_hl_plan(state, self.targets[condition], '{0}{1}'.format(condition, plan_id), self.plans.values()[0].params, self.state_inds, failed_preds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets


    # @abstractmethod
    # def get_sample_constr_cost(self, sample):
    #     raise NotImplementedError


    def randomize_init_state(self, condition=0):
        self.targets[condition] = self.prob.get_end_targets(self.num_objs)
        self.init_vecs[condition] = self.prob.get_random_initial_state_vec(self.num_objs, self.targets, self.dX, self.state_inds, 1)[0]
        self.x0[condition] = self.init_vecs[condition][:self.symbolic_bound]
        target_vec = np.zeros((self.target_dim,))
        for target_name in self.targets[condition]:
            target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
        self.target_vecs[condition] = target_vec


    def replace_conditions(self, conditions, keep=(0.2, 0.5)):
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


    def perturb_conditions(self, perturb_var=0.02):
        self.perturb_init_states(perturb_var)
        self.perturb_targets(perturb_var)


    def perturb_init_states(self, perturb_var=0.02):
        for c in range(len(self.x0)):
            x0 = self.x0[c]
            for obj in self.obj_list:
                obj_p = self.plans.values()[0].params[obj]
                if obj.is_symbol() or obj._type == 'Robot': continue
                x0[self.state_inds[obj, 'pose']] += np.random.normal(0, perturb_var, obj_p.pose.shape[0])


    def perturb_targets(self, perturb_var=0.02):
        for c in range(len(self.x0)):
            for target_name in self.targets[c]:
                target_p = self.plans.values()[0].params[target_name]
                self.targets[c][target_name] += np.random.normal(0, perturb_var, target_p.value.shape[0])
                self.target_vecs[c][self.target_inds[target_name, 'value']] = self.targets[c][target_name]


    def get_prim_options(self, cond, state):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = self.prob.get_prim_choices()
        plan = self.plans.values()[0]
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
        options = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
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
        prim_options = self.prob.get_prim_choices()
        return prim_options[enum].index(name)


    def get_prim_indices(self, names):
        task = [self.task_list.index(names[0])]
        for i in range(1, len(names)):
            task.append(self.get_prim_index(self.prim_dims.keys()[i-1], names[i]))
        return tuple(task)
