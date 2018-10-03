import copy
import sys
import traceback

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine


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

class NAMOSortingAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        # Note: All plans should contain identical sets of parameters
        self.plans = self._hyperparams['plans']
        self.task_list = self._hyperparams['task_list']
        self.task_durations = self._hyperparams['task_durations']
        self.task_encoding = self._hyperparams['task_encoding']
        # self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self._samples = {task: [] for task in self.task_list}
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.num_cans = self._hyperparams['num_cans']
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
        self.targ_list = self.targets[0].keys()
        self.obj_list = self._hyperparams['obj_list']

        self._get_hl_plan = self._hyperparams['get_hl_plan']
        self.env = self._hyperparams['env']
        self.openrave_bodies = self._hyperparams['openrave_bodies']

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

        self.get_plan = self._hyperparams['get_plan']
        self.move_limit = 1e-3


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(batch)

        return samples
        

    def add_sample_batch(self, samples, task):
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
        print 'Cleared samples. Remaining per task:'
        for task in self.task_list:
            print '    ', task, ': ', len(self._samples[task]), ' standard; ', len(self.optimal_samples[task]), 'optimal'


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


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    def sample_task(self, policy, condition, x0, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True, fixed_obj=True):
        task = tuple(task)
        plan = self.plans[task[:2]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon
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
            if OBS_ENUM in self._hyperparams['obs_include']:
                sample.set(OBS_ENUM, im.copy(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
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
            sample.set(OBJ_POSE_ENUM, self.state_inds[task[1], 'pose'], t)
            sample.set(TARG_POSE_ENUM, self.targets[condition][task[2]].copy(), t)
            sample.task = task[0]
            sample.obj = task[1]
            sample.targ = task[2]
            sample.condition = condition

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U = policy.act(sample.get_X(t=t).copy(), obs.copy(), t, noise[t])
            for param_name, attr in self.action_inds:
                if (param_name, attr) in self.state_inds:
                    inds1 = self.action_inds[param_name, attr]
                    inds2 = self.state_inds[param_name, attr]
                    for i in range(len(inds1)):
                        if U[inds1[i]] - X[inds2[i]] > 1:
                            U[inds1[i]] = X[inds2[i]] + 1
                        elif U[inds1[i]] - X[inds2[i]] < -1:
                            U[inds1[i]] = X[inds2[i]] - 1
            if np.all(np.abs(U - self.traj_hist[-1]) < self.move_limit):
                sample.use_ts[t] = 0

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
            
            self.traj_hist.append(U)
            while len(self.traj_hist) > self.hist_len:
                self.traj_hist.pop(0)

            obj = task[1] if fixed_obj else None

            self.run_policy_step(U, X, self.plans[task[:2]], t, obj)
            if np.any(np.abs(U) > 1e10):
                import ipdb; ipdb.set_trace()

        return sample


    def resample(self, sample_lists, policy, num_samples):
        samples = []
        for slist in sample_lists:
            if hasattr(slist, '__len__') and not len(slist): continue
            samples.append([])
            for i in range(num_samples):
                s = slist[0] if hasattr(slist, '__getitem__') else slist
                self.reset_hist(s.get(TRAJ_HIST_ENUM, t=0).reshape((self.hist_len, 3)).tolist())
                samples[-1].append(self.sample_task(policy, s.condition, s.get_X(t=0), (s.task, s.obj, s.targ), noisy=True))
            samples[-1] = SampleList(samples[-1])
        return samples


    def run_policy_step(self, u, x, plan, t, obj):
        u_inds = self.action_inds
        x_inds = self.state_inds
        in_gripper = False

        if t < plan.horizon - 1:
            for param, attr in u_inds:
                getattr(plan.params[param], attr)[:, t+1] = u[u_inds[param, attr]]

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


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[], fixed_targets=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean, fixed_targets)

        exclude_targets = []

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

        class optimal_pol:
            def act(self, X, O, t, noise):
                return opt_traj[t].copy()

        sample = self.sample_task(optimal_pol(), condition, state, [task, targets[0].name, targets[1].name], noisy=False, fixed_obj=True)
        self.optimal_samples[task].append(sample)
        sample.set_ref_X(sample.get(STATE_ENUM))
        sample.set_ref_U(sample.get_U())
        return sample


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


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        return self._get_hl_plan(state, self.targets[condition], '{0}{1}'.format(condition, plan_id), self.plans.values()[0].params, self.state_inds, failed_preds)


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
