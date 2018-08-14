import copy

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine


from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.namo.sorting_prob import *


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
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.x0[condition] = np.concatenate([self.x0[condition], target_vec])

        self._get_hl_plan = self._hyperparams['get_hl_plan']
        self.env = self._hyperparams['env']
        self.openrave_bodies = self._hyperparams['openrave_bodies']

        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        self.hist_len = self._hyperparams['hist_len']
        self.traj_hist = None
        self._reset_hist()

        self.optimal_samples = {task: [] for task in self.task_list}
        self.optimal_state_traj = [[] for _ in range(len(self.plans))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans))]

        self.task_paths = []

        self.get_plan = self._hyperparams['get_plan']

        self.in_left_grip = -1
        self.in_right_grip = -1


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(SampleList(batch))

        return samples
        

    def add_sample_batch(self, samples, task):
        self._samples[task].append(samples)


    def clear_samples(self, keep_prob=0., keep_opt_prob=1.):
        for task in self.task_list:
            n_keep = int(keep_prob * len(self._samples[task]))
            self._samples[task] = random.sample(self._samples[task], n_keep)

            n_opt_keep = int(keep_opt_prob * len(self.optimal_samples[task]))
            self.optimal_samples[task] = random.sample(self.optimal_samples[task], n_opt_keep)


    def add_task_paths(self, paths):
        self.task_paths.extend(paths)


    def get_task_paths(self):
        return copy.copy(self.task_paths)


    def clear_task_paths(self, keep_prob=0.):
        n_keep = int(keep_prob * len(self.task_paths))
        self.task_paths = random.sample(self.task_paths, n_keep)


    def _reset_hist(self):
        self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist() if self.hist_len > 0 else None


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    def sample_task(self, policy, condition, x0, task, save_global=False, verbose=False, use_base_t=True, noisy=True):
        task = tuple(task)
        plan = self.plans[task]
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

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            X = np.zeros((plan.symbolic_bound))
            fill_vector(plan.params, plan.state_inds, X, t) 

            sample.set(STATE_ENUM, X.copy(), t)
            if OBS_ENUM in self._hyperparams['obs_include']:
                sample.set(OBS_ENUM, im.copy(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            # sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[self.task_list.index(task[0])] = 1.
            sample.set(TASK_ENUM, task_vec, t)
            sample.set(TARGETS_ENUM, target_vec.copy(), t)
            sample.task = task[0]
            sample.condition = condition

            U = policy.act(sample.get_X(t=t), sample.get_obs(t=t), t, noise[t])
            sample.set(ACTION_ENUM, U.copy(), t)
            
 
            if len(self.traj_hist) >= self.hist_len:
                self.traj_hist.pop(0)
            self.traj_hist.append(U)

            self.run_policy_step(U, X, self.plans[task], t)

        return sample

    def run_policy_step(self, u, x, plan, t):
        u_inds = self.action_inds
        x_inds = self.state_inds
        in_gripper = False

        if t < plan.horizon - 1:
            for param, attr in u_inds:
                getattr(plan.params[param], attr)[:, t+1] = u[u_inds[param, attr]]

            for param in plan.params.values():
                if param._type == 'Can':
                    dist = x[x_inds['pr2', 'pose']] - x[x_inds[param.name, 'pose']]
                    radius1 = param.geom.radius
                    radius2 = plan.params['pr2'].geom.radius
                    grip_dist = radius1 + radius2
                    if u[u_inds['pr2', 'gripper']][0] > 0.9 and dist[0]**2 < 0.01 and np.abs(grip_dist + dist[1]) < 0.01:
                        param.pose[:, t+1] = u[u_inds['pr2', 'pose']] + [0, grip_dist]
                    elif param._type == 'Can':
                        param.pose[:, t+1] = param.pose[:, t]
        return True


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def sample_optimal_trajectory(self, state, task, condition, targets=[]):
        targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition]) if not len(targets) else targets
        plan = self.plans[task, targets[0].name] 
        set_params_attrs(plan.params, plan.state_inds, state, 0)
        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        for target in targets:
            if target.name in self.targets[condition]:
                plan.params[target.name].value[:,0] = self.targets[condition][target.name]
        
        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        dist = plan.params['pr2'].geom.radius + targets[0].geom.radius
        plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]
        success = self.solver._backtrack_solve(plan, n_resamples=3)
        if not success:
            return None

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    fill_vector(plan.params, plan.action_inds, U, t+1)
                else:
                    fill_vector(plan.params, plan.action_inds, U, t)
                return U

        sample = self.sample_task(optimal_pol(), condition, state, [task, targets[0].name], noisy=False)
        self.optimal_samples[task].append(sample)
        return sample


    def get_hl_plan(self, condition):
        return self._get_hl_plan(self.init_vecs[condition], self.plans.values()[0].params, self.state_inds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets
