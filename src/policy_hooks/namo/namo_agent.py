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


    def clear_samples(self):
        self._samples = {task: [] for task in self.task_list}


    def _reset_hist(self):
        self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist() if self.hist_len > 0 else None


    def sample(self, policy_map, condition, save_global=False, verbose=False, save=True, use_base_t=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.current_cond = condition
        x0 = np.zeros((self.dX,))
        utils.fill_vector(self.params[condition], self.state_inds, x0, 0)                
        num_tasks = len(self.task_encoding.keys())
        cur_task_ind = 0
        next_t, task = self.task_breaks[condition][cur_task_ind]
        policy = policy_map[task]['policy']
        base_t = 0
        self.T = next_t
        sample = Sample(self)
        samples = [sample]
        sample.init_t = 0
        print 'Starting on-policy sample for condition {0}.'.format(condition)
        # if self.stochastic_conditions and save_global:
        #     self.replace_cond(condition)

        attempts = 0
        success = False
        while not success and attempts < 3:
            if noisy:
                noise = generate_noise(self.T, self.dU, self._hyperparams)
            else:
                noise = np.zeros((self.T, self.dU))

            for t in range(0, (self.plans[condition].horizon-1)):
                if t >= next_t:
                    if save:
                        self._samples[condition][task].append(sample)
                    cur_task_ind += 1
                    next_t, task = self.task_breaks[condition][cur_task_ind]
                    policy = policy_map[task]['policy']
                    self.T = next_t - t
                    sample = Sample(self)
                    samples.append(sample)
                    sample.init_t = t

                base_t = sample.init_t

                X = np.zeros(self.symbolic_bound)
                fill_vector(plan.params.values(), self.x_inds, x, t)

                obs = []
                if OBS_ENUM in self._hyperparams['obs_include']:
                    im = self.get_obs()
                    obs = np.r_[obs, im]
                
                if STATE_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, X]

                if TRAJ_HIST_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, np.array(self.traj_hist).flatten()]

                if use_base_t:
                    U = policy.act(X.copy(), obs, t-base_t, noise[t-base_t])
                else:
                    U = policy.act(X.copy(), obs, t, noise[t-base_t])


                for i in range(1):
                    sample.set(STATE_ENUM, X.copy(), t-base_t+i)
                    if OBS_ENUM in self._hyperparams['obs_include']:
                        sample.set(OBS_ENUM, im.copy(), t-base_t+i)
                    sample.set(ACTION_ENUM, U.copy(), t-base_t+i)
                    sample.set(NOISE_ENUM, noise[t-base_t], t-base_t+i)
                    sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t-base_t+i)
                    task_vec = np.zeros((num_tasks,))
                    task_vec[self.task_encoding[task]] = 1
                    sample.set(TASK_ENUM, task_vec, t-base_t+i)


                if len(self.traj_hist) >= self.hist_len: self.traj_hist.pop(0)
                self.traj_hist.append(U)

                success = self.run_policy_step(U, X, self.plans[condition], t)

            print 'Finished on-policy sample.\n'.format(condition)

        if save:
            self._samples[condition][task].append(sample)
        return samples


    def sample_task(self, policy, condition, x0, task, save_global=False, verbose=False, use_base_t=True, noisy=True):
        plan = self.plans[task]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(self.plans[task].params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = self.task_durations[task]
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

            obs = []
            if OBS_ENUM in self._hyperparams['obs_include']:
                im = self.get_obs()
                obs = np.r_[obs, im]
            
            if STATE_ENUM in self._hyperparams['obs_include']:
                obs = np.r_[obs, X]

            if TRAJ_HIST_ENUM in self._hyperparams['obs_include']:
                obs = np.r_[obs, np.array(self.traj_hist).flatten()]

            U = policy.act(X.copy(), obs, t, noise[t])

            for i in range(1):
                sample.set(STATE_ENUM, X.copy(), t+i)
                if OBS_ENUM in self._hyperparams['obs_include']:
                    sample.set(OBS_ENUM, im.copy(), t+i)
                sample.set(ACTION_ENUM, U.copy(), t+i)
                sample.set(NOISE_ENUM, noise[t], t+i)
                # sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t+i)
                task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
                task_vec[self.task_list.index(task)] = 1.
                sample.set(TASK_ENUM, task_vec, t+i)
                sample.set(TARGETS_ENUM, target_vec.copy(), t+i)
            
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
                if param._type == 'Can' and u[u_inds['pr2', 'gripper']] == 0 and np.sum((x[x_inds['pr2', 'pose']] - x[x_inds[param.name, 'pose']]))**2 <= 0.0001:
                    param.pose[:, t+1] = u[u_inds['pr2', 'pose']]
                elif param._type == 'Can':
                    param.pose[:, t+1] = param.pose[:, t]
        return True


    def init_cost_trajectories(self, alg_map, center=False, full_solve=True):
        for m in range(0, len(self.plans)):
            old_params_free = {}
            for p in self.params[m]:
                if p.is_symbol():
                    if p not in init_act.params: continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in old_params_free[p].keys():
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, 0].copy()
                        p._free_attrs[attr][:, 0] = 0

            self.current_cond = m
            if full_solve:
                success = self.solver._backtrack_solve(self.plans[m], n_resamples=3)
            else:
                success = True
                self.set_plan_from_cost_trajs(alg_map.values()[0], 0, self.plans[m].horizon, m)

            while not success:
                print "Solve failed."
                for p in self.params[m]:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        p._free_attrs = old_params_free[p]
                    else:
                        for attr in p._free_attrs:
                            p._free_attrs[attr][:, 0] = old_params_free[p][attr]
                self.replace_cond(m)

                old_params_free = {}
                for p in self.params[m]:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        old_params_free[p] = p._free_attrs
                        p._free_attrs = {}
                        for attr in old_params_free[p].keys():
                            p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                    else:
                        p_attrs = {}
                        old_params_free[p] = p_attrs
                        for attr in p._free_attrs:
                            p_attrs[attr] = p._free_attrs[attr][:, 0].copy()
                            p._free_attrs[attr][:, 0] = 0

                success = self.solver._backtrack_solve(self.plans[m], n_resamples=3)

            for p in self.params[m]:
                if p.is_symbol():
                    if p not in init_act.params: continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, 0] = old_params_free[p][attr]

            self.set_cost_trajectories(0, self.plans[m].horizon-1, m, alg_map.values(), center=center)
            for alg in alg_map.values():
                alg.task_breaks = self.task_breaks

        self.initial_opt = False


    def set_plan_from_cost_trajs(self, alg, init_t, final_t, m):
        tgt_x = alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state']
        if utils.POLICY_STEPS_PER_SECOND < 1:
            for t in range(0, final_t-init_t, (1/utils.POLICY_STEPS_PER_SECOND)):
                utils.set_params_attrs(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
        else:
            for t in range(0, final_t-init_t):
                utils.set_params_attrs(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)


    def set_cost_trajectories(self, init_t, final_t, m, algs, center=False):
        tgt_x = np.zeros((final_t-init_t, self.symbolic_bound))
        tgt_u = np.zeros((final_t-init_t, self.dU))

        for t in range(0, final_t-init_t, int(1/utils.POLICY_STEPS_PER_SECOND)):
            utils.fill_vector(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
            tgt_u[t, self.action_inds['pr2', 'pose']] = self.plans[m].params['pr2'].pose[:, init_t+t+1]
            tgt_u[t, self.action_inds['pr2', 'gripper']] = self.plans[m].params['pr2'].gripper[:, init_t+t1]

            utils.fill_vector(self.params[m], self.state_inds, tgt_x[t], t+init_t)
            
        self.optimal_act_traj[m] = tgt_u
        self.optimal_state_traj[m] = tgt_x

        for alg in algs:
            alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state'] = tgt_x.copy()
            alg.cost[m]._costs[1]._hyperparams['data_types'][utils.ACTION_ENUM]['target_state'] = tgt_u.copy()

        if center:
            for alg in algs:
                for ts in alg.cur[m]:
                    alg.cur[m][ts].traj_distr.k = self.optimal_act_traj[m][ts:ts+alg.T]


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def sample_optimal_trajectory(self, state, task, condition, targets=[]):
        targets = get_next_target(self.plans[task], self.init_vecs[condition], task) if not len(targets) else targets
        plan = get_plan_for_task(task, [target.name for target in targets], self.num_cans, self.env, self.openrave_bodies)
        self.set_nonopt_attrs(plan, task)
        # set_params_attrs(plan.params, plan.state_inds, self.init_vecs[condition], 0, True)
        set_params_attrs(plan.params, plan.state_inds, state, 0)
        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        for target in targets:
            if target._type == 'Target':
                print target, self.targets[condition][target.name]
                target.value[:,0] = self.targets[condition][target.name]
        success = self.solver._backtrack_solve(plan, n_resamples=3)
        
        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU))
                fill_vector(plan.params, plan.action_inds, U, t)
                return U

        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)
        self.optimal_samples[task].append(sample)
        return sample


    def get_hl_plan(self, condition):
        return self._get_hl_plan(self.init_vecs[condition], self.plans.values()[0].params, self.state_inds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets

