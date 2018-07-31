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
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

from policy_hooks.can_world_policy_utils import *
from policy_hooks.policy_solver_utils import STATE_ENUM, OBS_ENUM, ACTION_ENUM, NOISE_ENUM, EE_ENUM, GRIPPER_ENUM, COLORS_ENUM, TRAJ_HIST_ENUM, TASK_ENUM
import policy_hooks.policy_solver_utils as utils


class NAMOSortingAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        # Note: All plans should contain identical sets of parameters
        self.plans = self._hyperparams['plans']
        self.task_breaks = self._hyperparams['task_breaks']
        self.task_encoding = self._hyperparams['task_encoding']
        self.task_durations = self._hyperparams['task_durations']
        self.color_maps = self._hyperparams['color_maps']
        self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.num_cans = self._hyperparams['num_cans']
        self.x0 = self._hyperparams['x0']
        self.sim = 'mujoco'
        self.viewers = self._hyperparams['viewers']

        self.symbols = [filter(lambda p: p.is_symbol(), self.plans[m].params.values()) for m in range(len(self.plans))]
        self.params = [filter(lambda p: not p.is_symbol(), self.plans[m].params.values()) for m in range(len(self.plans))]
        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        self.hist_len = self._hyperparams['hist_len']
        self.traj_hist = None
        self._reset_hist()

        self.optimal_state_traj = [[] for _ in range(len(self.plans))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans))]

        self.get_plan = self._hyperparams['get_plan']

        self.in_left_grip = -1
        self.in_right_grip = -1


    def get_samples(self, condition, task, start=0, end=None):
        if np.abs(start) >= len(self._samples[condition][task]):
            start = 0

        samples = {}
        if end is None:
            for sample in self._samples[condition][task][start:]:
                if sample.init_t not in samples:
                    samples[sample.init_t] = []
                samples[sample.init_t].append(sample)
        else:
            for sample in self._samples[condition][task][start:end]:
                if sample.init_t not in samples:
                    samples[sample.init_t] = []
                samples[sample.init_t].append(sample)

        for ts in samples:
            samples[ts] = SampleList(samples[ts])

        return samples

        
    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples[condition] = {task:[] for task in self.task_encoding.keys()}

        return X


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

        color_vec = np.zeros((len(self.color_maps[condition].keys())))
        for can_name in self.color_maps[condition]:
            color_vec[int(can_name[-1])] = self.color_maps[condition][can_name][0] * 100

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
                    sample.set(COLORS_ENUM, color_vec.copy(), t-base_t+i)
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


    def sample_task(self, policy, condition, x0, task, save_global=False, verbose=False, save=True, use_base_t=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(self.plans[condition].params[param], attr)[:,1] = x0[self.state_inds[param, attr]]
        num_tasks = len(self.task_encoding.keys())
        cur_task_ind = 0
        self.T = self.task_durations[task]
        base_t = 0
        sample = Sample(self)
        sample.init_t = 0
        print 'Starting on-policy sample for condition {0}.'.format(condition)
        # if self.stochastic_conditions and save_global:
        #     self.replace_cond(condition)

        color_vec = np.zeros((len(self.color_maps[condition].keys())))
        for can_name in self.color_maps[condition]:
            color_vec[int(can_name[-1])] = self.color_maps[condition][can_name][0] * 100

        attempts = 0
        success = False
        while not success and attempts < 3:
            self._set_simulator_state(condition, self.plans[condition], 1)

            if noisy:
                noise = generate_noise(self.T, self.dU, self._hyperparams)
            else:
                noise = np.zeros((self.T, self.dU))


            for t in range(0, self.T*utils.POLICY_STEPS_PER_SECOND):
                base_t = sample.init_t

                X, joints = self._get_simulator_state(self.state_inds, condition, self.symbolic_bound)

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
                    sample.set(COLORS_ENUM, color_vec.copy(), t-base_t+i)
                    # sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t-base_t+i)
                    task_vec = np.zeros((num_tasks,))
                    task_vec[self.task_encoding[task]] = 1
                    sample.set(TASK_ENUM, task_vec, t-base_t+i)

                
                if len(self.traj_hist) >= self.hist_len: self.traj_hist.pop(0)
                self.traj_hist.append(U)

                self.run_policy_step(U)
            print 'Finished on-policy sample.\n'.format(condition)

        if save:
            self._samples[condition][task].append(sample)
        return sample

    def run_policy_step(self, u, x, plan, t):
        u_inds = self.action_inds
        in_gripper = False

        for param, attr in u_inds:
            getattr(plan.params[param], attr)[:, t+1] = u[param, attr]

        for param in plan.params:
            if param._type == 'Can' and u['pr2', 'gripper'] == 0 and (x['pr2', 'pose'] - x[param.name, 'pose'])**2 <= 0.01:
                param.pose[:, t+1] = u['pr2', 'pose']

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
            tgt_u[t, self.action_inds['pr2', 'gripper']] = self.plans[m[.params['pr2'].gripper[:, init_t+t1]

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


    def sample_optimal_trajectories(self, n=1):
        class optimal_pol:
            def __init__(self, act_f):
                self.act = act_f

        def get_policy_map(m):
            policy_map = {}
            for task in self.task_encoding.keys():
                policy_map[task] = {}
                policy_map[task]['policy'] = optimal_pol(lambda X, O, t, noise: self.optimal_act_traj[m][t].copy())

            return policy_map

        samples = []

        for m in range(len(self.plans)):
            samples.append([])
            for _ in range(n):
                samples[-1].append(self.sample(get_policy_map(m), m, save=True, use_base_t=False, noisy=False))

        return samples

    def replace_cond(self, cond):
        print "Replacing Condition {0}.\n".format(cond)
        plan, task_breaks, color_map, goal_state = self.get_plan(self.num_cans)
        self.plans[cond].env.Destroy()
        self.plans[cond] = plan
        self.params[cond] = filter(lambda p: not p.is_symbol(), plan.params.values())
        self.symbols[cond] = filter(lambda p: p.is_symbol(), plan.params.values())
        self.task_breaks[cond] = task_breaks
        self.color_maps[cond] = color_map
        x = np.zeros((self.symbolic_bound,))
        utils.fill_vector(self.params[cond], self.state_inds, x, 0)                
        self.x0[cond] = x


    def replace_all_conds(self):
        for cond in range(len(self.plans)):
            self.replace_cond(cond)

