import numpy as np
import tensorflow as tf

import logging
import imp
import os
import os.path
import sys
import copy
import cPickle as pickle
import argparse
from datetime import datetime
import threading
import pprint
import time
import traceback

from multiprocessing import Manager, Process

from gps.sample.sample_list import SampleList
from gps.algorithm.policy_opt.tf_model_example import tf_network

from gps.algorithm.cost.cost_sum import CostSum

from gps.algorithm.cost.cost_utils import *

from core.util_classes.viewer import OpenRAVEViewer
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.task_net import tf_binary_network, tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver
from policy_hooks.state_mcts import StateMCTS
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.namo.namo_policy_solver import NAMOPolicySolver


def solve_condition(trainer, cond, paths=[], all_samples=[], all_successful_samples=[]):
    failed = []
    new_failed = []
    stop = False
    attempt = 0
    cur_sample = None
    cur_state = trainer.agent.x0[cond].copy()
    opt_hl_plan = []
    cur_path = []

    try:
        hl_plan = trainer.agent.get_hl_plan(cur_state, cond, failed)
    except:
        hl_plan = []

    last_reset = 0
    while not stop and attempt < 6 * trainer.config['num_objs']:
        last_reset += 1
        for step in hl_plan:
            print 'Current hl plan for '+str(cond)+ ': ', opt_hl_plan
            targets = [trainer.agent.plans.values()[0].params[p_name] for p_name in step[1]]
            plan = trainer.config['plan_f'](step[0], targets)
            if len(targets) < 2:
                targets.append(plan.params['{0}_end_target'.format(targets[0].name)])
            next_sample, new_failed, success = trainer.agent.sample_optimal_trajectory(cur_state, step[0], cond, fixed_targets=targets)
            all_samples.append(next_sample)
            next_sample.success = FAIL_LABEL
            if not success:
                if last_reset > 5:
                    failed = []
                    last_reset = 0
                next_sample.task_cost = trainer.fail_value
                next_sample.success = FAIL_LABEL
                if not len(new_failed):
                    stop = True
                    if not len(hl_plan):
                        print 'NAMO COULD NOT FIND HL PLAN FOR {0}'.format(cur_state)
                else:
                    failed.extend(new_failed)
                    try:
                        hl_plan = trainer.agent.get_hl_plan(cur_state, cond, failed)
                        if not len(hl_plan):
                            print '\nCould not solve hl for condition '+ str(cond)
                    except:
                        hl_plan = []
                    # attempt += 1
                break

            cur_path.append(next_sample)
            all_successful_samples.append(next_sample)
            cur_sample = next_sample
            cur_state = cur_sample.end_state.copy()
            opt_hl_plan.append(step)

        if trainer.config['goal_f'](cur_state, trainer.agent.targets[cond], trainer.agent.plans.values()[0]) == 0:
            for sample in cur_path:
                sample.success = SUCCESS_LABEL
            break

        attempt += 1

    paths.append(cur_path)
    print 'Final hl plan for '+str(cond)+ ': ', opt_hl_plan
    return [cur_path, all_samples, all_successful_samples]


def resolve(sample, all_samples, trainer, n=1):
    new_samples = []
    for _ in range(n):
        new_sample, _, success = trainer.agent.perturb_solve(sample)
        if success: all_samples.append(new_sample)
        new_samples.append(new_sample)
    return new_samples


class MultiProcessPretrainMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self.config = config
        prob = config['prob']

        conditions = self.config['num_conds']
        self.task_list = tuple(get_tasks(self.config['task_map_file']).keys())
        self.task_durations = get_task_durations(self.config['task_map_file'])
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)

        plans = {}
        task_breaks = []
        goal_states = []
        targets = []

        env = None
        openrave_bodies = {}
        obj_type = self.config['obj_type']
        num_objs = self.config['num_objs']
        for task in self.task_list:
            for c in range(num_objs):
                plans[task, '{0}{1}'.format(obj_type, c)] = prob.get_plan_for_task(task, ['{0}{1}'.format(obj_type, c), '{0}{1}_end_target'.format(obj_type, c)], num_objs, env, openrave_bodies)
                if env is None:
                    env = plans[task, '{0}{1}'.format(obj_type, c)].env
                    for param in plans[task, '{0}{1}'.format(obj_type, c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body

        state_vector_include, action_vector_include, target_vector_include = self.config['get_vector'](num_objs)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], self.config['robot_name'], self.config['attr_map'], state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], self.config['attr_map'], target_vector_include)

        for i in range(conditions):
            targets.append(prob.get_end_targets(num_objs))
        
        x0 = prob.get_random_initial_state_vec(num_objs, targets, self.dX, self.state_inds, conditions)
        obj_list = ['{0}{1}'.format(obj_type, c) for c in range(num_objs)]

        for plan in plans.values():
            plan.state_inds = self.state_inds
            plan.action_inds = self.action_inds
            plan.dX = self.dX
            plan.dU = self.dU
            plan.symbolic_bound = self.symbolic_bound
            plan.target_dim = self.target_dim
            plan.target_inds = self.target_inds

        sensor_dims = {
            utils.STATE_ENUM: self.symbolic_bound,
            utils.ACTION_ENUM: self.dU,
            utils.TRAJ_HIST_ENUM: self.dU*self.config['hist_len'],
            utils.TASK_ENUM: len(self.task_list),
            utils.TARGETS_ENUM: self.target_dim,
            utils.OBJ_ENUM: num_objs,
            utils.TARG_ENUM: len(targets[0].keys()),
            utils.OBJ_POSE_ENUM: 2,
            utils.TARG_POSE_ENUM: 2,
            utils.LIDAR_ENUM: self.config['n_dirs'],
            utils.EE_ENUM: 2,
        }

        self.config['plan_f'] = lambda task, targets: plans[task, targets[0].name] 
        self.config['goal_f'] = prob.goal_f
        self.config['cost_f'] = prob.cost_f
        self.config['target_f'] = prob.get_next_target
        self.config['encode_f'] = prob.sorting_state_encode
        # self.config['weight_file'] = 'tf_saved/2018-09-12 23:43:45.748906_namo_5.ckpt'

        self.config['task_durations'] = self.task_durations

        self.policy_inf_coeff = self.config['algorithm']['policy_inf_coeff']
        self.policy_out_coeff = self.config['algorithm']['policy_out_coeff']
        self.config['agent'] = {
            'type': self.config['agent_type'],
            'x0': x0,
            'targets': targets,
            'task_list': self.task_list,
            'plans': plans,
            'task_breaks': task_breaks,
            'task_encoding': task_encoding,
            'task_durations': self.task_durations,
            'state_inds': self.state_inds,
            'action_inds': self.action_inds,
            'target_inds': self.target_inds,
            'dU': self.dU,
            'dX': self.symbolic_bound,
            'symbolic_bound': self.symbolic_bound,
            'target_dim': self.target_dim,
            'get_plan': prob.get_plan,
            'sensor_dims': sensor_dims,
            'state_include': self.config['state_include'],
            'obs_include': self.config['obs_include'],
            'prim_obs_include': self.config['prim_obs_include'],
            'val_obs_include': self.config['val_obs_include'],
            'conditions': self.config['num_conds'],
            'solver': None,
            'num_cans': num_objs,
            'obj_list': obj_list,
            'stochastic_conditions': False,
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'hist_len': self.config['hist_len'],
            'T': 1,
            'viewer': config['viewer'],
            'model': None,
            'get_hl_plan': prob.hl_plan_for_state,
            'env': env,
            'openrave_bodies': openrave_bodies,
            'n_dirs': self.config['n_dirs'],
            'prob': prob,
            'attr_map': self.config['attr_map'],
        }

        self.config['algorithm']['dObj'] = sensor_dims[utils.OBJ_ENUM]
        self.config['algorithm']['dTarg'] = sensor_dims[utils.TARG_ENUM]

        # action_cost_wp = np.ones((self.config['agent']['T'], self.dU), dtype='float64')
        state_cost_wp = np.ones((self.symbolic_bound), dtype='float64')
        traj_cost = {
                        'type': StateTrajCost,
                        'data_types': {
                            utils.STATE_ENUM: {
                                'wp': state_cost_wp,
                                'target_state': np.zeros((1, self.symbolic_bound)),
                                'wp_final_multiplier': 1.0,
                            }
                        },
                        'ramp_option': RAMP_CONSTANT
                    }
        action_cost = {
                        'type': ActionTrajCost,
                        'data_types': {
                            utils.ACTION_ENUM: {
                                'wp': np.ones((1, self.dU), dtype='float64'),
                                'target_state': np.zeros((1, self.dU)),
                            }
                        },
                        'ramp_option': RAMP_CONSTANT
                     }

        # constr_cost = {
        #                 'type': TrajConstrCost,
        #               }

        self.config['algorithm']['cost'] = {
                                                'type': CostSum,
                                                'costs': [traj_cost, action_cost],
                                                'weights': [1.0, 1.0],
                                           }

        # self.config['algorithm']['cost'] = constr_cost

        self.config['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dt'] = 1.0

        self.config['algorithm']['policy_opt'] = {
            'type': MultiHeadPolicyOptTf,
            'network_params': {
                'obs_include': self.config['agent']['obs_include'],
                'prim_obs_include': self.config['agent']['prim_obs_include'],
                'val_obs_include': self.config['agent']['val_obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': self.config['n_layers'],
                'num_filters': [5,10],
                'dim_hidden': self.config['dim_hidden'],
            },
            'distilled_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 3,
                'num_filters': [5,10],
                'dim_hidden': [100, 100, 100]
            },
            'primitive_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 2,
                'num_filters': [5,10],
                'dim_hidden': [40, 40],
                'output_boundaries': [len(self.task_list),
                                      len(obj_list),
                                      len(targets[0].keys())],
                'output_order': ['task', 'obj', 'targ'],
            },
            'value_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 1,
                'num_filters': [5,10],
                'dim_hidden': [40]
            },
            'lr': self.config['lr'],
            'network_model': tf_network,
            'distilled_network_model': tf_network,
            'primitive_network_model': tf_classification_network,
            'value_network_model': tf_binary_network,
            'iterations': self.config['train_iterations'],
            'batch_size': self.config['batch_size'],
            'weight_decay': self.config['weight_decay'],
            'weights_file_prefix': 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list,
            'gpu_fraction': 0.4,
            'update_size': self.config['update_size'],
        }

        alg_map = {}
        for task in self.task_list:
            self.config['algorithm']['T'] = self.task_durations[task]
            alg_map[task] = self.config['algorithm']
        self.config['policy_opt'] = self.config['algorithm']['policy_opt']
        self.config['algorithm'] = alg_map

        self.agent = config['agent']['type'](config['agent'])
        self.config['dX'] = self.dX
        self.config['dU'] = self.dU
        self.config['symbolic_bound'] = self.symbolic_bound
        self.config['dO'] = self.agent.dO
        self.config['dPrimObs'] = self.agent.dPrim
        self.config['dValObs'] = self.agent.dVal
        self.config['dObj'] = self.config['algorithm'].values()[0]['dObj'] 
        self.config['dTarg'] = self.config['algorithm'].values()[0]['dTarg'] 
        self.config['state_inds'] = self.state_inds
        self.config['action_inds'] = self.action_inds
        self.config['policy_out_coeff'] = self.policy_out_coeff
        self.config['policy_inf_coeff'] = self.policy_inf_coeff
        self.config['target_inds'] = self.target_inds
        self.config['target_dim'] = self.target_dim
        self.config['task_list'] = self.task_list
        self.solver = NAMOPolicySolver(self.config)
        self.agent.solver = self.solver

        self._train_idx = range(config['num_conds'])
        self.fail_value = config['fail_value']
        self.alg_map = {}
        policy_opt = None
        self.task_durations = config['task_durations']
        for task in self.task_list:
            if task not in self.task_durations: continue
            config['algorithm'][task]['policy_opt']['prev'] = policy_opt
            config['algorithm'][task]['policy_opt']['weight_dir'] = self.config['weight_dir']
            config['algorithm'][task]['agent'] = self.agent
            config['algorithm'][task]['init_traj_distr']['T'] = self.task_durations[task]
            self.config['algorithm'][task]['task'] = task
            self.alg_map[task] = self.config['algorithm'][task]['type'](self.config['algorithm'][task])
            policy_opt = self.alg_map[task].policy_opt
            self.alg_map[task].set_conditions(len(self.agent.x0))
            self.alg_map[task].agent = self.agent

        self.policy_opt = policy_opt
        self.traj_opt_steps = self.config['traj_opt_steps']
        self.num_samples = self.config['num_samples']

        self.rollout_policies = {task: self.policy_opt.task_map[task]['policy'] for task in self.task_list}
        self.time_log = 'tf_saved/'+self.config['weight_dir']+'/pretrain_timing_info.txt'
        self.hl_timeout = self.config['hl_timeout']


    def run(self, itr_load=None):
        self.check_dirs()
        self.pretrain()
        self.policy_opt.sess.close()


    def check_dirs(self):
        if not os.path.exists('tf_saved/'+self.config['weight_dir']):
            os.makedirs('tf_saved/'+self.config['weight_dir'])
        if not os.path.exists('tf_saved/'+self.config['weight_dir']+'_trained'):
            os.makedirs('tf_saved/'+self.config['weight_dir']+'_trained')


    def pretrain(self):
        manager = Manager()
        sample_paths = manager.list()
        all_samples = manager.list()
        all_successful_samples = manager.list()
        cpu_times = []

        if self.config['log_timing']:
            with open(self.time_log, 'a+') as f:
                f.write('\n\nPretraining data for {0}:\n'.format(datetime.now()))

        for pretrain_step in range(self.config['pretrain_steps']):
            print '\n\nPretrain step {0}\n\n'.format(pretrain_step)
            self.agent.replace_conditions(len(self.agent.x0), keep=(1., 1.))
            start_len = len(all_successful_samples)

            start_time = time.time()
            processes = []
            solve_condition(self, 0, sample_paths, all_samples, all_successful_samples)
            import ipdb; ipdb.set_trace()
            for cond in self._train_idx:
                process = Process(target=solve_condition, args=(self, cond, sample_paths, all_samples, all_successful_samples))
                process.daemon = True
                process.start()
                processes.append(process)

            base_t = time.time()
            while time.time() - base_t < self.hl_timeout:
                if any(p.is_alive() for p in processes):
                    time.sleep(0.1)
                else:
                    break
            else:
                print '\n\nTerminating pretrain step early.'
                print 'Active processes: {0}\n\n'.format([p.pid for p in processes if p.is_alive()])
                for p in processes:
                    p.terminate()
                    p.join()

            end_len = len(all_successful_samples)
            import ipdb; ipdb.set_trace()

            # Resolve with perturbations
            iters = max(min((end_len-start_len) / 16, 5), 1)
            n_perturbs = 5
            for _ in range(iters):
                processes = []
                n_samples = min(len(all_successful_samples[start_len:end_len]), 16)
                to_resolve = np.random.choice(all_successful_samples[start_len:], n_samples, replace=False)
                for sample in to_resolve:
                    process = Process(target=resolve, args=(sample, all_successful_samples, self, n_perturbs))
                    process.daemon = True
                    process.start()
                    processes.append(process)

                base_t = time.time()
                while time.time() - base_t < self.hl_timeout:
                    if any(p.is_alive() for p in processes):
                        time.sleep(0.1)
                    else:
                        break
                else:
                    print '\n\nTerminating perturb step early.'
                    print 'Active processes: {0}\n\n'.format([p.pid for p in processes if p.is_alive()])
                    for p in processes:
                        p.terminate()
                        p.join()

                end_time = time.time()
                cpu_times.append(end_time-start_time)

        if self.config['log_timing']:
            with open(self.time_log, 'a') as f:
                f.write('Average time to solve and motion plan for {0} hl problems: '.format(len(self.agent.x0)))
                f.write(str(np.average(cpu_times)))
                f.write('\n')

        print 'Collected pretraining data. Moving to supervised learning step.'
        task_to_samples = {task: [] for task in self.task_list}       
        opt_samples = {task: [] for task in self.task_list}
        for sample in all_successful_samples:
            sample.agent = self.agent
            opt_samples[sample.task].append((sample, []))

        cpu_times = []
        traj_opt_steps = self.config['pretrain_traj_opt_steps']
        for task in self.alg_map:
            for i in range(traj_opt_steps):
                start_time = time.time()
                print '\nIterating on initial samples (iter {0})'.format(i)
                policy = self.rollout_policies[task]
                if policy.scale is None:
                    print 'Using lin gauss'
                    policy = self.alg_map[task].cur[0].traj_distr

                try:
                    task_to_samples[task] = self.alg_map[task].iteration(task_to_samples[task], opt_samples[task], reset=not i)
                    if len(task_to_samples[task]) and i < traj_opt_steps - 1:
                        task_to_samples[task] = self.agent.resample(task_to_samples[task], policy, self.config['num_samples'])
                        opt_samples[task] = []
                except:
                    traceback.print_exception(*sys.exc_info())

                end_time = time.time()
                cpu_times.append(end_time-start_time)

        if self.config['log_timing']:
            with open(self.time_log, 'a') as f:
                f.write('Average time to perform algorithm update with resample for a single task: ')
                f.write(str(np.average(cpu_times)))
                f.write('\n')

        path_samples = []
        for path in sample_paths:
            for sample in path:
                path_samples.append(sample)

        start_time = time.time()
        self.update_primitives(path_samples)
        self.update_value_network(all_samples, first_ts_only=True)
        self.policy_opt.store_weights()
        self.policy_opt.store_weights(self.policy_opt.weight_dir+'_trained')
        end_time = time.time()

        if self.config['log_timing']:
            with open(self.time_log, 'a') as f:
                f.write('Time to update value and primitive network and store weights: ')
                f.write(str(end_time-start_time))
                f.close()

        with open('tf_saved/'+self.config['weight_dir']+'/pretrain_path_samples.sl', 'w+') as f:
            pickle.dump(path_samples, f)
        with open('tf_saved/'+self.config['weight_dir']+'/pretrain_samples.sl', 'w+') as f:
            pickle.dump(all_successful_samples, f)

        import ipdb; ipdb.set_trace()


    def update_primitives(self, samples):
        dP, dO = len(self.task_list), self.alg_map.values()[0].dPrimObs
        dObj, dTarg = self.alg_map.values()[0].dObj, self.alg_map.values()[0].dTarg
        dP += dObj + dTarg
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        for sample in samples:
            sample.agent = self.agent
            for t in range(sample.T):
                obs = [sample.get_prim_obs(t=t)]
                mu = [np.concatenate([sample.get(TASK_ENUM, t=t), sample.get(OBJ_ENUM, t=t), sample.get(TARG_ENUM, t=t)])]
                prc = [np.eye(dP)]
                wt = [1. / (t+1)] # [np.exp(-sample.task_cost)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.policy_opt.update_primitive_filter(obs_data, tgt_mu, tgt_prc, tgt_wt)


    def update_value_network(self, samples, first_ts_only=False):
        dV, dO = 2, self.agent.dVal

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            sample.agent = self.agent
            for t in range(sample.T):
                obs = [sample.get_val_obs(t=t)]
                mu = [sample.success]
                prc = [np.eye(dV)]
                wt = [10. / (t+1)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))
                if first_ts_only: break

        if len(tgt_mu):
            self.policy_opt.update_value(obs_data, tgt_mu, tgt_prc, tgt_wt)


    def sample_current_policies(self, state, cond=0, task=None):
        if task is None:
            sample = Sample(self.agent)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TARGETS_ENUM, self.agent.target_vecs[cond].copy(), 0)
            obs = sample.get_prim_obs(t=0)
            task_distr, obj_distr, targ_distr = self.policy_opt.task_distr(obs)
            task_ind, obj_ind, targ_ind = np.argmax(task_distr), np.argmax(obj_distr), np.argmax(targ_distr)
        else:
            task_ind, obj_ind, targ_ind = task

        task = self.agent.task_list[task_ind]
        obj = self.agent.obj_list[obj_ind]
        targ = self.agent.targ_list[targ_ind]
        print 'Executing {0} on {1} to {2}'.format(task, obj, targ)

        policy = self.rollout_policies[task]
        sample = self.agent.sample_task(policy, cond, state, (task, obj, targ), noisy=False)
        return sample


    def run_condition(self, cond, steps=5):
        state = self.agent.x0[cond]
        for _ in range(steps):
            sample = self.sample_current_policies(state, cond)
            self.agent.animate_sample(sample)
            state = sample.end_state
