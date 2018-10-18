from multiprocessing import Process, Pool
import atexit
import subprocess
import logging
import imp
import os
import os.path
import sys
import copy
import argparse
from datetime import datetime
import threading
import pprint
import time
import traceback

import numpy as np
import tensorflow as tf

from roslaunch.core import RLException
from roslaunch.parent import ROSLaunchParent

from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *
from gps.sample.sample_list import SampleList

from policy_hooks.mcts import MCTS
from policy_hooks.state_mcts import StateMCTS
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
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
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.value_server import ValueServer
from policy_hooks.primitive_server import PrimitiveServer
from policy_hooks.policy_server import PolicyServer
from policy_hooks.rollout_server import RolloutServer


def spawn_server(cls, hyperparams):
    server = cls(hyperparams)
    server.run()

class MultiProcessMain(object):
    def __init__(self, config):
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
            'state_include': [utils.STATE_ENUM],
            'obs_include': [utils.STATE_ENUM,
                            utils.TARGETS_ENUM,
                            utils.TASK_ENUM,
                            utils.OBJ_POSE_ENUM,
                            utils.TARG_POSE_ENUM],
            'prim_obs_include': [utils.STATE_ENUM,
                                 utils.TARGETS_ENUM],
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
            'viewer': None,
            'model': None,
            'get_hl_plan': prob.hl_plan_for_state,
            'env': env,
            'openrave_bodies': openrave_bodies,
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
            'gpu_fraction': 0.2,
            'update_size': self.config['update_size'],
        }

        alg_map = {}
        for task in self.task_list:
            self.config['algorithm']['T'] = self.task_durations[task]
            alg_map[task] = self.config['algorithm']
        self.config['policy_opt'] = self.config['algorithm']['policy_opt']

        self.config['algorithm'] = alg_map

        self.agent = self.config['agent']['type'](self.config['agent'])

        self.fail_value = self.config['fail_value']
        self.alg_map = {}
        self.policy_opt = None
        self.config['algorithm'][task]['policy_opt']['scope'] = 'value'
        self.config['algorithm'][task]['policy_opt']['weight_dir'] = self.config['weight_dir']
        self.task_durations = self.config['task_durations']
        for task in self.task_list:
            self.config['algorithm'][task]['policy_opt']['prev'] = 'skip'
            self.config['algorithm'][task]['agent'] = self.agent
            self.config['algorithm'][task]['init_traj_distr']['T'] = self.task_durations[task]
            self.config['algorithm'][task]['task'] = task
            self.alg_map[task] = self.config['algorithm'][task]['type'](self.config['algorithm'][task])
            self.policy_opt = self.alg_map[task].policy_opt
            self.alg_map[task].set_conditions(len(self.agent.x0))
            self.alg_map[task].agent = self.agent

        for task in self.task_list:
            self.config['algorithm'][task]['policy_opt']['prev'] = None
        self.config['alg_map'] = self.alg_map

        # self.policy_opt.sess.close()
        # self.policy_opt.sess = None

        self.weight_dir = self.config['weight_dir']

        self.traj_opt_steps = self.config['traj_opt_steps']
        self.num_samples = self.config['num_samples']

        self.mcts = []

        for condition in range(len(self.agent.x0)):
            self.mcts.append(MCTS(
                                  self.task_list,
                                  None,
                                  self.config['plan_f'],
                                  self.config['cost_f'],
                                  self.config['goal_f'],
                                  self.config['target_f'],
                                  self.config['encode_f'],
                                  None,
                                  None,
                                  None,
                                  condition,
                                  self.agent,
                                  self.config['branching_factor'],
                                  self.config['num_samples'],
                                  self.config['num_distilled_samples'],
                                  soft_decision=1.0,
                                  C=2.,
                                  max_depth=self.config['max_tree_depth'],
                                  always_opt=False
                                  ))

        self.config['mcts'] = self.mcts
        self.config['agent'] = self.agent
        self.config['alg_map'] = self.alg_map
        self.config['dX'] = self.dX
        self.config['dU'] = self.dU
        self.config['symbolic_bound'] = self.symbolic_bound
        self.config['dO'] = self.agent.dO
        self.config['dPrimObs'] = self.agent.dPrim
        self.config['dObj'] = self.config['algorithm'].values()[0]['dObj'] 
        self.config['dTarg'] = self.config['algorithm'].values()[0]['dTarg'] 
        self.config['state_inds'] = self.state_inds
        self.config['action_inds'] = self.action_inds
        self.config['policy_out_coeff'] = self.policy_out_coeff
        self.config['policy_inf_coeff'] = self.policy_inf_coeff
        self.config['target_inds'] = self.target_inds
        self.config['target_dim'] = self.target_dim
        self.config['task_list'] = self.task_list
        self.config['time_log'] = 'tf_saved/'+self.config['weight_dir']+'/timing_info.txt'

        self.spawn_servers(self.config)

    def spawn_servers(self, config):
        self.processes = []
        if self.config['mp_server']:
            self.create_mp_servers(config)
        if self.config['pol_server']:
            self.create_pol_servers(config)
        if self.config['mcts_server']:
            self.create_rollout_servers(config)

    def start_servers(self):
        for p in self.processes:
            p.start()
            time.sleep(1)

    def create_server(self, server_cls, hyperparams):
        p = Process(target=spawn_server, args=(server_cls, hyperparams))
        p.daemon = True
        self.processes.append(p)

    def create_mp_servers(self, hyperparams):
        for n in range(hyperparams['n_optimizers']):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['id'] = n
            if 'scope' in new_hyperparams:
                new_hyperparams['policy_opt']['scope'] = new_hyperparams['scope']
            self.create_server(new_hyperparams['opt_server_type'], new_hyperparams)

    def create_pol_servers(self, hyperparams):
        for task in self.task_list:
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['scope'] = task
            self.create_server(PolicyServer, new_hyperparams)

        new_hyperparams = copy.copy(hyperparams)
        new_hyperparams['scope'] = 'value'
        self.create_server(ValueServer, new_hyperparams)

        new_hyperparams = copy.copy(hyperparams)
        new_hyperparams['scope'] = 'primitive'
        self.create_server(PrimitiveServer, new_hyperparams)

    def create_rollout_servers(self, hyperparams):
        for n in range(hyperparams['n_rollout_servers']):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['id'] = n
            self.create_server(RolloutServer, new_hyperparams)

    def watch_processes(self, kill_all=False):
        exit = False
        while not exit and len(self.processes):
            for n in range(len(self.processes)):
                p = self.processes[n]
                if not p.is_alive():
                    message = 'Killing All.' if kill_all else 'Restarting Dead Process.'
                    print 'Process died. ' + message
                    exit = kill_all
                    if exit: break
            time.sleep(1)

        for p in self.processes:
            if p.is_alive(): p.terminate()

    def start(self, kill_all=False):
        if self.config['log_timing']:
            with open(self.config['time_log'], 'a+') as f:
                f.write('\n\n\n\n\nTiming info for {0}:'.format(datetime.now()))
        try:
            self.roscore = ROSLaunchParent('train_roscore', [], is_core=True, num_workers=16, verbose=True)
            self.roscore.start()
        except RLException as e:
            self.roscore = None
        time.sleep(1)
        self.start_servers()
        self.watch_processes(kill_all)
        if self.roscore is not None: self.roscore.shutdown()
