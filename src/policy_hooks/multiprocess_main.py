from multiprocessing import Process, Pool
import multiprocessing as mp
import atexit
from collections import OrderedDict
import subprocess
import logging
import imp
import os
import os.path
import psutil
import sys
import copy
import argparse
from datetime import datetime
from threading import Thread
import pprint
import psutil
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

from roslaunch.core import RLException
from roslaunch.parent import ROSLaunchParent
import rosgraph
import rospy

from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *
from gps.sample.sample_list import SampleList

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.mcts import MCTS
from policy_hooks.state_mcts import StateMCTS
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.task_net import tf_value_network, tf_binary_network, tf_classification_network
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
from policy_hooks.log_server import LogServer
from policy_hooks.tf_models import tf_network, multi_modal_network_fp
from policy_hooks.view_server import ViewServer


def spawn_server(cls, hyperparams):
    server = cls(hyperparams)
    server.run()

class MultiProcessMain(object):
    def __init__(self, config):
        self.monitor = True
        self.cpu_use = []
        self.config = config
        prob = config['prob']
        self.config['group_id'] = config.get('group_id', 0)
        time_limit = config.get('time_limit', 3600)

        conditions = self.config['num_conds']
        self.task_list = tuple(get_tasks(self.config['task_map_file']).keys())
        self.cur_n_rollout = 0
        if 'multi_policy' not in self.config: self.config['multi_policy'] = False
        self.pol_list = self.task_list if self.config.get('split_nets', False) else ('control',)
        self.config['policy_list'] = self.pol_list
        self.task_durations = get_task_durations(self.config['task_map_file'])
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)

        plans = {}
        task_breaks = []
        goal_states = []

        plans, openrave_bodies, env = prob.get_plans()

        state_vector_include, action_vector_include, target_vector_include = self.config['get_vector'](self.config)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], self.config['robot_name'], self.config['attr_map'], state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], self.config['attr_map'], target_vector_include)
        
        x0, targets = prob.get_random_initial_state_vec(self.config, plans, self.dX, self.state_inds, conditions)

        '''
        targets = []
        cur_targs = {pname: x0[self.state_inds[pname, 'pose']] for pname in self.state_inds if (pname, 'pose') in self.state_inds}
        for _ in range(conditions):
            targets.append(prob.get_end_targets())
        '''


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
        }
        for enum in self.config['sensor_dims']:
            sensor_dims[enum] = self.config['sensor_dims'][enum]

        self.prim_bounds = []
        self.prim_dims = OrderedDict({})
        self.config['prim_dims'] = self.prim_dims
        options = prob.get_prim_choices()
        ind = len(self.task_list)
        self.prim_bounds.append((0, ind))
        for enum in options:
            if enum == utils.TASK_ENUM: continue
            n_options = len(options[enum])
            next_ind = ind+n_options
            self.prim_bounds.append((ind, next_ind))
            self.prim_dims[enum] = n_options
            ind = next_ind
        for enum in self.prim_dims:
            sensor_dims[enum] = self.prim_dims[enum]
        self.config['prim_bounds'] = self.prim_bounds
        self.config['prim_dims'] = self.prim_dims

        # self.config['goal_f'] = prob.goal_f
        # self.config['cost_f'] = prob.cost_f
        self.config['target_f'] = None # prob.get_next_target
        self.config['encode_f'] = None # prob.sorting_state_encode

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
            'get_plan': None, # prob.get_plan,
            'sensor_dims': sensor_dims,
            'state_include': self.config['state_include'],
            'obs_include': self.config['obs_include'],
            'prim_obs_include': self.config['prim_obs_include'],
            'prim_out_include': self.config['prim_out_include'],
            'val_obs_include': self.config['val_obs_include'],
            'conditions': self.config['num_conds'],
            'solver': None,
            'num_objs': self.config['num_objs'],
            'obj_list': [],
            'stochastic_conditions': False,
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'hist_len': self.config['hist_len'],
            'T': 1,
            'viewer': config.get('viewer', False),
            'model': None,
            'get_hl_plan': None,
            'env': env,
            'openrave_bodies': openrave_bodies,
            'n_dirs': self.config['n_dirs'],
            'prob': prob,
            'attr_map': self.config['attr_map'],
            'image_width': self.config['image_width'],
            'image_height': self.config['image_height'],
            'image_channels': self.config['image_channels'],
            'prim_dims': self.prim_dims,
            'solver_type': self.config['solver_type'],
            'robot_name': self.config['robot_name'],
            'policy_inf_coeff': self.config['policy_inf_coeff'],
            'policy_out_coeff': self.config['policy_out_coeff'],

        }
        if 'cloth_width' in self.config:
            self.config['agent']['cloth_width'] = self.config['cloth_width']
            self.config['agent']['cloth_length'] = self.config['cloth_length']
            self.config['agent']['cloth_spacing'] = self.config['cloth_spacing']
            self.config['agent']['cloth_radius'] = self.config['cloth_radius']

        # action_cost_wp = np.ones((self.config['agent']['T'], self.dU), dtype='float64')
        state_cost_wp = np.ones((self.symbolic_bound), dtype='float64') if 'cost_wp_mult' not in self.config else self.config['cost_wp_mult']
        traj_cost = {
                        'type': StateTrajCost,
                        'data_types': {
                            utils.STATE_ENUM: {
                                'wp': state_cost_wp,
                                'target_state': np.zeros((1, self.symbolic_bound)),
                                'wp_final_multiplier': 5e1,
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

        # self.config['algorithm']['cost'] = {
        #                                         'type': CostSum,
        #                                         'costs': [traj_cost, action_cost],
        #                                         'weights': [1.0, 1.0],
        #                                    }

        self.config['algorithm']['cost'] = traj_cost

        self.config['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dt'] = 1.0

        self.config['algorithm']['policy_opt'] = {
            'type': MultiHeadPolicyOptTf if self.config['multi_policy'] else ControlAttentionPolicyOpt,
            'network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': sensor_dims,
                'n_layers': self.config['n_layers'],
                'num_filters': [5,10],
                'dim_hidden': self.config['dim_hidden'],
            },
            # 'image_network_params': {
            #     'obs_include': ['image'],
            #     'obs_image_data': [OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
            #     'image_width': self.config['image_width'],
            #     'image_height': self.config['image_height'],
            #     'image_channels': self.config['image_channels'],
            #     'sensor_dims': sensor_dims,
            #     'n_fc_layers': self.config['n_layers'],
            #     'num_filters': [5,10],
            #     'fc_layer_size': self.config['dim_hidden'],
            # },
            'primitive_network_params': {
                'obs_include': self.config['agent']['prim_obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': sensor_dims,
                'n_layers': 2,
                'num_filters': [5,10],
                'dim_hidden': [40, 40],
                'output_boundaries': self.prim_bounds,
                'output_order': ['task', 'obj', 'targ'],
            },
            'value_network_params': {
                'obs_include': self.config['agent']['val_obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': sensor_dims,
                'n_layers': 1,
                'num_filters': [5,10],
                'dim_hidden': [40]
            },
            'lr': self.config['lr'],
            'network_model': tf_network,
            'distilled_network_model': tf_network,
            'primitive_network_model': tf_classification_network,
            'value_network_model': tf_value_network,
            'image_network_model': multi_modal_network_fp if 'image' in self.config['agent']['obs_include'] else None,
            'iterations': self.config['train_iterations'],
            'batch_size': self.config['batch_size'],
            'weight_decay': self.config['weight_decay'],
            'weights_file_prefix': 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list,
            'gpu_fraction': 0.25,
            'allow_growth': True,
            'update_size': self.config['update_size'],
        }

        alg_map = {}
        for task in self.task_list:
            self.config['algorithm']['T'] = self.task_durations[task]
            alg_map[task] = copy.copy(self.config['algorithm'])
        self.config['policy_opt'] = self.config['algorithm']['policy_opt']
        self.config['policy_opt']['split_nets'] = self.config.get('split_nets', False)

        self.config['algorithm'] = alg_map

        self.agent = self.config['agent']['type'](self.config['agent'])
        if hasattr(self.agent, 'cloth_init_joints'):
            self.config['agent']['cloth_init_joints'] = self.agent.cloth_init_joints

        self.fail_value = self.config['fail_value']
        self.alg_map = {}
        self.policy_opt = None
        self.task_durations = self.config['task_durations']
        for task in self.task_list:
            self.config['algorithm'][task]['policy_opt']['scope'] = 'value'
            self.config['algorithm'][task]['policy_opt']['weight_dir'] = self.config['weight_dir']
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
        self.check_dirs()

        self.traj_opt_steps = self.config['traj_opt_steps']
        self.num_samples = self.config['num_samples']

        self.mcts = []

        gmms = {}
        for task_name in self.alg_map:
            gmms[task_name] = self.alg_map[task_name].mp_policy_prior.gmm
        for condition in range(len(self.agent.x0)):
            self.mcts.append(MCTS(
                                  self.task_list,
                                  self.prim_dims,
                                  gmms,
                                  None,
                                  None,
                                  condition,
                                  self.agent,
                                  self.config['branching_factor'],
                                  self.config['num_samples'],
                                  self.config['num_distilled_samples'],
                                  soft_decision=False,
                                  max_depth=self.config['max_tree_depth'],
                                  explore_depth=5,
                                  opt_strength=self.config.get('opt_strength', 0),
                                  log_prefix=None,#'tf_saved/'+self.config['weight_dir']+'/rollouts',
                                  curric_thresh=self.config.get('curric_thresh', -1),
                                  her=self.config.get('her', False),
                                  ))

        self.config['mcts'] = self.mcts
        # self.config['agent'] = self.agent
        self.config['alg_map'] = self.alg_map
        self.config['dX'] = self.dX
        self.config['dU'] = self.dU
        self.config['symbolic_bound'] = self.symbolic_bound
        self.config['dO'] = self.agent.dO
        self.config['dPrimObs'] = self.agent.dPrim
        self.config['dValObs'] = self.agent.dVal
        self.config['dPrimOut'] = self.agent.dPrimOut 
        self.config['state_inds'] = self.state_inds
        self.config['action_inds'] = self.action_inds
        self.config['policy_out_coeff'] = self.policy_out_coeff
        self.config['policy_inf_coeff'] = self.policy_inf_coeff
        self.config['target_inds'] = self.target_inds
        self.config['target_dim'] = self.target_dim
        self.config['task_list'] = self.task_list
        self.config['time_log'] = 'tf_saved/'+self.config['weight_dir']+'/timing_info.txt'
        self.config['time_limit'] = time_limit
        self.config['start_t'] = time.time()

        with open('memory_info.txt', 'w+') as f:
            f.write('Process memory info:\n')

        with open('tf_saved/'+self.config['weight_dir']+'/traj_init_log.txt', 'w+') as f:
            f.write('Traj init info:\n')

        self.roscore = None

    def allocate_shared_buffers(self, config):
        buffers = {}
        buf_sizes = {}
        if self.config['policy_opt'].get('split_nets', False):
            for scope in self.task_list:
                buffers[scope] = mp.Array('c', 20 * (2**20))
                buf_sizes[scope] = mp.Value('i')
        else:
            buffers['control'] = mp.Array('c', 20 * (2**20))
            buf_sizes['control'] = mp.Value('i')
        buffers['image'] = mp.Array('c', 20 * (2**20))
        buffers['primitive'] = mp.Array('c', 20 * (2**20))
        buf_sizes['image'] = mp.Value('i')
        buf_sizes['primitive'] = mp.Value('i')
        config['share_buffer'] = True
        config['policy_opt']['share_buffer'] = True
        config['policy_opt']['buffers'] = buffers
        config['policy_opt']['buffer_sizes'] = buf_sizes


    def spawn_servers(self, config):
        self.processes = []
        self.process_info = []
        self.process_configs = {}
        self.threads = []
        if self.config['mp_server']:
            self.create_mp_servers(config)
        if self.config['pol_server']:
            self.create_pol_servers(config)
        if self.config['mcts_server']:
            self.create_rollout_servers(config)
        if self.config['log_server']:
            self.create_log_server(config)
        if self.config['view_server']:
            self.create_view_server(config)

    def start_servers(self):
        for p in self.processes:
            p.start()
            time.sleep(0.5)
        for t in self.threads:
            t.start()

    def create_server(self, server_cls, hyperparams, process=True):
        if process:
            p = Process(target=spawn_server, args=(server_cls, hyperparams))
            p.name = str(server_cls) + '_run_training'
            p.daemon = True
            self.processes.append(p)
            server_id = hyperparams['id'] if 'id' in hyperparams else hyperparams['scope']
            self.process_info.append((server_cls, server_id))
            self.process_configs[p.pid] = (server_cls, hyperparams)
            return p
        else:
            t = Thread(target=spawn_server, args=(server_cls, hyperparams))
            t.daemon = True
            self.threads.append(t)
            return t

    def create_mp_servers(self, hyperparams):
        n_tasks = len(hyperparams['task_list'])
        for n in range(hyperparams['n_optimizers']):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['id'] = n
            new_hyperparams['task_name'] = hyperparams['task_list'][n % n_tasks]
            if 'scope' in new_hyperparams:
                new_hyperparams['policy_opt']['scope'] = new_hyperparams['scope']
            self.create_server(new_hyperparams['opt_server_type'], new_hyperparams)

    def create_pol_servers(self, hyperparams):
        for task in self.pol_list+('primitive',): # ('value', 'primitive'):
            # print task
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['scope'] = task
            # new_hyperparams['id'] = config['server_id']+'_'+str(task)
            self.create_server(PolicyServer, new_hyperparams)

        # new_hyperparams = copy.copy(hyperparams)
        # new_hyperparams['scope'] = 'value'
        # self.create_server(ValueServer, new_hyperparams)

        # new_hyperparams = copy.copy(hyperparams)
        # new_hyperparams['scope'] = 'primitive'
        # self.create_server(PrimitiveServer, new_hyperparams)

    def create_rollout_servers(self, hyperparams, start_idx=0):
        split_mcts_alg = hyperparams.get('split_mcts_alg', True)
        if split_mcts_alg:
            hyperparams['run_mcts_rollouts'] = True
            hyperparams['run_alg_updates'] = False
            for n in range(hyperparams['n_rollout_servers']):
                self._create_rollout_server(hyperparams, start_idx+n)

            hyperparams['run_mcts_rollouts'] = False
            hyperparams['run_alg_updates'] = True
            for n in range(hyperparams['n_alg_servers']):
                self._create_rollout_server(hyperparams, start_idx+n)
        else:
            for n in range(hyperparams['n_rollout_servers']):
                new_hyperparams = copy.copy(hyperparams)
                new_hyperparams['id'] = hyperparams['server_id']+'_'+str(n)
                self.create_server(RolloutServer, new_hyperparams)

        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['id'] = hyperparams['server_id']+'_test'
        self.create_server(RolloutServer, hyperparams)


    def _create_rollout_server(self, hyperparams, idx):
        hyperparams = copy.copy(hyperparams)
        hyperparams['id'] = str(idx)
        p = self.create_server(RolloutServer, hyperparams)
        self.cur_n_rollout += 1
        return p


    def create_log_server(self, hyperparams):
        new_hyperparams = copy.copy(hyperparams)
        spawn_server(LogServer, new_hyperparams)
        if self.roscore is not None: self.roscore.shutdown()
        sys.exit(0)

    def create_view_server(self, hyperparams):
        new_hyperparams = copy.copy(hyperparams)
        spawn_server(ViewServer, new_hyperparams)
        if self.roscore is not None: self.roscore.shutdown()
        sys.exit(0)

    def kill_processes(self):
        for p in self.processes:
            p.terminate()

    def check_processes(self):
        states = []
        for n in range(len(self.processes)):
            p = self.processes[n]
            states.append(p.exitcode)
        return states

    def watch_processes(self, kill_all=False):
        exit = False
        while not exit and len(self.processes):
            for n in range(len(self.processes)):
                p = self.processes[n]
                if not p.is_alive():
                    message = 'Killing All.' if kill_all else 'Restarting Dead Process.'
                    print '\n\nProcess died: ' + str(self.process_info[n]) + ' - ' + message
                    exit = kill_all
                    if kill_all: break
                    process_config = self.process_configs[p.pid]
                    del self.process_info[n]
                    self.create_server(*process_config)
                    print "Relaunched dead process"
            time.sleep(60)
            self.log_mem_info() 

        for p in self.processes:
            if p.is_alive(): p.terminate()

    def check_dirs(self):
        if not os.path.exists('tf_saved/'+self.config['weight_dir']):
            os.makedirs('tf_saved/'+self.config['weight_dir'])
        if not os.path.exists('tf_saved/'+self.config['weight_dir']+'_trained'):
            os.makedirs('tf_saved/'+self.config['weight_dir']+'_trained')

    def start_ros(self):
        if self.roscore is not None or rosgraph.is_master_online(): return
        print('STARTING ROSCORE FROM PYTHON')
        try:
            self.roscore = ROSLaunchParent('train_roscore', [], is_core=True, num_workers=16, verbose=True)
            self.roscore.start()
        except RLException as e:
            print(e)
            pass

    def start(self, kill_all=False):
        self.check_dirs()
        if self.config.get('log_timing', False):
            with open(self.config['time_log'], 'a+') as f:
                f.write('\n\nTiming info for {0}:'.format(datetime.now()))
        self.start_ros()
        if self.config.get('share_buffer', False):
            self.allocate_shared_buffers(self.config)

        self.spawn_servers(self.config)
        self.start_servers()

        if self.monitor:
            self.watch_processes(kill_all)
            if self.roscore is not None: self.roscore.shutdown()

    
    def expand_rollout_servers(self):
        if time.time() - self.config['start_t'] < 1200: return
        self.cpu_use.append(psutil.cpu_percent(interval=1.))
        if np.mean(self.cpu_use[-1:]) < 92.5:
            hyp = copy.copy(self.config)
            hyp['split_mcts_alg'] = True
            hyp['run_alg_updates'] = False
            hyp['run_mcts_rollouts'] = True
            print('Starting rollout server {0}'.format(self.cur_n_rollout))
            p = self._create_rollout_server(hyp, idx=self.cur_n_rollout)
            try:
                p.start()
            except Exception as e:
                print(e)
                print('Failed to expand rollout servers')
            time.sleep(1.)


    def log_mem_info(self):
        '''
        Get list of running process sorted by Memory Usage
        '''
        listOfProcObjects = []
        # Iterate over the list
        for proc in psutil.process_iter():
            try:
                # Fetch process details as dict
                pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
                pinfo['vms'] = proc.memory_info().vms / (1024 * 1024)
                # Append dict to list
                listOfProcObjects.append(pinfo);
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
 
        # Sort list of dict by key vms i.e. memory usage
        listOfProcObjects = sorted(listOfProcObjects, key=lambda procObj: procObj['vms'], reverse=True)
        
        with open('memory_info.txt', 'a+') as f:
            f.write(str(listOfProcObjects)) 
        return listOfProcObjects

