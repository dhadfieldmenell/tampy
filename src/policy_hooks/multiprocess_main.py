import multiprocessing as mp
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Pool, Queue
from queue import PriorityQueue
import atexit
from collections import OrderedDict
import subprocess
import ctypes
import logging
import imp
import importlib
import os
import os.path
import pickle
import psutil
import sys
import shutil
import copy
import argparse
from datetime import datetime
from threading import Thread
import pprint
import psutil
import time
import traceback
import random

import numpy as np

import software_constants
from gps.algorithm.cost.cost_utils import *

from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.mcts import MCTS
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.task_net import * 
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.policy_server import PolicyServer
from policy_hooks.rollout_server import RolloutServer
from policy_hooks.motion_server import MotionServer
from policy_hooks.task_server import TaskServer
from policy_hooks.tf_models import tf_network, multi_modal_network_fp
import policy_hooks.hl_retrain as hl_retrain
from policy_hooks.utils.load_agent import *


DIR_KEY = 'experiment_logs/'
def spawn_server(cls, hyperparams, load_at_spawn=False):
    if load_at_spawn:
        new_config, config_mod = load_config(hyperparams['args'])
        new_config.update(hyperparams)
        hyperparams = new_config
        hyperparams['main'].init(hyperparams)
        hyperparams['policy_opt']['share_buffer'] = True
        hyperparams['policy_opt']['buffers'] = hyperparams['buffers']
        hyperparams['policy_opt']['buffer_sizes'] = hyperparams['buffer_sizes']

    server = cls(hyperparams)
    server.run()

class QueueManager(SyncManager):
    pass
QueueManager.register('PriorityQueue', PriorityQueue)

class MultiProcessMain(object):
    def __init__(self, config, load_at_spawn=False):
        self.monitor = True
        self.cpu_use = []
        self.config = config
        if load_at_spawn:
            setup_dirs(config, config['args'])
            task_file = config.get('task_map_file', '')
            self.pol_list = ('control',) if not config['args'].split_nets else tuple(get_tasks(task_file).keys())
            config['main'] = self
        else:
            self.init(config)
            self.check_dirs()

    def init(self, config):
        self.config = config
        prob = config['prob']
        self.config['group_id'] = config.get('group_id', 0)
        if 'id' not in self.config: self.config['id'] = -1
        time_limit = config.get('time_limit', 14400)

        conditions = self.config['num_conds']
        self.task_list = tuple(sorted(list(get_tasks(self.config['task_map_file']).keys())))
        self.cur_n_rollout = 0
        if 'multi_policy' not in self.config: self.config['multi_policy'] = False
        self.pol_list = self.task_list if self.config.get('split_nets', False) else ('control',)
        self.config['policy_list'] = self.pol_list
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)
        plans = {}
        task_breaks = []
        goal_states = []

        plans, openrave_bodies, env = prob.get_plans()
        self.plans = plans

        state_vector_include, action_vector_include, target_vector_include = self.config['get_vector'](self.config)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(list(plans.values())[0], self.config['robot_name'], self.config['attr_map'], state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(list(plans.values())[0], self.config['attr_map'], target_vector_include)

        x0, targets = prob.get_random_initial_state_vec(self.config, plans, self.dX, self.state_inds, conditions)

        for plan in list(plans.values()):
            plan.state_inds = self.state_inds
            plan.action_inds = self.action_inds
            plan.dX = self.dX
            plan.dU = self.dU
            plan.symbolic_bound = self.symbolic_bound
            plan.target_dim = self.target_dim
            plan.target_inds = self.target_inds
        self.config['target_f'] = None # prob.get_next_target
        self.config['encode_f'] = None # prob.sorting_state_encode

        config['agent'] = load_agent(config)
        self.sensor_dims = config['agent']['sensor_dims']

        if 'cloth_width' in self.config:
            self.config['agent']['cloth_width'] = self.config['cloth_width']
            self.config['agent']['cloth_length'] = self.config['cloth_length']
            self.config['agent']['cloth_spacing'] = self.config['cloth_spacing']
            self.config['agent']['cloth_radius'] = self.config['cloth_radius']
        self.agent = self.config['agent']['type'](self.config['agent'])
        if hasattr(self.agent, 'cloth_init_joints'):
            self.config['agent']['cloth_init_joints'] = self.agent.cloth_init_joints

        self.fail_value = self.config['fail_value']
        self.policy_opt = None

        self.weight_dir = self.config['weight_dir']

        self.traj_opt_steps = self.config['traj_opt_steps']
        self.num_samples = self.config['num_samples']

        self.mcts = []

        for condition in range(len(self.agent.x0)):
            self.mcts.append(MCTS(
                                  self.task_list,
                                  self.config['agent']['prim_dims'],
                                  None,
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
                                  n_thresh=self.config.get('n_thresh', 10),
                                  her=self.config.get('her', False),
                                  onehot_task=self.config.get('onehot_task', False),
                                  soft=self.config.get('soft', False),
                                  ff_thresh=1.,#self.config.get('ff_thresh', 0),
                                  eta=self.config.get('eta', 1.),
                                  ))
        self._map_cont_discr_tasks()
        self._set_alg_config()
        self.config['mcts'] = self.mcts
        # self.config['agent'] = self.agent
        self.config['alg_map'] = self.alg_map
        self.config['dX'] = self.dX
        self.config['dU'] = self.dU
        self.config['symbolic_bound'] = self.symbolic_bound
        self.config['dO'] = self.agent.dO
        self.config['dPrimObs'] = self.agent.dPrim
        self.config['dValObs'] = self.agent.dVal #+ np.sum([len(options[e]) for e in options])
        self.config['dPrimOut'] = self.agent.dPrimOut
        self.config['state_inds'] = self.state_inds
        self.config['action_inds'] = self.action_inds
        self.config['policy_out_coeff'] = self.policy_out_coeff
        self.config['policy_inf_coeff'] = self.policy_inf_coeff
        self.config['target_inds'] = self.target_inds
        self.config['target_dim'] = self.target_dim
        self.config['task_list'] = self.task_list
        self.config['time_log'] = 'experiment_logs/'+self.config['weight_dir']+'/timing_info.txt'
        self.config['time_limit'] = time_limit
        self.config['start_t'] = time.time()

        self.roscore = None
        self.processes = []


    def _map_cont_discr_tasks(self):
        self.task_types = []
        self.discrete_opts = []
        self.continuous_opts = []
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for key, val in opts.items():
            if hasattr(val, '__len__'):
                self.task_types.append('discrete')
                self.discrete_opts.append(key)
            else:
                self.task_types.append('continuous')
                self.continuous_opts.append(key)


    def _set_alg_config(self):
        self.policy_inf_coeff = self.config['algorithm']['policy_inf_coeff']
        self.policy_out_coeff = self.config['algorithm']['policy_out_coeff']
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

        self.config['algorithm']['cost'] = traj_cost
        self.config['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dt'] = 1.0

        if self.config.get('add_hl_image', False) and any([t.find('cont') >= 0 for t in self.task_types]):
            primitive_network_model = fp_multi_modal_cond_network
        elif self.config.get('add_hl_image', False):
            primitive_network_model = fp_multi_modal_class_network
        elif any([t.find('cont') >= 0 for t in self.task_types]):
            primitive_network_model = tf_cond_network
        elif self.config.get('split_hl_loss', False):
            primitive_network_model = tf_balanced_classification_network
        elif self.config.get('conditional', False):
            primitive_network_model = tf_cond_classification_network
        else:
            primitive_network_model = tf_classification_network if self.config.get('discrete_prim', True) else tf_network

        self.config['algorithm']['policy_opt'] = {
            'q_imwt': self.config.get('q_imwt', 0),
            'll_policy': self.config.get('ll_policy', ''),
            'hl_policy': self.config.get('hl_policy', ''),
            'type': ControlAttentionPolicyOpt,
            'network_params': {
                'obs_include': self.config['agent']['obs_include'],
                'obs_image_data': [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': self.sensor_dims,
                'n_layers': self.config['n_layers'],
                'num_filters': [32, 32, 16],
                'filter_sizes': [5, 5, 5],
                'q_imwt': 1,
                'dim_hidden': self.config['dim_hidden'],
            },
            'primitive_network_params': {
                'obs_include': self.config['agent']['prim_obs_include'],
                'obs_image_data': [IM_ENUM, OVERHEAD_IMAGE_ENUM, LEFT_IMAGE_ENUM, RIGHT_IMAGE_ENUM],
                'image_width': self.config['image_width'],
                'image_height': self.config['image_height'],
                'image_channels': self.config['image_channels'],
                'sensor_dims': self.sensor_dims,
                'n_layers': self.config['prim_n_layers'],
                'num_filters': [32, 32, 8],
                'filter_sizes': [8, 6, 4],
                'dim_hidden': self.config['prim_dim_hidden'],
                'output_boundaries': self.config['prim_bounds'],
                'aux_boundaries': self.config['aux_bounds'],
                'types': self.task_types,
            },
            'aux_boundaries': self.config['aux_bounds'],
            'lr': self.config['lr'],
            'hllr': self.config['hllr'],
            'network_model': tf_network,
            'primitive_network_model': primitive_network_model,
            'iterations': self.config['train_iterations'],
            'batch_size': self.config['batch_size'],
            'weight_decay': self.config['weight_decay'],
            'prim_weight_decay': self.config['prim_weight_decay'],
            'val_weight_decay': self.config['val_weight_decay'],
            'weights_file_prefix': 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list,
            'gpu_fraction': 0.25,
            'allow_growth': True,
            'update_size': self.config['update_size'],
            'prim_update_size': self.config['prim_update_size'],
            'val_update_size': self.config['val_update_size'],
            'solver_type': self.config['solver_type'],
        }
        if self.config.get('conditional', False):
            self.config['algorithm']['policy_opt']['primitive_network_model'] = tf_cond_classification_network

        self.alg_map = {}
        alg_map = {}
        for ind, task in enumerate(self.task_list):
            plan = [pl for lab, pl in self.plans.items() if lab[0] == ind][0]
            self.config['algorithm']['T'] = plan.horizon
            alg_map[task] = copy.copy(self.config['algorithm'])
        self.config['policy_opt'] = self.config['algorithm']['policy_opt']
        self.config['policy_opt']['split_nets'] = self.config.get('split_nets', False)

        self.config['algorithm'] = alg_map
        for task in self.task_list:
            self.config['algorithm'][task]['policy_opt']['scope'] = 'value'
            self.config['algorithm'][task]['policy_opt']['weight_dir'] = self.config['weight_dir']
            self.config['algorithm'][task]['policy_opt']['prev'] = 'skip'
            self.config['algorithm'][task]['agent'] = self.agent
            self.config['algorithm'][task]['init_traj_distr']['T'] = alg_map[task]['T']
            self.config['algorithm'][task]['task'] = task
            self.alg_map[task] = self.config['algorithm'][task]['type'](self.config['algorithm'][task])
            self.policy_opt = self.alg_map[task].policy_opt
            self.alg_map[task].set_conditions(len(self.agent.x0))
            self.alg_map[task].agent = self.agent

        for task in self.task_list:
            self.config['algorithm'][task]['policy_opt']['prev'] = None
        self.config['alg_map'] = self.alg_map


    def allocate_shared_buffers(self, config):
        buffers = {}
        buf_sizes = {}
        #if self.config['policy_opt'].get('split_nets', False):
        #    for scope in self.task_list:
        #        buffers[scope] = mp.Array(ctypes.c_char, (2**28))
        #        buf_sizes[scope] = mp.Value('i')
        #        buf_sizes[scope].value = 0
        #else:
        #    buffers['control'] = mp.Array(ctypes.c_char, (2**28))
        #    buf_sizes['control'] = mp.Value('i')
        #    buf_sizes['control'].value = 0
        for task in self.pol_list:
            buffers[task] = mp.Array(ctypes.c_char, (2**28))
            buf_sizes[task] = mp.Value('i')
            buf_sizes[task].value = 0
        buffers['primitive'] = mp.Array(ctypes.c_char, 20 * (2**28))
        buf_sizes['primitive'] = mp.Value('i')
        buf_sizes['primitive'].value = 0
        buf_sizes['n_data'] = mp.Value('i')
        buf_sizes['n_data'].value = 0
        buf_sizes['n_plans'] = mp.Value('i')
        buf_sizes['n_plans'].value = 0
        buf_sizes['n_failed'] = mp.Value('i')
        buf_sizes['n_failed'].value = 0
        buf_sizes['n_mcts'] = mp.Value('i')
        buf_sizes['n_mcts'].value = 0
        buf_sizes['n_ff'] = mp.Value('i')
        buf_sizes['n_ff'].value = 0
        buf_sizes['n_postcond'] = mp.Value('i')
        buf_sizes['n_postcond'].value = 0
        buf_sizes['n_explore'] = mp.Value('i')
        buf_sizes['n_explore'].value = 0
        buf_sizes['n_rollout'] = mp.Value('i')
        buf_sizes['n_rollout'].value = 0
        config['share_buffer'] = True
        config['buffers'] = buffers
        config['buffer_sizes'] = buf_sizes


    def spawn_servers(self, config):
        self.processes = []
        self.process_info = []
        self.process_configs = {}
        self.threads = []
        self.create_servers(config)


    def start_servers(self):
        for p in self.processes:
            p.start()
            time.sleep(0.1)
        for t in self.threads:
            t.start()


    def create_server(self, server_cls, hyperparams, process=True):
        if hyperparams.get('seq', False):
            spawn_server(server_cls, hyperparams)

        if process:
            p = Process(target=spawn_server, args=(server_cls, hyperparams, True))
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


    def create_pol_servers(self, hyperparams):
        for task in self.pol_list+('primitive',):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['scope'] = task
            new_hyperparams['id'] = task
            self.create_server(PolicyServer, new_hyperparams)


    def create_servers(self, hyperparams, start_idx=0):
        self.create_pol_servers(hyperparams)
        hyperparams['view'] = False
        for n in range(hyperparams['num_motion']):
            self._create_server(hyperparams, MotionServer, start_idx+n)
        for n in range(hyperparams['num_task']):
            self._create_server(hyperparams, TaskServer, start_idx+n)
        for n in range(hyperparams['num_rollout']):
            self._create_server(hyperparams, RolloutServer, start_idx+n)
        hyperparams = copy.copy(hyperparams)
        hyperparams['run_hl_test'] = True
        hyperparams['id'] = 'test'
        hyperparams['view'] = hyperparams['view_policy']
        hyperparams['load_render'] = True
        hyperparams['check_precond'] = False
        self.create_server(RolloutServer, copy.copy(hyperparams))
        hyperparams['id'] = 'moretest'
        hyperparams['view'] = False
        self.create_server(RolloutServer, copy.copy(hyperparams))
        hyperparams['run_hl_test'] = False


    def _create_server(self, hyperparams, cls, idx):
        hyperparams = copy.copy(hyperparams)
        hyperparams['id'] = cls.__name__ + str(idx)
        p = self.create_server(cls, hyperparams)
        return p


    def hl_only_retrain(self, hyperparams):
        software_constants.USE_ROS = False
        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['share_buffers'] = True
        hyperparams['id'] = 'test'
        hyperparams['scope'] = 'primitive'
        descr = hyperparams.get('descr', '')
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        server = PolicyServer(hyperparams)
        server.agent = hyperparams['agent']['type'](hyperparams['agent'])
        ll_dir = hyperparams['ll_policy']
        hl_dir = hyperparams['hl_data']
        print(('Launching hl retrain from', ll_dir, hl_dir))
        hl_retrain.retrain_hl_from_samples(server, hl_dir)


    def hl_retrain(self, hyperparams):
        software_constants.USE_ROS = False
        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['share_buffers'] = True
        hyperparams['id'] = 'test'
        descr = hyperparams.get('descr', '')
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        server = RolloutServer(hyperparams)
        ll_dir = hyperparams['ll_policy']
        hl_dir = hyperparams['hl_data']
        hl_retrain.retrain(server, hl_dir, ll_dir)


    def run_test(self, hyperparams):
        software_constants.USE_ROS = False
        hyperparams['run_mcts_rollouts'] = False
        hyperparams['run_alg_updates'] = False
        hyperparams['run_hl_test'] = True
        hyperparams['check_precond'] = False
        hyperparams['share_buffers'] = False
        hyperparams['load_render'] = True
        hyperparams['agent']['image_height']  = 256
        hyperparams['agent']['image_width']  = 256
        descr = hyperparams.get('descr', '')
        # hyperparams['weight_dir'] = hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
        hyperparams['id'] = 'test'
        self.allocate_shared_buffers(hyperparams)
        self.allocate_queues(hyperparams)
        server = RolloutServer(hyperparams)
        newdir = 'experiment_logs/'+hyperparams['weight_dir'].replace('exp_id0', 'rerun_{0}'.format(descr))
        if not os.path.isdir(newdir):
            os.mkdir(newdir)
        server.hl_test_log = newdir + '/hl_test_rerun_log.npy'
        # if not os.path.isdir('tf_saved/'+hyperparams['weight_dir']+'_testruns'):
        #     os.mkdir('tf_saved/'+hyperparams['weight_dir']+'_testruns')
        # server.hl_test_log = 'tf_saved/' + hyperparams['weight_dir'] + '_testruns/hl_test_rerun_log.npy'
        ind = 0

        no = hyperparams['num_objs']
        print(server.agent.task_list, server.task_list)
        n_vids = 20
        for test_run in range(hyperparams['num_tests']):
            print('RUN:', test_run)
            server.agent.replace_cond(0)
            server.agent.reset(0)
            server.test_hl(save=True, save_video=test_run<n_vids, save_fail=False)
        server.check_hl_statistics()
        '''
        while server.policy_opt.restore_ckpts(ind):
            for _ in range(50):
                server.agent.replace_cond(0)
                server.test_hl(5, save=True, ckpt_ind=ind)
            ind += 1
        '''
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
                    print('\n\nProcess died: ' + str(self.process_info[n]) + ' - ' + message)
                    exit = kill_all
                    if kill_all: break
                    process_config = self.process_configs[p.pid]
                    del self.process_info[n]
                    self.create_server(*process_config)
                    print("Relaunched dead process")
            time.sleep(60)
            self.log_mem_info()

        for p in self.processes:
            if p.is_alive(): p.terminate()

    def check_dirs(self):
        if not os.path.exists('experiment_logs/'+self.config['weight_dir']):
            os.makedirs('experiment_logs/'+self.config['weight_dir'])
        #if not os.path.exists('tf_saved/'+self.config['weight_dir']+'_trained'):
        #    os.makedirs('tf_saved/'+self.config['weight_dir']+'_trained')


    def start(self, kill_all=False):
        #self.check_dirs()
        if self.config.get('share_buffer', True):
            self.allocate_shared_buffers(self.config)
            self.allocate_queues(self.config)

        self.spawn_servers(self.config)
        self.start_servers()

        if self.monitor:
            self.watch_processes(kill_all)


    def expand_rollout_servers(self):
        if not self.config['expand_process'] or time.time() - self.config['start_t'] < 1200: return
        self.cpu_use.append(psutil.cpu_percent(interval=1.))
        if np.mean(self.cpu_use[-1:]) < 92.5:
            hyp = copy.copy(self.config)
            hyp['split_mcts_alg'] = True
            hyp['run_alg_updates'] = False
            hyp['run_mcts_rollouts'] = True
            hyp['run_hl_test'] = False
            print(('Starting rollout server {0}'.format(self.cur_n_rollout)))
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

        return listOfProcObjects


    def allocate_queues(self, config):
        self.queue_manager = QueueManager()
        self.queue_manager.start()

        queue_size = 100
        queues = {}
        config['hl_queue'] = Queue(queue_size)
        config['ll_queue'] = Queue(queue_size)
        config['motion_queue'] = self.queue_manager.PriorityQueue(queue_size)
        config['task_queue'] = self.queue_manager.PriorityQueue(queue_size)
        config['rollout_queue'] = self.queue_manager.PriorityQueue(queue_size)

        for task in self.pol_list+('primitive',):
            queues['{0}_pol'.format(task)] = Queue(50)
        config['queues'] = queues
        return queues


def load_config(args, config=None, reload_module=None):
    config_file = args.config
    if reload_module is not None:
        config_module = reload_module
        imp.reload(config_module)
    else:
        config_module = importlib.import_module(config_file)
    config = config_module.refresh_config(args.nobjs, args.nobjs)
    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds']
    config['common']['num_conds'] = config['num_conds']
    config['algorithm']['conditions'] = config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    #config['weight_dir'] = get_dir_name(config['base_weight_dir'], config['num_objs'], config['num_targs'], i, config['descr'], args)
    #config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs'])
    config['log_timing'] = args.timing
    # config['pretrain_timeout'] = args.pretrain_timeout
    config['hl_timeout'] = args.hl_timeout if args.hl_timeout > 0 else config['hl_timeout']
    config['mcts_server'] = args.mcts_server or args.all_servers
    config['mp_server'] = args.mp_server or args.all_servers
    config['pol_server'] = args.policy_server or args.all_servers
    config['log_server'] = args.log_server or args.all_servers
    config['view_server'] = args.view_server
    config['pretrain_steps'] = args.pretrain_steps if args.pretrain_steps > 0 else config['pretrain_steps']
    config['viewer'] = args.viewer
    config['server_id'] = args.server_id if args.server_id != '' else str(random.randint(0,2**32))
    return config, config_module


def setup_dirs(c, args):
    current_id = 0 if c.get('index', -1) < 0 else c['index']
    if c.get('index', -1) < 0:
        while os.path.isdir(DIR_KEY+c['weight_dir']+'_'+str(current_id)):
            current_id += 1
    c['group_id'] = current_id
    c['weight_dir'] = c['weight_dir']+'_{0}'.format(current_id)
    dir_name = ''
    sub_dirs = [DIR_KEY] + c['weight_dir'].split('/')

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception as e:
        rank = 0
    if rank < 0: rank = 0

    c['rank'] = rank
    if rank == 0:
        for d_ind, d in enumerate(sub_dirs):
            dir_name += d + '/'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
        if args.hl_retrain:
            src = DIR_KEY + args.hl_data + '/hyp.py'
        elif hasattr(args, 'expert_path') and len(args.expert_path):
            src = args.expert_path+'/hyp.py'
        else:
            src = c['source'].replace('.', '/')+'.py'
        shutil.copyfile(src, DIR_KEY+c['weight_dir']+'/hyp.py')
        with open(DIR_KEY+c['weight_dir']+'/__init__.py', 'w+') as f:
            f.write('')
        with open(DIR_KEY+c['weight_dir']+'/args.pkl', 'wb+') as f:
            pickle.dump(args, f, protocol=0)
        with open(DIR_KEY+c['weight_dir']+'/args.txt', 'w+') as f:
            f.write(str(vars(args)))
    else:
        time.sleep(0.1) # Give others a chance to let base set up dirs
    return current_id


def check_dirs(config):
    if not os.path.exists('experiment_logs/'+config['weight_dir']):
        os.makedirs('experiment_logs/'+config['weight_dir'])


def get_dir_name(base, no, nt, ind, descr, args=None):
    dir_name = base + 'objs{0}_{1}/{2}'.format(no, nt, descr)
    return dir_name


