from multiprocessing import Process, Pool
import atexit
from collections import OrderedDict
import subprocess
import logging
import imp
import os
import os.path
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
# import tensorflow as tf

from roslaunch.core import RLException
from roslaunch.parent import ROSLaunchParent
import rosgraph
import rospy

# from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
# from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *
from gps.sample.sample_list import SampleList

# from policy_hooks.control_attention_policy_opt import ControlAttentionPolicyOpt
from policy_hooks.mcts_explore import MCTSExplore
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
# from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
# from policy_hooks.task_net import tf_binary_network, tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver
from policy_hooks.utils.load_task_definitions import *
# from policy_hooks.value_server import ValueServer
# from policy_hooks.primitive_server import PrimitiveServer
# from policy_hooks.policy_server import PolicyServer
# from policy_hooks.rollout_server import RolloutServer
# from policy_hooks.tf_models import tf_network, multi_modal_network_fp
# from policy_hooks.view_server import ViewServer
from policy_hooks.vae.reward_trainer import RewardTrainer
from policy_hooks.vae.vae_server import VAEServer
from policy_hooks.vae.vae_trainer import VAETrainer
from policy_hooks.vae.vae_rollout_server import VAERolloutServer
from policy_hooks.vae.vae_tamp_rollout_server import VAETampRolloutServer


def spawn_server(cls, hyperparams):
    server = cls(hyperparams)
    server.run()


class MultiProcessMain(object):
    def __init__(self, config):
        if config is None:
            return

        self.config = config
        prob = config['prob']

        if 'num_objs' in config:
            prob.NUM_OBJS = config['num_objs']

        conditions = self.config['num_conds']
        self.task_list = tuple(get_tasks(self.config['task_map_file']).keys())

        self.task_durations = get_task_durations(self.config['task_map_file'])
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)

        plans = {}
        task_breaks = []
        goal_states = []
        targets = []
        for _ in range(conditions):
            targets.append(prob.get_end_targets(prob.NUM_OBJS))

        plans, openrave_bodies, env = prob.get_plans()

        state_vector_include, action_vector_include, target_vector_include = prob.get_vector(self.config)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(list(plans.values())[0], self.config['robot_name'], self.config['attr_map'], state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(list(plans.values())[0], self.config['attr_map'], target_vector_include)

        x0 = prob.get_random_initial_state_vec(self.config, plans, self.dX, self.state_inds, conditions)

        for plan in list(plans.values()):
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
            'get_plan': None, # prob.get_plan,
            'sensor_dims': sensor_dims,
            'state_include': self.config['state_include'],
            'obs_include': self.config['obs_include'],
            'prim_obs_include': self.config['prim_obs_include'],
            'prim_out_include': self.config['prim_out_include'],
            'val_obs_include': self.config['val_obs_include'],
            'conditions': self.config['num_conds'],
            'solver': None,
            'num_objs': prob.NUM_OBJS,
            'obj_list': [],
            'stochastic_conditions': False,
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'hist_len': self.config['hist_len'],
            'T': 1,
            'viewer': config['viewer'],
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

        # self.config['algorithm']['cost'] = {
        #                                         'type': CostSum,
        #                                         'costs': [traj_cost, action_cost],
        #                                         'weights': [1.0, 1.0],
        #
        self.agent = self.config['agent']['type'](self.config['agent'])
        self.weight_dir = self.config['weight_dir']

        self.traj_opt_steps = self.config['traj_opt_steps']
        self.num_samples = self.config['num_samples']

        self.mcts = []

        for condition in range(len(x0)):
            self.mcts.append(MCTS(
                                  self.task_list,
                                  self.prim_dims,
                                  [],
                                  None,
                                  None,
                                  condition,
                                  self.agent,
                                  self.config['branching_factor'],
                                  self.config['num_samples'],
                                  self.config['num_distilled_samples'],
                                  soft_decision=1.0,
                                  C=2,
                                  max_depth=self.config['max_tree_depth'],
                                  explore_depth=5,
                                  opt_strength=0,
                                  ))

        self.config['mcts'] = self.mcts
        # self.config['agent'] = self.agent
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
        self.config['time_log'] = self.config['weight_dir']+'/timing_info.txt'

        self.config['rollout_len'] =self.config.get('rollout_len', 20)
        self.config['vae'] = {}
        self.config['vae']['task_dims'] = int(len(self.task_list) + np.sum(list(self.prim_dims.values())))
        self.config['vae']['obs_dims'] = (utils.IM_W, utils.IM_H, 3)
        self.config['vae']['weight_dir'] = self.weight_dir
        self.config['vae']['rollout_len'] = self.config['rollout_len']
        self.config['vae'].update(self.config['train_params'])

        self.rollout_type = VAETampRolloutServer

        self.roscore = None


    @classmethod
    def no_config_load(cls, env, name, config):
        main = cls(None)
        config['env'] = env
        if config['rollout_len'] <= 0:
            config['rollout_len'] = 50
        temp_env = env()
        act_space = temp_env.action_space
        prim_dims =  {'prim{}'.format(i): act_space.nvec[i] for i in range(1, len(act_space.nvec))} if hasattr(act_space, 'nvec') else {}
        n = act_space.nvec[0] if hasattr(act_space, 'nvec') else act_space.n
        config['weight_dir'] = 'tf_saved/'+name.lower()+'_t{0}_vae_data'.format(config['rollout_len']) if config['weight_dir'] == '' else config['weight_dir']
        if hasattr(temp_env, 'n_blocks'):
            config['weight_dir'] += '_{0}_blocks'.format(temp_env.n_blocks)

        config['mcts'] = MCTS(
                              ['task{0}'.format(i) for i in range(n)],
                              prim_dims,
                              [],
                              None,
                              None,
                              0,
                              None,
                              20,
                              1,
                              0,
                              soft_decision=1.0,
                              C=2,
                              max_depth=config['rollout_len'],
                              explore_depth=config['rollout_len'],
                              opt_strength=0,
                              )
        config['vae'] = {}
        config['vae']['task_dims'] = int(n * np.sum(list(prim_dims.values())))
        config['vae']['obs_dims'] = (temp_env.im_height, temp_env.im_wid, 3)
        config['vae']['weight_dir'] = config['weight_dir']
        config['vae']['rollout_len'] = config['rollout_len']
        config['vae']['load_step'] = config['load_step']
        config['vae'].update(config['train_params'])
        config['topic'] = name
        temp_env.close()

        main.rollout_type = VAERolloutServer
        main.config = config
        main.roscore = None
        main.weight_dir = config['weight_dir']
        return main


    def spawn_servers(self, config):
        self.processes = []
        self.process_info = []
        self.process_configs = {}
        self.threads = []
        if self.config['vae_server']:
            self.create_vae_server(config)
        if self.config['rollout_server']:
            self.create_rollout_servers(config)


    def start_servers(self):
        for p in self.processes:
            p.start()
            time.sleep(2)
        for t in self.threads:
            t.start()


    def create_server(self, server_cls, hyperparams, process=True):
        process = not hyperparams['no_child_process']
        if process:
            p = Process(target=spawn_server, args=(server_cls, hyperparams))
            p.daemon = True
            self.processes.append(p)
            server_id = hyperparams['id'] if 'id' in hyperparams else hyperparams['scope']
            self.process_info.append((server_cls, server_id))
            self.process_configs[p.pid] = (server_cls, hyperparams)
        else:
            # t = Thread(target=spawn_server, args=(server_cls, hyperparams))
            # t.daemon = True
            # self.threads.append(t)
            spawn_server(server_cls, hyperparams)


    def create_vae_server(self, hyperparams):
        new_hyperparams = copy.copy(hyperparams)
        new_hyperparams['scope'] = 'vae'
        self.create_server(VAEServer, new_hyperparams)


    def create_rollout_servers(self, hyperparams):
        for n in range(hyperparams['n_rollout_servers']):
            new_hyperparams = copy.copy(hyperparams)
            new_hyperparams['id'] = hyperparams['server_id']+'_'+str(n)
            self.create_server(self.rollout_type, new_hyperparams)


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
            # self.log_mem_info()

        for p in self.processes:
            if p.is_alive(): p.terminate()


    def check_dirs(self):
        if not os.path.exists(self.config['weight_dir']):
            os.makedirs(self.config['weight_dir'])
        if not os.path.exists(self.config['weight_dir']+'_trained'):
            os.makedirs(self.config['weight_dir']+'_trained')


    def start_ros(self):
        if self.roscore is not None or rosgraph.is_master_online(): return
        try:
            self.roscore = ROSLaunchParent('train_roscore', [], is_core=True, num_workers=16, verbose=True)
            self.roscore.start()
        except RLException as e:

            pass


    def start(self, kill_all=False):
        if self.config['train_vae']:
            self.config['id'] = 0
            self.config['vae']['train_mode'] = 'unconditional' if self.config['unconditional'] else 'conditional'
            trainer = VAETrainer(self.config)
            # o, d, d2 = trainer.vae.test_decode()
            # import ipdb; ipdb.set_trace()
            trainer.train()
        elif self.config['train_reward']:
            self.config['id'] = 0
            self.config['vae']['train_mode'] = 'conditional'
            trainer = RewardTrainer(self.config)
            trainer.train()
        else:
            self.check_dirs()
            if 'log_timing' in self.config and self.config['log_timing']:
                with open(self.config['time_log'], 'a+') as f:
                    f.write('\n\nTiming info for {0}:'.format(datetime.now()))
            self.start_ros()
            time.sleep(1)
            self.spawn_servers(self.config)
            self.start_servers()
            self.watch_processes(kill_all)
            if self.roscore is not None: self.roscore.shutdown()
