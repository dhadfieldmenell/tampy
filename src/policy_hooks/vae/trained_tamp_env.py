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

from core.util_classes.openrave_body import OpenRAVEBody
from policy_hooks.mcts_explore import MCTSExplore
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.mcts import MCTS
from policy_hooks.utils.load_task_definitions import *

from policy_hooks.vae.reward_trainer import RewardTrainer
from policy_hooks.vae.vae_server import VAEServer
from policy_hooks.vae.vae_trainer import VAETrainer
from policy_hooks.vae.vae_rollout_server import VAERolloutServer
from policy_hooks.vae.vae_tamp_rollout_server import VAETampRolloutServer

from policy_hooks.namo.namo_hyperparams import config as namo_config

from baxter_gym.envs import MJCEnv
from policy_hooks.vae.vae_env import VAEEnvWrapper

class NAMOSortenv(VAEEnvWrapper):
    def __init__(self):
        # self.config = namo_config
        # prob = config['prob']

        # if 'num_objs' in config:
        #     prob.NUM_OBJS = config['num_objs']

        # conditions = self.config['num_conds']
        # self.task_list = tuple(get_tasks(self.config['task_map_file']).keys())

        # self.task_durations = get_task_durations(self.config['task_map_file'])
        # self.config['task_list'] = self.task_list
        # task_encoding = get_task_encoding(self.task_list)

        # plans = {}
        # task_breaks = []
        # goal_states = []
        # targets = []
        # for _ in range(conditions):
        #     targets.append(prob.get_end_targets(prob.NUM_OBJS))

        # plans, openrave_bodies, env = prob.get_plans()

        # state_vector_include, action_vector_include, target_vector_include = prob.get_vector(self.config)

        # self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], self.config['robot_name'], self.config['attr_map'], state_vector_include, action_vector_include)

        # self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], self.config['attr_map'], target_vector_include)
        
        # x0 = prob.get_random_initial_state_vec(self.config, plans, self.dX, self.state_inds, conditions)

        # for plan in plans.values():
        #     plan.state_inds = self.state_inds
        #     plan.action_inds = self.action_inds
        #     plan.dX = self.dX
        #     plan.dU = self.dU
        #     plan.symbolic_bound = self.symbolic_bound
        #     plan.target_dim = self.target_dim
        #     plan.target_inds = self.target_inds

        # sensor_dims = {
        #     utils.STATE_ENUM: self.symbolic_bound,
        #     utils.ACTION_ENUM: self.dU,
        #     utils.TRAJ_HIST_ENUM: self.dU*self.config['hist_len'],
        #     utils.TASK_ENUM: len(self.task_list),
        #     utils.TARGETS_ENUM: self.target_dim,
        # }
        # for enum in self.config['sensor_dims']:
        #     sensor_dims[enum] = self.config['sensor_dims'][enum]

        # self.prim_bounds = []
        # self.prim_dims = OrderedDict({})
        # self.config['prim_dims'] = self.prim_dims
        # options = prob.get_prim_choices()
        # ind = len(self.task_list)
        # self.prim_bounds.append((0, ind))
        # for enum in options:
        #     if enum == utils.TASK_ENUM: continue
        #     n_options = len(options[enum])
        #     next_ind = ind+n_options
        #     self.prim_bounds.append((ind, next_ind))
        #     self.prim_dims[enum] = n_options
        #     ind = next_ind
        # for enum in self.prim_dims:
        #     sensor_dims[enum] = self.prim_dims[enum]
        # self.config['prim_bounds'] = self.prim_bounds
        # self.config['prim_dims'] = self.prim_dims

        # # self.config['goal_f'] = prob.goal_f
        # # self.config['cost_f'] = prob.cost_f
        # self.config['target_f'] = None # prob.get_next_target
        # self.config['encode_f'] = None # prob.sorting_state_encode
        # # self.config['weight_file'] = 'tf_saved/2018-09-12 23:43:45.748906_namo_5.ckpt'

        # self.config['task_durations'] = self.task_durations

        # self.policy_inf_coeff = self.config['algorithm']['policy_inf_coeff']
        # self.policy_out_coeff = self.config['algorithm']['policy_out_coeff']
        # self.config['agent'] = {
        #     'type': self.config['agent_type'],
        #     'x0': x0,
        #     'targets': targets,
        #     'task_list': self.task_list,
        #     'plans': plans,
        #     'task_breaks': task_breaks,
        #     'task_encoding': task_encoding,
        #     'task_durations': self.task_durations,
        #     'state_inds': self.state_inds,
        #     'action_inds': self.action_inds,
        #     'target_inds': self.target_inds,
        #     'dU': self.dU,
        #     'dX': self.symbolic_bound,
        #     'symbolic_bound': self.symbolic_bound,
        #     'target_dim': self.target_dim,
        #     'get_plan': None, # prob.get_plan,
        #     'sensor_dims': sensor_dims,
        #     'state_include': self.config['state_include'],
        #     'obs_include': self.config['obs_include'],
        #     'prim_obs_include': self.config['prim_obs_include'],
        #     'prim_out_include': self.config['prim_out_include'],
        #     'val_obs_include': self.config['val_obs_include'],
        #     'conditions': self.config['num_conds'],
        #     'solver': None,
        #     'num_objs': prob.NUM_OBJS,
        #     'obj_list': [],
        #     'stochastic_conditions': False,
        #     'image_width': utils.IM_W,
        #     'image_height': utils.IM_H,
        #     'image_channels': utils.IM_C,
        #     'hist_len': self.config['hist_len'],
        #     'T': 1,
        #     'viewer': config['viewer'],
        #     'model': None,
        #     'get_hl_plan': None,
        #     'env': env,
        #     'openrave_bodies': openrave_bodies,
        #     'n_dirs': self.config['n_dirs'],
        #     'prob': prob,
        #     'attr_map': self.config['attr_map'],
        #     'image_width': self.config['image_width'],
        #     'image_height': self.config['image_height'],
        #     'image_channels': self.config['image_channels'],
        #     'prim_dims': self.prim_dims,
        #     'solver_type': self.config['solver_type'],
        #     'robot_name': self.config['robot_name'],
        #     'policy_inf_coeff': self.config['policy_inf_coeff'],
        #     'policy_out_coeff': self.config['policy_out_coeff'],

        # }

        # agent = self.config['agent']['type'](self.config['agent'])
        # self.weight_dir = self.config['weight_dir']
        # self.config['dX'] = self.dX
        # self.config['dU'] = self.dU
        # self.config['symbolic_bound'] = self.symbolic_bound
        # self.config['dO'] = agent.dO
        # self.config['dPrimObs'] = agent.dPrim
        # self.config['dValObs'] = agent.dVal
        # self.config['dPrimOut'] = agent.dPrimOut 
        # self.config['state_inds'] = self.state_inds
        # self.config['action_inds'] = self.action_inds
        # self.config['policy_out_coeff'] = self.policy_out_coeff
        # self.config['policy_inf_coeff'] = self.policy_inf_coeff
        # self.config['target_inds'] = self.target_inds
        # self.config['target_dim'] = self.target_dim
        # self.config['task_list'] = self.task_list
        # env = agent.mjc_env


        wall_dims = OpenRAVEBody.get_wall_dims('closet')
        config = {
            'obs_include': ['overhead_camera'],
            'include_files': [],
            'include_items': [
                {'name': 'pr2', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.6, 1.), 'rgba': (1, 1, 1, 1)},
            ],
            'view': False,
            'image_dimensions': (hyperparams['image_width'], hyperparams['image_height'])
        }

        self.prob = namo_config['prob']
        self.main_camera_id = 0
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

        items = config['include_items']
        prim_options = self.prob.get_prim_choices()
        for name in prim_options[OBJ_ENUM]:
            if name =='pr2': continue
            cur_color = colors.pop(0)
            items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.4, 1.), 'rgba': tuple(cur_color)})
        for i in range(len(wall_dims)):
            dim, next_trans = wall_dims[i]
            next_trans[0,3] -= 3.5
            next_dim = dim # [dim[1], dim[0], dim[2]]
            pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
            items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})
        env = MJCEnv.load_config(config)

        config = {}
        act_space = env.action_space
        prim_dims =  {'prim{}'.format(i): act_space.nvec[i] for i in range(0, len(act_space.nvec))}
        config['vae'] = {}
        config['vae']['task_dims'] = int(np.prod(prim_dims.values()))
        config['vae']['obs_dims'] = (env.im_height, env.im_wid, 3)
        config['vae']['weight_dir'] = '/home/michaelmcdonald/tampy/src/tf_saved/namo_4'
        config['vae']['rollout_len'] = 20
        config['vae']['load_step'] = 200000
        config['vae']['train_mode'] = 'conditional'
        config['topic'] = 'NAMO'
        super(BlockSortEnv, self).__init__(config, env=env)
