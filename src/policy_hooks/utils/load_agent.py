from collections import OrderedDict
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

import pma.backtrack_ll_solver as bt_ll
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver
from policy_hooks.utils.load_task_definitions import *


def load_agent(config):
    prob = config['prob']
    bt_ll.COL_COEFF = config['col_coeff']
    time_limit = config.get('time_limit', 14400)
    conditions = config['num_conds']
    task_list = tuple(sorted(list(get_tasks(config['task_map_file']).keys())))
    cur_n_rollout = 0
    task_durations = get_task_durations(config['task_map_file'])
    config['task_list'] = task_list
    task_encoding = get_task_encoding(task_list)
    plans = {}
    task_breaks = []
    goal_states = []

    plans, openrave_bodies, env = prob.get_plans()
    state_vector_include, action_vector_include, target_vector_include = config['get_vector'](config)
    dX, state_inds, dU, action_inds, symbolic_bound = utils.get_state_action_inds(list(plans.values())[0], config['robot_name'], config['attr_map'], state_vector_include, action_vector_include)
    target_dim, target_inds = utils.get_target_inds(list(plans.values())[0], config['attr_map'], target_vector_include)
    x0, targets = prob.get_random_initial_state_vec(config, plans, dX, state_inds, conditions)


    im_h = config.get('image_height', utils.IM_H)
    im_w = config.get('image_width', utils.IM_W)
    im_c = config.get('image_channels', utils.IM_C)
    config['image_height'] = im_h
    config['image_width'] = im_w
    config['image_channels'] = im_c
    for plan in list(plans.values()):
        plan.state_inds = state_inds
        plan.action_inds = action_inds
        plan.dX = dX
        plan.dU = dU
        plan.symbolic_bound = symbolic_bound
        plan.target_dim = target_dim
        plan.target_inds = target_inds

    sensor_dims = {
        utils.DONE_ENUM: 1,
        utils.TASK_DONE_ENUM: 1,
        utils.STATE_ENUM: symbolic_bound,
        utils.ACTION_ENUM: dU,
        utils.TRAJ_HIST_ENUM: int(dU*config['hist_len']),
        utils.STATE_DELTA_ENUM: int(symbolic_bound*config['hist_len']),
        utils.STATE_HIST_ENUM: int((1+symbolic_bound)*config['hist_len']),
        utils.TASK_ENUM: len(task_list),
        utils.TARGETS_ENUM: target_dim,
        utils.ONEHOT_TASK_ENUM: len(list(plans.keys())),
        utils.IM_ENUM: im_h * im_w * im_c
    }
    for enum in config['sensor_dims']:
        sensor_dims[enum] = config['sensor_dims'][enum]
    if config['add_action_hist']:
        config['prim_obs_include'].append(utils.TRAJ_HIST_ENUM)
    if config['add_obs_delta']:
        config['prim_obs_include'].append(utils.STATE_DELTA_ENUM)
    if config['add_task_hist']:
        config['prim_obs_include'].append(utils.TASK_HIST_ENUM)
    if config.get('add_hl_image', False):
        config['prim_obs_include'].append(utils.IM_ENUM)
        config['load_render'] = True
    if config.get('add_image', False):
        config['obs_include'].append(utils.IM_ENUM)
        config['load_render'] = True
    prim_bounds = []
    prim_dims = OrderedDict({})
    config['prim_dims'] = prim_dims
    options = prob.get_prim_choices()
    nacts = np.prod([len(options[e]) for e in options])
    ind = len(task_list)
    prim_bounds.append((0, ind))
    for enum in options:
        if enum == utils.TASK_ENUM: continue
        n_options = len(options[enum])
        next_ind = ind+n_options
        prim_dims[enum] = n_options
        ind = next_ind
    for enum in prim_dims:
        sensor_dims[enum] = prim_dims[enum]
    ind = 0
    prim_bounds = []
    if config.get('onehot_task', False):
        config['prim_out_include'] = [utils.ONEHOT_TASK_ENUM]
        prim_bounds.append((0, len(list(plans.keys()))))
        ind = prim_bounds[0][1]
    else:
        prim_bounds.append((0, len(task_list)))
        ind = len(task_list)
        for enum in config['prim_out_include']:
            if enum == utils.TASK_ENUM: continue
            prim_bounds.append((ind, ind+prim_dims[enum]))
            ind += prim_dims[enum]
    sensor_dims[utils.TASK_HIST_ENUM] = int(config['task_hist_len'] * ind)
    config['prim_bounds'] = prim_bounds
    config['prim_dims'] = prim_dims

    config['target_f'] = None # prob.get_next_target
    config['encode_f'] = None # prob.sorting_state_encode

    config['task_durations'] = task_durations

    agent_config = {
        'num_objs': config['num_objs'],
        'num_targs': config['num_targs'],
        'type': config['agent_type'],
        'x0': x0,
        'targets': targets,
        'task_list': task_list,
        'plans': plans,
        'task_breaks': task_breaks,
        'task_encoding': task_encoding,
        'task_durations': task_durations,
        'state_inds': state_inds,
        'action_inds': action_inds,
        'target_inds': target_inds,
        'dU': dU,
        'dX': symbolic_bound,
        'symbolic_bound': symbolic_bound,
        'target_dim': target_dim,
        'get_plan': None, # prob.get_plan,
        'sensor_dims': sensor_dims,
        'state_include': config['state_include'],
        'obs_include': config['obs_include'],
        'prim_obs_include': config['prim_obs_include'],
        'prim_out_include': config['prim_out_include'],
        'val_obs_include': config['val_obs_include'],
        'conditions': config['num_conds'],
        'solver': None,
        'rollout_seed': config.get('rollout_seed', False),
        'num_objs': config['num_objs'],
        'obj_list': [],
        'stochastic_conditions': False,
        'image_width': im_w,
        'image_height': im_h,
        'image_channels': im_c,
        'hist_len': config['hist_len'],
        'T': 1,
        'viewer': config.get('viewer', False),
        'model': None,
        'get_hl_plan': None,
        'env': env,
        'openrave_bodies': openrave_bodies,
        'n_dirs': config['n_dirs'],
        'prob': prob,
        'attr_map': config['attr_map'],
        'prim_dims': prim_dims,
        'mp_solver_type': config['mp_solver_type'],
        'robot_name': config['robot_name'],
        'split_nets': config['split_nets'],
        'policy_inf_coeff': config['policy_inf_coeff'],
        'policy_out_coeff': config['policy_out_coeff'],
        'master_config': config,
    }
    agent_config['agent_load'] = True
    return agent_config


def build_agent(config):
    agent = config['type'](config)
    return agent

