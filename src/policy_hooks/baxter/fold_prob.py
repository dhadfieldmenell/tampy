"""
Defines utility functions for planning in the sorting domain
"""
from collections import OrderedDict
import copy
import itertools
import numpy as np
import random
import time

from core.internal_repr.plan import Plan
from core.util_classes.namo_predicates import dsafe
from pma.hl_solver import FFSolver
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.policy_solver_utils import *


prob_file = "../domains/laundry_domain/laundry_probs/fold.prob"
domain_file = "../domains/laundry_domain/laundry.domain"
mapping_file = "policy_hooks/baxter/fold_task_mapping"


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    # Information is track by the environment
    x0s = []
    for i in range(conditions):
        x0 = np.zeros((dX,))
        x0[state_inds['baxter', 'lArmPose']] = [0.39620987, -0.97739414, -0.04612781, 1.74220501, 0.03562036, 0.8089644, -0.45207411]
        x0[state_inds['baxter', 'rArmPose']] = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
        x0s.append(x0)
    return x0s

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

def get_plans():
    tasks = get_tasks(mapping_file)
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    for task in tasks:
        next_task_str = copy.deepcopy(tasks[task])
        plan = plan_from_str(next_task_str, prob_file, domain_file, env, openrave_bodies)
        for i in range(len(prim_options[utils.LEFT_TARG_ENUM])):
            for j in range(len(prim_options[utils.RIGHT_TARG_ENUM])):
                plans[(list(tasks.keys()).index(task), i, j)] = plan
        if env is None:
            env = plan.env
            for param in list(plan.params.values()):
                if not param.is_symbol() and param.openrave_body is not None:
                    openrave_bodies[param.name] = param.openrave_body
    return plans, openrave_bodies, env


def get_end_targets():
    return {}


# def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
#     tasks = get_tasks(mapping_file)
#     next_task_str = copy.deepcopy(tasks[task])
#     for j in range(len(next_task_str)):
#         next_task_str[j]= next_task_str[j].format(*targets)

#     return plan_from_str(next_task_str, prob_file.format(num_cans), domain_file, env, openrave_bodies)


def get_prim_choices():
    out = OrderedDict({})
    out[utils.TASK_ENUM] = list(get_tasks(mapping_file).keys())
    out[utils.LEFT_TARG_ENUM] = ['left_rest_pose', 'bottom_left', 'top_left', 'bottom_right', 'top_right', 'leftmost', 'rightmost']
    out[utils.RIGHT_TARG_ENUM] = ['right_rest_pose', 'bottom_left', 'top_left', 'bottom_right', 'top_right', 'leftmost', 'rightmost']
    return out


def get_vector(config):
    state_vector_include = {
        'baxter': ['lArmPose', 'lGripper', 'rArmPose', 'rGripper', 'ee_left_pos', 'ee_right_pos'] ,
        'left_corner': ['pose'],
        'right_corner': ['pose'],
        'leftmost': ['pose'],
        'rightmost': ['pose'],
        'highest_left': ['pose'],
        'highest_right': ['pose'],
    }

    action_vector_include = {
        'baxter': ['ee_left_pos', 'lGripper', 'ee_right_pos', 'rGripper']
    }

    target_vector_include = {
        'left_rest_pose': ['value'],
        'right_rest_pose': ['value']
    }


    return state_vector_include, action_vector_include, target_vector_include
