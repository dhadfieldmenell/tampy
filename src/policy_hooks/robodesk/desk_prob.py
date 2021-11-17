import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

from core.internal_repr.plan import Plan
from core.util_classes.namo_predicates import dsafe
from core.util_classes.openrave_body import *
from pma.hl_solver import FFSolver
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils


domain_file = "../domains/robot_domain/right_desk.domain"
mapping_file = "policy_hooks/robodesk/robot_task_mapping"

GOAL_OPTIONS = [
                '(SlideDoorOpen shelf_handle shelf)',
                '(SlideDoorOpen drawer_handle drawer)',
                '(NearGripper panda green_button)',
                '(Lifted upright_block panda)',
                '(Lifted ball panda)',
                '(Near upright_block off_desk_target)',
                '(Near upright_block shelf_target)',
                '(Near flat_block bin_target)',
                '(Stacked upright_block flat_block)'
                ]


def prob_file(descr=None):
    return "../domains/robot_domain/probs/robodesk_prob.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))

    out[utils.OBJ_ENUM] = ['ball', 'upright_block', 'flat_block']
    out[utils.TARG_ENUM] = ['bin_target', 'shelf_target', 'off_desk_target']
    out[utils.DOOR_ENUM] = ['drawer', 'shelf']
    for door in ['drawer', 'shelf']:
        out[utils.OBJ_ENUM].append('{}_handle'.format(door))
    return out


def get_vector(config):
    state_vector_include = {
        'panda': ['right', 'right_ee_pos', 'right_ee_rot', 'right_gripper', 'pose']
    }

    for item in ['ball', 'upright_block', 'flat_block', 'green_button']:
        state_vector_include[item] = ['pose', 'rotation']

    for door in ['drawer', 'shelf']:
        state_vector_include[door] = ['pose', 'rotation', 'hinge']

    action_vector_include = {
        'panda': ['right', 'right_gripper']
    }

    target_vector_include = {
        'bin_target': ['value', 'rotation'],
        'shelf_target': ['value', 'rotation'],
        'off_desk_target': ['value', 'rotation'],
    }
    return state_vector_include, action_vector_include, target_vector_include


def get_plans(use_tf=False):
    tasks = get_tasks(mapping_file)
    task_ids = sorted(list(get_tasks(mapping_file).keys()))
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    params = None
    sess = None

    for obj_ind, obj in enumerate(prim_options[OBJ_ENUM]):
        for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
            for door_ind, door in enumerate(prim_options[DOOR_ENUM]):
                for task_ind, task in enumerate(tasks.keys()):
                    next_task_str = copy.deepcopy(tasks[task])
                    new_task_str = []
                    for step in next_task_str:
                        new_task_str.append(step.format(obj, targ, door))
                    plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                    params = plan.params
                    plans[task_ind, obj_ind, targ_ind, door_ind] = plan
                    if env is None:
                        env = plan.env
                        for param in list(plan.params.values()):
                            if hasattr(param, 'geom'):
                                if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                    param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                                openrave_bodies[param.name] = param.openrave_body

    return plans, openrave_bodies, env


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    return [np.zeros(dX)], [{'shelf_target': np.zeros(3)}]

