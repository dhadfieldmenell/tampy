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


NUM_CLOTHS = 5

prob_file = "../domains/laundry_domain/laundry_probs/sort{0}.prob".format(NUM_CLOTHS)
pddl_file = "../domains/laundry_domain/sort.pddl"
domain_file = "../domains/laundry_domain/laundry.domain"
mapping_file = "policy_hooks/baxter/sort_task_mapping"

POSSIBLE_CLOTH_LOCS = []
for i in range(40, 70, 3):
    for j in range(20, 60, 2):
        x = i * 0.01
        y = j * 0.01
        POSSIBLE_CLOTH_LOCS.append((x, y, 0.625))
        POSSIBLE_CLOTH_LOCS.append((x, -y, 0.625))

def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    # Information is track by the environment
    x0s = []
    baxter = list(plans.values())[0].params['baxter']
    robot_body = baxter.openrave_body
    baxter.lArmPose[:, 0] = [0.39620987, -0.97739414, -0.04612781, 1.74220501, 0.03562036, 0.8089644, -0.45207411]
    baxter.rArmPose[:, 0] = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
    for i in range(conditions):
        x0 = np.zeros((dX,))
        x0[state_inds['baxter', 'lArmPose']] = [0.39620987, -0.97739414, -0.04612781, 1.74220501, 0.03562036, 0.8089644, -0.45207411]
        x0[state_inds['baxter', 'rArmPose']] = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
        ee_pose = robot_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], 0)
        if ('baxter', 'ee_left_pos') in state_inds:
            x0[state_inds['baxter', 'ee_left_pos']] = ee_pose['left_gripper']['pos']
        if ('baxter', 'ee_right_pos') in state_inds:
            x0[state_inds['baxter', 'ee_right_pos']] = ee_pose['right_gripper']['pos']
        x0s.append(x0)
        locs = np.random.choice(list(range(len(POSSIBLE_CLOTH_LOCS))), NUM_CLOTHS, replace=False)
        for j in range(NUM_CLOTHS):
            x0[state_inds['cloth{0}'.format(j), 'pose']] = POSSIBLE_CLOTH_LOCS[locs[j]]
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
        for i in range(len(prim_options[utils.OBJ_ENUM])):
            for j in range(len(prim_options[utils.TARG_ENUM])):
                obj = prim_options[utils.OBJ_ENUM][i]
                targ = prim_options[utils.TARG_ENUM][j]
                new_task_str = []
                for step in next_task_str:
                    new_task_str.append(step.format(obj, targ))
                plan = plan_from_str(new_task_str, prob_file, domain_file, env, openrave_bodies)
                plans[(list(tasks.keys()).index(task), i, j)] = plan
                if env is None:
                    env = plan.env
                    for param in list(plan.params.values()):
                        if not param.is_symbol() and param.openrave_body is not None:
                            openrave_bodies[param.name] = param.openrave_body
    return plans, openrave_bodies, env


# def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
#     tasks = get_tasks(mapping_file)
#     next_task_str = copy.deepcopy(tasks[task])
#     for j in range(len(next_task_str)):
#         next_task_str[j]= next_task_str[j].format(*targets)

#     return plan_from_str(next_task_str, prob_file.format(num_cans), domain_file, env, openrave_bodies)


def get_end_targets():
    target_map = {}
    target_map['middle_target_1'] = np.array([0.35, 0., 0.625])
    target_map['left_target_1'] = np.array([0.4, 0.75, 0.625])
    target_map['right_target_1'] = np.array([0.4, -0.75, 0.625])
    return target_map


def get_prim_choices():
    out = OrderedDict({})
    out[utils.TASK_ENUM] = list(get_tasks(mapping_file).keys())
    out[utils.OBJ_ENUM] = []
    out[utils.TARG_ENUM] = ['left_target_1', 'right_target_1', 'middle_target_1']
    for i in range(NUM_CLOTHS):
        out[utils.OBJ_ENUM].append('cloth{0}'.format(i))
    return out


def get_vector(config):
    state_vector_include = {
        'baxter': ['lArmPose', 'lGripper', 'rArmPose', 'rGripper', 'ee_left_pos', 'ee_right_pos'],
    }
    for i in range(NUM_CLOTHS):
        state_vector_include['cloth{0}'.format(i)] = ['pose']

    action_vector_include = {
        'baxter': ['ee_left_pos', 'lGripper', 'ee_right_pos', 'rGripper']
    }

    target_vector_include = {
        'left_target_1': ['value'],
        'right_target_1': ['value'],
        'middle_target_1': ['value']
    }


    return state_vector_include, action_vector_include, target_vector_include


def get_sorting_problem(cloth_locs, targets, baxter, failed_preds=[]):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for cloth in cloth_locs:
        hl_plan_str += " {0} - Cloth".format(cloth)

    for target in targets:
        hl_plan_str += " {0} - Target".format(target)

    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(cloth_locs, targets, baxter, failed_preds)

    goal_state = {}
    goal_str = "(:goal (and"
    for i in range(len(cloth_locs)):
        if i > len(cloth_locs) / 2:
            goal_str += " (ClothAtTarget cloth{0} left_target_1)".format(i)
            goal_state["(ClothAtTarget cloth{0} left_target_1)".format(i)] = True
        else:
            goal_str += " (ClothAtTarget cloth{0} right_target_1)".format(i)
            goal_state["(ClothAtTarget cloth{0} right_target_1)".format(i)] = True

    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state


def parse_initial_state(cloth_locs, targets, baxter, failed_preds=[]):
    hl_init_state = "(:init "

    for target in targets:
        if targets[target][1] >= -0.1:
            hl_init_state += " (TargetInReachLeft {0})".format(target)
        if targets[target][1] <= 0.1:
            hl_init_state += " (TargetInReachRight {0})".format(target)


    for cloth in cloth_locs:
        loc = cloth_locs[cloth]

        if loc[1] >= -0.1:
            hl_init_state += " (ClothInReachLeft {0})".format(cloth)
        if loc[1] <= 0.1:
            hl_init_state += " (ClothInReachRight {0})".format(cloth)

        closest_target = None
        closest_dist = np.inf
        for target in targets:
            dist = np.sum((targets[target] - loc)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target

        if closest_dist < 0.0025:
            hl_init_state += " (ClothAtTarget {0} {1})".format(cloth, closest_target)

        ee_data = baxter.openrave_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], 0)
        if np.all(np.abs(loc - ee_data['left_gripper']['pose']) < 0.03) and baxter.lGripper[:,0] < 0.016:
            hl_init_state += " (ClothInGripperLeft {0})".format(cloth)
        if np.all(np.abs(loc - ee_data['right_gripper']['pose']) < 0.03) and baxter.rGripper[:,0] < 0.016:
            hl_init_state += " (ClothInGripperRight {0})".format(cloth)

    hl_init_state += ")\n"
    print(hl_init_state)
    return hl_init_state


def get_hl_plan(prob, plan_id):
    with open(pddl_file, 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob, 'namo_{0}'.format(plan_id))

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

def hl_plan_for_state(state, targets, plan_id, param_map, state_inds, failed_preds=[]):
    cloth_locs = {}

    for param_name in param_map:
        param = param_map[param_name]
        if param_map[param_name]._type == 'Cloth':
            cloth_locs[param.name] = state[state_inds[param.name, 'pose']]

    prob, goal = get_sorting_problem(can_locs, targets, param_map['baxter'], failed_preds)
    hl_plan = get_hl_plan(prob, plan_id)
    if hl_plan == Plan.IMPOSSIBLE:
        # print 'Impossible HL plan for {0}'.format(prob)
        return []
    return parse_hl_plan(hl_plan)
