"""
Defines utility functions for planning in the sorting domain
"""
import copy
import numpy as np
import random

from pma.hl_solver import FFSolver
from policy_hooks.utils.cloth_color_utils import *
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str

possible_can_locs = [[-3, 2], [-1, 2], [1, 2], [3, 2]] 

targets = {
            'blue_target': [-3, -2],
            'green_target': [-1, -2], 
            'yellow_target':[1, -2],
            'white_target': [3, -2],
          }

def get_sorting_problem(can_locs, color_map):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for can in color_map:
        hl_plan_str += " {0} - can".format(can)
    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(can_locs)

    goal_state = {}
    goal_str = "(:goal (and"
    for can in color_map:
        if color_map[can][0] == BLUE:
            goal_str += " (CanAtTarget {0 blue_target})".format(can)
            goal_state["(CanAtTarget {0} blue_target)".format(can)] = True
        elif color_map[can][0] == WHITE:
            goal_str += " (CanAtTarget {0} white_target)".format(can)
            goal_state["(CanAtTarget {0} white_target)".format(can)] = True
        elif color_map[can][0] == YELLOW:
            goal_str += " (CanAtTarget {0} yellow_target)".format(can)
            goal_state["(CanAtTarget {0} yellow_target)".format(can)] = True
        elif color_map[can][0] == GREEN:
            goal_str += " (CanAtTarget {0} green_target)".format(can)
            goal_state["(CanAtTarget {0} green_target)".format(can)] = True
    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(can_locs):
    hl_init_state = "(:init "
    for i in range(len(can_locs)):
        loc = can_locs[i]
        if loc[1] > 0:
            hl_init_state += " (CanInLeftRegion Cloth{0})".format(i)
        else:
            hl_init_state += " (CanInRightRegion Cloth{0})".format(i)

        if np.all(np.abs(np.array(targets['blue_target']) - loc) < 0.03):
            hl_init_state += " (CanAtBlueTarget Cloth{0})".format(i)
        
        if np.all(np.abs(np.array(targets['green_target']) - loc) < 0.03):
            hl_init_state += " (CanAtGreenTarget Cloth{0})".format(i)
        
        if np.all(np.abs(np.array(targets['yellow_target']) - loc) < 0.03):
            hl_init_state += " (CanAtYellowTarget Cloth{0})".format(i)
        
        if np.all(np.abs(np.array(targets['white_target']) - loc) < 0.03):
            hl_init_state += " (CanAtWhiteTarget Cloth{0})".format(i)
        
    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/laundry_domain/sorting_domain_2.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def get_ll_plan_str(hl_plan, num_cans):
    tasks = get_tasks('policy_hooks/sorting_task_mapping_2')
    ll_plan_str = []
    actions_per_task = []
    last_pose = "ROBOT_INIT_POSE"
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        can = act_params[2].lower()
        if can in region_targets:
            final_target = region_targets[can] + "_OFF"
        else:
            final_target = "CAN_TARGET_BEGIN_{0}".format(can[-1])
        target = "BLUE_TARGET"
        if len(act_params) > 3:
            target = act_params[3].upper()

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(can[-1], target, i, last_pose, final_target)
        ll_plan_str.extend(next_task_str)
        actions_per_task.append((len(next_task_str), act_params[1].lower()))
        last_pose = "CAN_PUTDOWN_END_{0}".format(i)
    return ll_plan_str, actions_per_task

def get_plan(num_cans):
    cans = ["Can{0}".format(i) for i in range(num_cans)]
    color_map, colors = get_cloth_color_mapping(cans)
    can_locs = get_random_initial_can_locations(num_cans)
    print "\n\n", can_locs, "\n\n"
    prob, goal_state = get_sorting_problem(can_locs, color_map)
    hl_plan = get_hl_plan(prob)
    ll_plan_str, actions_per_task = get_ll_plan_str(hl_plan, num_cans)
    plan = plan_from_str(ll_plan_str, num_cans)
    for i in range(len(can_locs)):
        plan.params['can{0}'.format(i)].pose[:,0] = can_locs[i]
        plan.params['can_target_begin_{0}'.format(i)].value[:,0] = plan.params['can{0}'.format(i)].pose[:,0]

    task_timesteps = []
    cur_act = 0
    for i in range(len(hl_plan)):
        num_actions = actions_per_task[i][0]
        final_t = plan.actions[cur_act+num_actions-1].active_timesteps[1]
        task_timesteps.append((final_t, actions_per_task[i][1]))
        cur_act += num_actions

    plan.task_breaks = task_timesteps
    return plan, task_timesteps, color_map, goal_state

def get_target_state_vector(state_inds, goal_state, dX):
    state = np.zeros((dX, ))
    weights = np.zeros((dX, ))
    preds = goal_state.split('(')[3:-1]
    for i in range(len(preds)):
        preds[i] = preds[i].split()
        preds[i][-1] = preds[i][-1][:-1]

        if preds[i][0] == "CanAtBlueTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['blue_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "CanAtGreenTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['green_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "CanAtYellowTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['yellow_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "CanAtWhiteTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['white_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
    return state, weights

# def get_task_durations():
#     tasks = get_tasks('policy_hooks/sorting_task_mapping')
#     durations = []
#     for task in tasks:
#         for i in range(len(task)):
#             task[i].format('can0', 'blue_target', 'left_region', 0)
#         plan = plan_from_str(task[i])
#         durations.append(plan.horizon-1)
#     return durations

def fill_random_initial_configuration(plan):
    for param in plan.params:
        if plan.params[param]._Type == "Can":
            next_pos = random.choice(possible_can_locs)
            next_pos[1] *= random.choice([-1, 1])
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_can_locations(num_cans):
    locs = []
    for _ in range(num_cans):
        next_loc = random.choice(possible_can_locs)
        next_loc[1] *= random.choice([-1, 1])
        while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.08):
            next_loc = random.choice(possible_can_locs)
            next_loc[1] *= random.choice([-1, 1])

        locs.append(next_loc)

    def compare_locs(a, b):
        if b[0] > a[0]: return 1
        if b[0] < a[0]: return -1
        if b[1] > a[1]: return 1
        if b[1] < a[1]: return -1
        return 0

    locs.sort(compare_locs)
    
    return locs

def sorting_state_eval(x, state_inds, num_cans):
    hl_state = {}
    for can in range(num_cans):
        can_loc = np.array(x[state_inds['can{}'.format(can), 'pose']])
        if np.sqrt(np.sum((can_loc-targets['blue_target'])**2)) < 0.07:
            hl_state["(CanAtTarget can{0} blue_target)".format(can)]
        elif np.sqrt(np.sum((can_loc-targets['green_target'])**2)) < 0.07:
            hl_state["(CanAtTarget can{0} green_target)".format(can)]
        elif np.sqrt(np.sum((can_loc-targets['yellow_target'])**2)) < 0.07:
            hl_state["(CanAtTarget can{0} yellow_target)".format(can)]
        elif np.sqrt(np.sum((can_loc-targets['white_target'])**2)) < 0.07:
            hl_state["(CanAtTarget can{0})".format(can)]

    return hl_state
