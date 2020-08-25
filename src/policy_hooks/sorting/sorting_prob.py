"""
Defines utility functions for planning in the sorting domain
"""
import copy
import numpy as np
import random

from pma.hl_solver import FFSolver
from policy_hooks.cloth_color_utils import *
from policy_hooks.cloth_locs import cloth_locs as possible_cloth_locs
from policy_hooks.load_task_definitions import get_tasks, plan_from_str

targets = {
            'blue_target': [0.65, 0.85, 0.65],
            'green_target': [0.3, 0.9, 0.65],
            'yellow_target': [0.3, -0.9, 0.65],
            'white_target': [0.65, -0.85, 0.65],
          }

## Uncomment these two functions for more general tasks (use with sorting_domain and sorting_task_mapping)
# def get_sorting_problem(cloth_locs, color_map):
#     hl_plan_str = "(define (problem sorting_problem)\n"
#     hl_plan_str += "(:domain sorting_domain)\n"

#     hl_plan_str += "(:objects blue_target white_target yellow_target green_target basket_start_target"
#     for cloth in color_map:
#         hl_plan_str += " {0}".format(cloth)
#     hl_plan_str += ")\n"

#     hl_plan_str += parse_initial_state(cloth_locs)

#     goal_str = "(:goal (and"
#     for cloth in color_map:
#         if color_map[cloth][0] == BLUE:
#             goal_str += " (ClothAtLeftTarget {0} blue_target)".format(cloth)
#         elif color_map[cloth][0] == WHITE:
#             goal_str += " (ClothAtRightTarget {0} white_target)".format(cloth)
#         elif color_map[cloth][0] == YELLOW:
#             goal_str += " (ClothAtRightTarget {0} yellow_target)".format(cloth)
#         elif color_map[cloth][0] == GREEN:
#             goal_str += " (ClothAtLeftTarget {0} green_target)".format(cloth)
#     goal_str += " (BasketAtTarget basket_start_target) "
#     goal_str += "))\n"

#     hl_plan_str += goal_str

#     hl_plan_str += "\n)"
#     return hl_plan_str, goal_str

# def parse_initial_state(cloth_locs):
#     hl_init_state = "(:init "
#     for i in range(len(cloth_locs)):
#         loc = cloth_locs[i]
#         if loc[1] > 0:
#             hl_init_state += " (ClothInLeftRegion Cloth{0})".format(i)
#         else:
#             hl_init_state += " (ClothInRightRegion Cloth{0})".format(i)

#         for target in ['blue_target', 'green_target']:
#             if np.all(np.abs(np.array(targets[target]) - loc) < 0.03):
#                 hl_init_state += " (ClothAtLeftTarget Cloth{0} {1})".format(i, target)
#             else:
#                 hl_init_state += " (not (ClothAtLeftTarget Cloth{0} {1}))".format(i, target)

#         for target in ['white_target', 'yellow_target']:
#             if np.all(np.abs(np.array(targets[target]) - loc) < 0.03):
#                 hl_init_state += " (ClothAtRightTarget Cloth{0} {1})".format(i, target)
#             else:
#                 hl_init_state += " (not (ClothAtRightTarget Cloth{0} {1}))".format(i, target)
#     hl_init_state += " (BasketAtTarget basket_start_target)"
#     hl_init_state += ")\n"
#     return hl_init_state


def get_sorting_problem(cloth_locs, color_map):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for cloth in color_map:
        hl_plan_str += " {0} - cloth".format(cloth)
    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(cloth_locs)

    goal_state = {}
    goal_str = "(:goal (and"
    for cloth in color_map:
        if color_map[cloth][0] == BLUE:
            goal_str += " (ClothAtBlueTarget {0})".format(cloth)
            goal_state["(ClothAtBlueTarget {0})".format(cloth)] = True
        elif color_map[cloth][0] == WHITE:
            goal_str += " (ClothAtWhiteTarget {0})".format(cloth)
            goal_state["(ClothAtWhiteTarget {0})".format(cloth)] = True
        elif color_map[cloth][0] == YELLOW:
            goal_str += " (ClothAtYellowTarget {0})".format(cloth)
            goal_state["(ClothAtYellowTarget {0})".format(cloth)] = True
        elif color_map[cloth][0] == GREEN:
            goal_str += " (ClothAtGreenTarget {0})".format(cloth)
            goal_state["(ClothAtGreenTarget {0})".format(cloth)] = True
    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(cloth_locs):
    hl_init_state = "(:init "
    for i in range(len(cloth_locs)):
        loc = cloth_locs[i]
        if loc[1] > 0:
            hl_init_state += " (ClothInLeftRegion Cloth{0})".format(i)
        else:
            hl_init_state += " (ClothInRightRegion Cloth{0})".format(i)

        if np.all(np.abs(np.array(targets['blue_target']) - loc) < 0.03):
            hl_init_state += " (ClothAtBlueTarget Cloth{0})".format(i)

        if np.all(np.abs(np.array(targets['green_target']) - loc) < 0.03):
            hl_init_state += " (ClothAtGreenTarget Cloth{0})".format(i)

        if np.all(np.abs(np.array(targets['yellow_target']) - loc) < 0.03):
            hl_init_state += " (ClothAtYellowTarget Cloth{0})".format(i)

        if np.all(np.abs(np.array(targets['white_target']) - loc) < 0.03):
            hl_init_state += " (ClothAtWhiteTarget Cloth{0})".format(i)

    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/laundry_domain/sorting_domain_2.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def get_ll_plan_str(hl_plan, num_cloths):
    tasks = get_tasks('policy_hooks/sorting_task_mapping_2')
    ll_plan_str = []
    actions_per_task = []
    last_pose = "ROBOT_INIT_POSE"
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        cloth = act_params[2].lower()
        if cloth in region_targets:
            final_target = region_targets[cloth] + "_OFF"
        else:
            final_target = "CLOTH_TARGET_BEGIN_{0}".format(cloth[-1])
        target = "BLUE_TARGET"
        if len(act_params) > 3:
            target = act_params[3].upper()
        region_target = "LEFT_CLOTH_TARGET_{0}".format(0)
        if act_params[1].lower() == "move_cloth_to_right_region":
            region_target = "RIGHT_CLOTH_TARGET_{0}".format(0)

        if act_params[1].lower() in ["move_cloth_to_right_region", "move_cloth_to_left_region"]:
            region_targets[cloth] = region_target

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(cloth[-1], target, region_target, i, last_pose, final_target)
        ll_plan_str.extend(next_task_str)
        actions_per_task.append((len(next_task_str), act_params[1].lower()))
        last_pose = "CLOTH_PUTDOWN_END_{0}".format(i)
    return ll_plan_str, actions_per_task

def get_plan(num_cloths):
    cloths = ["Cloth{0}".format(i) for i in range(num_cloths)]
    color_map, colors = get_cloth_color_mapping(cloths)
    cloth_locs = get_random_initial_cloth_locations(num_cloths)
    print("\n\n", cloth_locs, "\n\n")
    prob, goal_state = get_sorting_problem(cloth_locs, color_map)
    hl_plan = get_hl_plan(prob)
    ll_plan_str, actions_per_task = get_ll_plan_str(hl_plan, num_cloths)
    plan = plan_from_str(ll_plan_str, num_cloths)
    for i in range(len(cloth_locs)):
        plan.params['cloth{0}'.format(i)].pose[:,0] = cloth_locs[i]
        plan.params['cloth_target_begin_{0}'.format(i)].value[:,0] = plan.params['cloth{0}'.format(i)].pose[:,0]

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

        if preds[i][0] == "ClothAtBlueTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['blue_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "ClothAtGreenTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['green_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "ClothAtYellowTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['yellow_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
        if preds[i][0] == "ClothAtWhiteTarget":
            target = preds[i][2]
            state[state_inds[(preds[i][1], 'pose')]] = targets['white_target']
            weights[state_inds[(preds[i][1], 'pose')]] = 1.0
    return state, weights

# def get_task_durations():
#     tasks = get_tasks('policy_hooks/sorting_task_mapping')
#     durations = []
#     for task in tasks:
#         for i in range(len(task)):
#             task[i].format('cloth0', 'blue_target', 'left_region', 0)
#         plan = plan_from_str(task[i])
#         durations.append(plan.horizon-1)
#     return durations

def fill_random_initial_configuration(plan):
    for param in plan.params:
        if plan.params[param]._Type == "Cloth":
            next_pos = random.choice(possible_cloth_locs)
            next_pos[1] *= random.choice([-1, 1])
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_cloth_locations(num_cloths):
    locs = []
    for _ in range(num_cloths):
        next_loc = random.choice(possible_cloth_locs)
        next_loc[1] *= random.choice([-1, 1])
        while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.08):
            next_loc = random.choice(possible_cloth_locs)
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

def sorting_state_eval(x, state_inds, num_cloths):
    hl_state = {}
    for cloth in range(num_cloths):
        cloth_loc = np.array(x[state_inds['cloth{}'.format(cloth), 'pose']])
        if np.sqrt(np.sum((cloth_loc-targets['blue_target'])**2)) < 0.07:
            hl_state["(ClothAtBlueTarget cloth{0})".format(cloth)]
        elif np.sqrt(np.sum((cloth_loc-targets['green_target'])**2)) < 0.07:
            hl_state["(ClothAtGreenTarget cloth{0})".format(cloth)]
        elif np.sqrt(np.sum((cloth_loc-targets['yellow_target'])**2)) < 0.07:
            hl_state["(ClothAtYellowTarget cloth{0})".format(cloth)]
        elif np.sqrt(np.sum((cloth_loc-targets['white_target'])**2)) < 0.07:
            hl_state["(ClothAtWhiteTarget cloth{0})".format(cloth)]

    return hl_state
