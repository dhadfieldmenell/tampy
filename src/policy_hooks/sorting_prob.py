"""
Defines utility functions for planning in the sorting domain
"""
import numpy as np

from pma.hl_solver import FFSolver
from policy_hooks.cloth_color_utils import get_cloth_color_mapping
from policy_hooks.load_task_definitions import get_tasks, plan_from_str

def get_sorting_problem(plan, color_map):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects blue_target white_target yellow_target green_target basket_start_target"
    for cloth in color_map:
        hl_plan_str += " {0}".format(cloth)
    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(plan)

    goal_str = "(:goal (and"
    for cloth in color_map:
        if color_map[cloth][0] == BLUE:
            goal_str += " (ClothAtLeftTarget {0} blue_target)"
        elif color_map[cloth][0] == WHITE:
            goal_str += " (ClothAtLeftTarget {0} white_target)"
        elif color_map[cloth][0] == YELLOW:
            goal_str += " (ClothAtRightTarget {0} yellow_target)"
        elif color_map[cloth][0] == GREEN:
            goal_str += " (ClothAtRightTarget {0} green_target)"
    goal_str += " (BasketAtTarget basket_start_target) "
    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_str

def parse_initial_state(plan):
    hl_init_state = "(and "
    blue_target = plan.params['blue_target']
    green_target = plan.params['green_target']
    white_target = plan.params['white_target']
    yellow_target = plan.params['yellow_target']
    for param in plan.params:
        if param._type == "Cloth":
            if param.pose[1,0] > 0:
                hl_init_state += " (ClothInLeftRegion {0})".format(param.name)
            else:
                hl_init_state += " (ClothInRightRegion {0})".format(param.name)

            for target in [blue_target, white_target]:
                if np.all(np.abs(target.value - param.pose[:,0]) < 0.03):
                    hl_init_state += " (ClothAtLeftTarget {0} {1})".format(param.name, target.name)
                else:
                    hl_init_state += " (not (ClothAtLeftTarget {0} {1}))".format(param.name, target.name)
            
            for target in [green_target, yellow_target]:
                if np.all(np.abs(target.value - param.pose[:,0]) < 0.03):
                    hl_init_state += " (ClothAtRightTarget {0} {1})".format(param.name, target.name)
                else:
                    hl_init_state += " (not (ClothAtRightTarget {0} {1}))".format(param.name, target.name)
    hl_init_state += " (BasketAtTarget basket_start_target)"
    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/laundry_domain/sorting_domain.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def get_ll_plan_str(hl_plan):
    tasks = get_tasks('sorting_task_mapping')
    ll_plan_str = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = tasks[act_params[1].lower()]
        cloth = act_params[2].lower()
        target = "blue_target"
        if len(act_params > 3):
            target = act_params[3]
        region = "left_region"
        if act_params[1].lower() == "move_cloth_to_right_region":
            region = "right_region"
        for j in range(len(next_task_str)):
            next_task_str[j].format(cloth, target, region, i)
        ll_plan_str.extend(next_task_str)
    return ll_plan_str

def get_plan(plan):
    cloths = []
    for param in plan.params:
        if param._Type == "Cloth":
            cloths.append(param)

    color_map = get_cloth_color_mapping(cloths)
    prob = get_sorting_problem(plan, color_map)
    hl_plan = get_hl_plan(prob)
    ll_plan_str = get_ll_plan_str(hl_plan)
    return plan_from_str(ll_plan_str)

def get_target_state_vector(plan, goal_state):
    state = np.zeros((plan.symbolic_bound, ))
    weights = np.zeros((plan.symbolic_bound, ))
    preds = goal_state.split('(')[3:-1]
    for i in range(len(preds)):
        preds[i] = preds[i].split()
        preds[i][-1] = preds[i][-1][:-1]

        if preds[i][0] == "ClothAtLeftTarget" or preds[i][0] == "ClothAtRightTarget":
            state[plan.state_inds[(preds[i][1], 'pose')]] = plan.params[preds[i][2]].value[:,0]
            weights[plan.state_inds[(preds[i][1], 'pose')]] = 1.0

def get_task_durations():
    tasks = get_tasks('sorting_task_mapping')
    durations = []
    for task in tasks:
        for i in range(len(task)):
            task[i].format('cloth0', 'blue_target', 'left_region', 0)
        plan = plan_from_str(task[i])
        durations.append(plan.horizon-1)
    return durations
