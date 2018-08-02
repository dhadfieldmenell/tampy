"""
Defines utility functions for planning in the sorting domain
"""
import copy
import numpy as np
import random

from pma.hl_solver import FFSolver
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str

possible_can_locs = [[-3, 2], [-1, 2], [1, 2], [3, 2]] 


prob_file = "../domains/namo_domain/sort_prob_{0}.prob"
domain_file = "../domains/namo_domain/namo.domain"


def get_end_targets(num_cans):
    pass

def get_sorting_problem(can_locs, end_targets):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for can in color_map:
        hl_plan_str += " {0} - Can".format(can)

    for target in targets:
        hl_plan_str += " {0} - Target".format(target)

    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(can_locs)

    goal_state = {}
    goal_str = "(:goal (and"
    for i range(len(can_locs)):
        goal_str += " (CanAtTarget can{0} can{1}_end_target)".format(i, i)
        goal_state["(CanAtTarget can{0} can{1}_end_target)".format(i, i)] = True

    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(can_locs, targets):
    hl_init_state = "(:init "
    for i in range(len(can_locs)):
        loc = can_locs[i]

        closest_target = np.argmin(np.sum((np.array(targets) - loc)**2, axis=1))
        if (targets[closest_target] - loc)**2 < 0.001:
            hl_init_state += " (CanAtTarget can{0} can{1}_end_target)".format(i, closest_target)

    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/namo_domain/sorting_domain.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def get_ll_plan_str(hl_plan, num_cans):
    tasks = get_tasks('policy_hooks/namo/sorting_task_mapping')
    ll_plan_str = []
    actions_per_task = []
    last_pose = "ROBOT_INIT_POSE"
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        can = act_params[2].lower()
        if len(act_params) > 1:
            target = act_params[1].upper()
        else:
            target = "CAN{}_INIT_TARGET".format(can[-1])

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(can[-1], target)
        ll_plan_str.extend(next_task_str)
        actions_per_task.append((len(next_task_str), act_params[1].lower()))
    return ll_plan_str, actions_per_task

def get_plan(num_cans):
    cans = ["Can{0}".format(i) for i in range(num_cans)]
    can_locs = get_random_initial_can_locations(num_cans)
    end_targets = get_can_end_targets(num_cans)
    prob, goal_state = get_sorting_problem(can_locs, end_targets)
    hl_plan = get_hl_plan(prob)
    ll_plan_str, actions_per_task = get_ll_plan_str(hl_plan, num_cans)
    plan = plan_from_str(ll_plan_str, prob_file.format(num_cans), domain_file, None, {})
    for i in range(len(can_locs)):
        plan.params['can{0}'.format(i)].pose[:,0] = can_locs[i]
        plan.params['can{0}_init_target'.format(i)].value[:,0] = plan.params['can{0}'.format(i)].pose[:,0]
        plan.params['can{0}_end_target'format(i)].value[:, 0] = end_targets[i]

    task_timesteps = []
    cur_act = 0
    for i in range(len(hl_plan)):
        num_actions = actions_per_task[i][0]
        final_t = plan.actions[cur_act+num_actions-1].active_timesteps[1]
        task_timesteps.append((final_t, actions_per_task[i][1]))
        cur_act += num_actions

    plan.task_breaks = task_timesteps
    return plan, task_timesteps, goal_state

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
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_can_locations(num_cans):
    locs = []
    for _ in range(num_cans):
        next_loc = random.choice(possible_can_locs)
        while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.2):
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
        if np.sqrt(np.sum((can_loc-targets['blue_target'])**2)) < 0.001:
            hl_state["(CanAtTarget can{0} blue_target)".format(can)] = True
        else:
            hl_state["(CanAtTarget can{0} blue_target)".format(can)] = False

        if np.sqrt(np.sum((can_loc-targets['green_target'])**2)) < 0.001:
            hl_state["(CanAtTarget can{0} green_target)".format(can)] = True
        else:
            hl_state["(CanAtTarget can{0} green_target)".format(can)] = False

        if np.sqrt(np.sum((can_loc-targets['yellow_target'])**2)) < 0.001:
            hl_state["(CanAtTarget can{0} yellow_target)".format(can)] = True
        else:
            hl_state["(CanAtTarget can{0} yellow_target)".format(can)] = False

        if np.sqrt(np.sum((can_loc-targets['white_target'])**2)) < 0.001:
            hl_state["(CanAtTarget can{0} white_target)".format(can)] = True
        else:
            hl_state["(CanAtTarget can{0} white_target)".format(can)] = False

        if np.sqrt(np.sum((x[state_inds['pr2', 'pose']] - can_loc)**2)) < 0.001:
            hl_state["(CanInGripper can{0})".format(can)] = True
        else:
            hl_state["(CanInGripper can{0})".format(can)] = False
    return hl_state

def get_next_target(plan, state, task):
    robot_pose = state[plan.state_inds['PR2', 'pose']]
    target_poses = {name: state[plan.state_inds[name, 'value']] for name in targets}

    if task == 'grasp':
        for can in range(plan.num_cans):
            param = plan.params['can{0}'.format(can)]
            param_pose = state[plan.state_inds[param.name, 'pose']]
            if np.sum((param_pose - target_poses['{0}_target'.format(param.color)])**2) > 0.0001:
                return param, plan.params['{0}_target'.format(param.color)]
    
    if task == 'putdown':
        target_occupied = False
        middle_occupied = False
        for param in plan.params.values():
            param_pose = state[plan.state_inds[param.name, 'pose']]
            if np.sum((param_pose-robot_pose)**2) < 0.0001:
                for param_2 in plan.params.values():
                    if param_2.name != param.name:
                        param_2_pose = state[plan.state_inds[param_2.name, 'pose']]
                        taget_occupied = target_occupied or (param_2_pose - plan.params['can{0}_end_target'.format(param.name[-1])].value[:,0])**2 < 0.0001
                    middle_occupied = middle_occupied or (param_2_pose - plan.params['middle_target'.format(param.name[-1])].value[:,0])**2 < 0.0001
                if target_occupied:
                    if middle_occupied: return None, None

                    return param, plan.params['middle_target']

                else:
                    return param, plan.params['can{0}_end_target'.format(param.name[-1])]

    return None, None

def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
    tasks = get_tasks('policy_hooks/namo/sorting_task_mapping')
    plan_str = copy.deepcopy(tasks[task])
    for j in range(len(next_task_str)):
        next_task_str[j]= next_task_str[j].format(*targets)

    return plan_from_str(next_task_str, prob_file.format(num_cans), domain_file, env, openrave_bodies)
    
