"""
Defines utility functions for planning in the sorting domain
"""
import copy
import itertools
import numpy as np
import random

from core.internal_repr.plan import Plan
from core.util_classes.namo_predicates import dsafe
from pma.hl_solver import FFSolver
from policy_hooks.baxter.cloth_locs import cloth_locs as possible_cloth_locs
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from policy_hooks.utils.policy_solver_utils import *

TABLE_TOP = 0.655

prob_file = "../domains/laundry_domain/laundry_probs/baxter_policy_{0}.prob"
domain_file = "../domains/laundry_domain/laundry.domain"



def get_end_targets(num_cans):
    target_map = {}
    target_map['cloth0_end_target'] = [0.7, 0.5, TABLE_TOP]
    target_map['cloth1_end_target'] = [0.7, -0.5, TABLE_TOP]
    target_map['cloth2_end_target'] = [0.2, 0.8, TABLE_TOP]
    target_map['cloth3_end_target'] = [0.2, -0.8, TABLE_TOP]
    target_map['middle_target_1'] = [0.4, 0, TABLE_TOP]
    target_map['left_mid_target'] = [0.5, 0.2, TABLE_TOP]
    target_map['right_mid_target'] = [0.5, -0.2, TABLE_TOP]
    # target_map['middle_target_2'] = [0.45, 0, TABLE_TOP]
    return target_map

def get_random_initial_state_vec(num_cans, targets, dX, state_inds, num_vecs=1):
    Xs = np.zeros((num_vecs, dX), dtype='float32')
    keep = False

    for i in range(num_vecs):
        can_locs = get_random_initial_can_locations(num_cans)
        # while not keep:
        #     can_locs = get_random_initial_can_locations(num_cans)
        #     for i in range(num_cans):
        #         if can_locs[i][0] != targets['can{0}_end_target'.format(i)][0] or can_locs[i][1] != targets['can{0}_end_target'.format(i)][1]:
        #             keep = True

        for n in range(num_cans):
            Xs[i, state_inds['cloth{0}'.format(n), 'pose']] = can_locs[n]

    return [np.array(X) for X in Xs.tolist()]

def get_sorting_problem(can_locs, targets, robot, failed_preds=[]):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for can in can_locs:
        hl_plan_str += " {0} - Cloth".format(can)

    for target in targets:
        if 'middle' in target: continue
        if targets[target][1] >= 0:
            hl_plan_str += " {0} - LeftTarget".format(target)
        else:
            hl_plan_str += " {0} - RightTarget".format(target)

    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(can_locs, targets, robot, failed_preds)

    goal_state = {}
    goal_str = "(:goal (and"
    for i in range(len(can_locs)):
        if targets["cloth{0}_end_target".format(i)][1] >= 0:
            goal_str += " (ClothAtLeftTarget cloth{0} cloth{1}_end_target)".format(i, i)
            goal_state["(ClothAtLeftTarget cloth{0} cloth{1}_end_target)".format(i, i)] = True
        else:
            goal_str += " (ClothAtRightTarget cloth{0} cloth{1}_end_target)".format(i, i)
            goal_state["(ClothAtRightTarget cloth{0} cloth{1}_end_target)".format(i, i)] = True

    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(can_locs, targets, robot, failed_preds=[]):
    hl_init_state = "(:init "
    for can in can_locs:
        loc = can_locs[can]

        closest_target = None
        closest_dist = np.inf
        for target in targets:
            dist = np.sum((targets[target] - loc)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target

        if closest_dist < 0.01:
            if targets[closest_target][1] <= 0.1:
                if 'middle' not in closest_target:
                    hl_init_state += " (ClothAtRightTarget {0} {1})".format(can, closest_target)

            if targets[closest_target][1] >= -0.1:
                if 'middle' not in closest_target:
                    hl_init_state += " (ClothAtLeftTarget {0} {1})".format(can, closest_target)
            if 'middle' in closest_target:
                hl_init_state += " (ClothInMiddle {0})".format(can)

        if loc[1] <= 0.1:
            hl_init_state += " (ClothInRightRegion {0})".format(can)
        if loc[1] >= -0.1:
            hl_init_state += " (ClothInLeftRegion {0})".format(can)

    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/laundry_domain/sorting_domain.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob)

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

def hl_plan_for_state(state, targets, param_map, state_inds, failed_preds=[]):
    can_locs = {}

    for param_name in param_map:
        param = param_map[param_name]
        if param_map[param_name]._type == 'Cloth':
            can_locs[param.name] = state[state_inds[param.name, 'pose']]

    prob, goal = get_sorting_problem(can_locs, targets, param_map['baxter'], failed_preds)
    hl_plan = get_hl_plan(prob)
    if hl_plan == Plan.IMPOSSIBLE:
        import ipdb; ipdb. set_trace()
        return []
    return parse_hl_plan(hl_plan)

def get_ll_plan_str(hl_plan, num_cans):
    tasks = get_tasks('policy_hooks/baxter/sorting_task_mapping')
    ll_plan_str = []
    actions_per_task = []
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        can = act_params[2].lower()
        if len(act_params) > 3:
            target = act_params[3].upper()
        else:
            target = "{}_INIT_TARGET".format(can)

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(can, target)
        ll_plan_str.extend(next_task_str)
        actions_per_task.append((len(next_task_str), act_params[1].lower()))
    return ll_plan_str, actions_per_task

def get_plan(num_cans):
    cans = ["Cloth{0}".format(i) for i in range(num_cans)]
    can_locs = get_random_initial_can_locations(num_cans)
    end_targets = get_can_end_targets(num_cans)

    prob, goal_state = get_sorting_problem(can_locs, end_targets)
    hl_plan = get_hl_plan(prob)
    ll_plan_str, actions_per_task = get_ll_plan_str(hl_plan, num_cans)
    plan = plan_from_str(ll_plan_str, prob_file.format(num_cans), domain_file, None, {})
    for i in range(len(can_locs)):
        plan.params['cloth{0}'.format(i)].pose[:,0] = can_locs[i]
        plan.params['cloth{0}_init_target'.format(i)].value[:,0] = plan.params['can{0}'.format(i)].pose[:,0]
        plan.params['cloth{0}_end_target'.format(i)].value[:, 0] = end_targets[i]

    task_timesteps = []
    cur_act = 0
    for i in range(len(hl_plan)):
        num_actions = actions_per_task[i][0]
        final_t = plan.actions[cur_act+num_actions-1].active_timesteps[1]
        task_timesteps.append((final_t, actions_per_task[i][1]))
        cur_act += num_actions

    plan.task_breaks = task_timesteps
    return plan, task_timesteps, goal_state

def fill_random_initial_configuration(plan):
    for param in plan.params:
        if plan.params[param]._Type == "Cloth":
            next_pos = random.choice(possible_cloth_locs)
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_can_locations(num_cans):
    locs = []
    for _ in range(num_cans):
        next_loc = random.choice(possible_cloth_locs)
        while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.01):
            next_loc = random.choice(possible_cloth_locs)

        locs.append(next_loc)

    def compare_locs(a, b):
        if b[0] > a[0]: return 1
        if b[0] < a[0]: return -1
        if b[1] > a[1]: return 1
        if b[1] < a[1]: return -1
        return 0

    # locs.sort(compare_locs)

    return locs

def sorting_state_encode(state, plan, targets, task=(None, None, None)):
    pred_list = []
    for param_name in plan.params:
        param = plan.params[param_name]
        if param._type == 'Cloth':
            for target_name in targets:
                pred_list.append('ClothAtTarget {0} {1}'.format(param_name, target_name))
            pred_list.append('ClothInMiddle {0}'.format(param_name))
            pred_list.append('ClothInLeftRegion {0}'.format(param_name))
            pred_list.append('ClothInRightRegion {0}'.format(param_name))
    state_encoding = dict(list(zip(pred_list, list(range(len(pred_list))))))
    hl_state = np.zeros((len(pred_list)))
    for param_name in plan.params:
        if plan.params[param_name]._type != 'Cloth': continue
        for target_name in targets:
            if np.all(np.abs(state[plan.state_inds[param_name, 'pose']] - targets[target_name]) < 0.1):
                hl_state[state_encoding['ClothAtTarget {0} {1}'.format(param_name, target_name)]] = 1
                if 'middle' in target_name:
                    hl_state[state_encoding['ClothInMiddle {0}'.format(param_name)]] = 1
                if targets[target_name][1] <= 0.1:
                    hl_state[state_encoding['ClothInRightRegion {0}'.format(param_name)]] = 1
                if targets[target_name][1] >= -0.1:
                    hl_state[state_encoding['ClothInLeftRegion {0}'.format(param_name)]] = 1

    if task[0] is not None:
        if task[0] == 'move_cloth_to_left_region':
            for pred in state_encoding:
                if 'ClothAtTarget {0}'.format(task[1]) in pred:
                    hl_state[state_encoding[pred]] = 0.
            hl_state[state_encoding['ClothInMiddle {0}'.format(task[1])]] = 1.
            hl_state[state_encoding['ClothInLeftRegion {0}'.format(task[1])]] = 1.
            # hl_state[state_encoding['ClothAtTarget {0} {1}'.format(task[1], task[2])]] = 1.

        if task[0] == 'move_cloth_to_right_region':
            for pred in state_encoding:
                if 'ClothAtTarget {0}'.format(task[1]) in pred:
                    hl_state[state_encoding[pred]] = 0.
            hl_state[state_encoding['ClothInMiddle {0}'.format(task[1])]] = 1.
            hl_state[state_encoding['ClothInRightRegion {0}'.format(task[1])]] = 1.
            # hl_state[state_encoding['ClothAtTarget {0} {1}'.format(task[1], task[2])]] = 1.

        if task[0] == 'move_cloth_to_left_target':
            for pred in state_encoding:
                if 'ClothAtTarget {0}'.format(task[1]) in pred or 'ClothInRightRegion {0}'.format(task[1]) == pred:
                    hl_state[state_encoding[pred]] = 0.
            hl_state[state_encoding['ClothInMiddle {0}'.format(task[1])]] = 0.
            hl_state[state_encoding['ClothInLeftRegion {0}'.format(task[1])]] = 1.
            hl_state[state_encoding['ClothAtTarget {0} {1}'.format(task[1], task[2])]] = 1.

        if task[0] == 'move_cloth_to_right_target':
            for pred in state_encoding:
                if 'ClothAtTarget {0}'.format(task[1]) in pred or 'ClothInLeftRegion {0}'.format(task[1]) == pred:
                    hl_state[state_encoding[pred]] = 0.
            hl_state[state_encoding['ClothInMiddle {0}'.format(task[1])]] = 0.
            hl_state[state_encoding['ClothInRightRegion {0}'.format(task[1])]] = 1.
            hl_state[state_encoding['ClothAtTarget {0} {1}'.format(task[1], task[2])]] = 1.

    return tuple(hl_state)

def get_next_target(plan, state, task, target_poses, sample_traj=[], exclude=[]):
    state = np.array(state)
    robot_pose = state[plan.state_inds['baxter', 'pose']]
    if task == 'grasp':

        if len(sample_traj):
            robot_end_pose = sample_traj[-1, plan.state_inds['baxter', 'pose']]
            closest_dist = np.inf
            closest_can = None
            for param in list(plan.params.values()):

                if param._type != 'Cloth' or param.name in exclude: continue
                param_pose = state[plan.state_inds[param.name, 'pose']]
                target = target_poses['{0}_end_target'.format(param.name)]

                if np.sum((param_pose - target)**2) > 0.01 and np.sum((param_pose - robot_end_pose)**2) < closest_dist:
                    closest_dist = np.sum((param_pose - robot_end_pose)**2)
                    closest_can = param

            if closest_can is None:
                return None, None

            return closest_can, plan.params['{0}_end_target'.format(closest_can.name)]

        for param in list(plan.params.values()):
            if param._type != 'Can' or param.name in exclude: continue
            param_pose = state[plan.state_inds[param.name, 'pose']]
            if np.sum((param_pose - target_poses['{0}_end_target'.format(param.name)])**2) > 0.01:
                return param, plan.params['{0}_end_target'.format(param.name)]

        import ipdb; ipdb.set_trace()

    if task == 'putdown':
        target_occupied = False
        middle_occupied = False
        left_occupied = False
        right_occupied = False

        if len(sample_traj):
            robot_init_pose = sample_traj[0, plan.state_inds['baxter', 'pose']]
            robot_end_pose = sample_traj[-1, plan.state_inds['baxter', 'pose']]
            closest_dist = np.inf
            closest_can = None

            for param in list(plan.params.values()):

                if param._type != 'Can' or param.name in exclude: continue
                param_pose = state[plan.state_inds[param.name, 'pose']]

                if np.sum((param_pose - robot_end_pose)**2) < closest_dist and \
                   np.sum((param_pose - robot_init_pose) ** 2) < 1.:
                    closest_dist = np.sum((param_pose - robot_end_pose)**2)
                    closest_can = param

            if closest_can is not None:
                for param_2 in list(plan.params.values()):
                    if param_2._type != 'Can': continue

                    param_2_pose = state[plan.state_inds[param_2.name, 'pose']]
                    if param_2.name != param.name:
                        param_2_pose = state[plan.state_inds[param_2.name, 'pose']]
                        taget_occupied = target_occupied or np.sum((param_2_pose - plan.params['can{0}_end_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2

                    middle_occupied = middle_occupied or np.sum((param_2_pose - plan.params['middle_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2
                    left_occupied = left_occupied or np.sum((param_2_pose - plan.params['left_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2
                    right_occupied = right_occupied or np.sum((param_2_pose - plan.params['right_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2

                if target_occupied:
                    if not middle_occupied: return closest_can, plan.params['middle_target']
                    if not left_occupied: return closest_can, plan.params['left_target']
                    if not right_occupied: return closest_can, plan.params['right_target']
                    return None, None
                else:
                    return closest_can, plan.params['{0}_end_target'.format(closest_can.name)]

        for param in list(plan.params.values()):
            if param._type != 'Can' or param.name in exclude: continue
            param_pose = state[plan.state_inds[param.name, 'pose']]

            if np.all(np.abs(param_pose-robot_pose-[0, param.geom.radius+plan.params['baxter'].geom.radius+dsafe]) < 0.7):
                for param_2 in list(plan.params.values()):
                    if param_2._type != 'Can': continue

                    param_2_pose = state[plan.state_inds[param_2.name, 'pose']]
                    if param_2.name != param.name:
                        param_2_pose = state[plan.state_inds[param_2.name, 'pose']]
                        taget_occupied = target_occupied or np.sum((param_2_pose - plan.params['can{0}_end_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2

                    middle_occupied = middle_occupied or np.sum((param_2_pose - plan.params['middle_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2
                    left_occupied = left_occupied or np.sum((param_2_pose - plan.params['left_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2
                    right_occupied = right_occupied or np.sum((param_2_pose - plan.params['right_target'.format(param.name[-1])].value[:,0])**2) < param_2.geom.radius**2

                if target_occupied:
                    if not middle_occupied: return param, plan.params['middle_target']
                    if not left_occupied: return param, plan.params['left_target']
                    if not right_occupied: return param, plan.params['right_target']
                    continue
                else:
                    return param, plan.params['can{0}_end_target'.format(param.name[-1])]

    return None, None

def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
    tasks = get_tasks('policy_hooks/baxter/sorting_task_mapping')
    next_task_str = copy.deepcopy(tasks[task])
    for j in range(len(next_task_str)):
        next_task_str[j]= next_task_str[j].format(*targets)

    return plan_from_str(next_task_str, prob_file.format(num_cans), domain_file, env, openrave_bodies)

def cost_f(Xs, task, params, targets, plan, active_ts=None):
    tol = 1e-1

    if active_ts == None:
        active_ts = (1, plan.horizon-1)

    for t in range(active_ts[0], active_ts[1]):
        set_params_attrs(plan.params, plan.state_inds, Xs[t], t)

    for param in plan.params:
        if plan.params[param]._type == 'Cloth':
            plan.params['{0}_init_target'.format(param)].value[:,0] = plan.params[param].pose[:,0]
            plan.params['{0}_end_target'.format(param)].value[:,0] = targets['{0}_end_target'.format(param)]

    plan.params['robot_init_pose'].value[:,0] = plan.params['baxter'].pose[:,0]
    plan.params['robot_end_pose'].value[:,0] = plan.params['baxter'].pose[:,-1]
    plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
    plan.params['robot_end_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,-1]
    plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
    plan.params['robot_end_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,-1]
    plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
    plan.params['robot_end_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,-1]
    plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
    plan.params['robot_end_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,-1]
    plan.params['{0}_init_target'.format(params[0].name)].value[:,0] = plan.params[params[0].name].pose[:,0]
    plan.params['{0}_end_target'.format(params[0].name)].value[:,0] = targets['{0}_end_target'.format(params[0].name)]

    # failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
    try:
        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
    except:
        import ipdb; ipdb.set_trace()

    cost = 0
    for failed in failed_preds:
        if failed[1].get_type() == 'BaxterRobotAt': continue
        for t in range(active_ts[0], active_ts[1]+1):
            if t + failed[1].active_range[1] > active_ts[1]:
                break

            try:
                viol = np.max(failed[1].check_pred_violation(t, negated=failed[0], tol=tol))
                if viol is not None:
                    cost += viol
            except:
                import ipdb; ipdb.set_trace()

    return cost

def goal_f(X, targets, plan):
    cost = 0
    for param in list(plan.params.values()):
        if param._type == 'Cloth':
            dist = np.sum((X[plan.state_inds[param.name, 'pose']] - targets['{0}_end_target'.format(param.name)])**2)
            cost += dist if dist > 0.01 else 0
    return cost
