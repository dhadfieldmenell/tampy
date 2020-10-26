"""
Defines utility functions for planning in the sorting domain
"""
import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

from core.internal_repr.plan import Plan
from core.util_classes.namo_predicates import dsafe, GRIP_VAL
from core.util_classes.openrave_body import *
from pma.hl_solver import FFSolver
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils

NO_COL = True
NUM_OBJS = 4
NUM_TARGS = 4
SORT_CLOSET = False
USE_PERTURB = False
OPT_MCTS_FEEDBACK = True
N_GRASPS = 4
FIX_TARGETS = False

CONST_TARGETS = False
CONST_ORDER = False

# domain_file = "../domains/namo_domain/new_namo.domain"
domain_file = "../domains/namo_domain/nopose.domain"
mapping_file = "policy_hooks/namo/sorting_task_mapping_8"
pddl_file = "../domains/namo_domain/sorting_domain_3.pddl"

descriptor = 'namo_{0}_obj_sort_closet_{1}_perturb_{2}_feedback_to_tree_{3}'.format(NUM_OBJS, SORT_CLOSET, USE_PERTURB, OPT_MCTS_FEEDBACK)

# END_TARGETS = [(0., 5.8),
#            (0., 5.),
#            (0., 4.),
#            (2., -2.),
#            (0., -2.),
#            (4., 0.),
#            (-4, 0.),
#            (4., -2.),
#            (-4., -2.),
#            (-2., -2.)]

END_TARGETS =[(0., 5.8), (0., 5.), (0., 4.)] if SORT_CLOSET else []
END_TARGETS.extend([(7.5, -4.),
                   (-7.5, -4.),
                   (6, -5.5),
                   (-6, -5.5),
                   (3., 2.),
                   (-3, 2.),
                   (7.5, 1.),
                   (-7.5, 1.),
                   ])

n_aux = 4
possible_can_locs = [(0, 57), (0, 50), (0, 43), (0, 35)] if SORT_CLOSET else []
MAX_Y = 25
# possible_can_locs.extend(list(itertools.product(range(-45, 45, 2), range(-35, MAX_Y, 2))))

# possible_can_locs.extend(list(itertools.product(range(-50, 50, 4), range(-50, -10, 2))))
#possible_can_locs.extend(list(itertools.product(range(-50, 50, 4), range(-40, 0, 2))))
possible_can_locs.extend(list(itertools.product(list(range(-50, 50, 2)), list(range(-45, 0, 2)))))
# possible_can_locs.extend(list(itertools.product(range(-50, 50, 4), range(6, 25, 4))))


for i in range(len(possible_can_locs)):
    loc = list(possible_can_locs[i])
    loc[0] *= 0.1
    loc[1] *= 0.1
    possible_can_locs[i] = tuple(loc)


def prob_file(descr=None):
    if descr is None:
        descr = 'sort_closet_prob_{0}_{1}end_{2}_{3}aux'.format(NUM_OBJS, len(END_TARGETS), n_aux, N_GRASPS)
    return "../domains/namo_domain/namo_probs/{0}.prob".format(descr)


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(task_list)
    out[utils.OBJ_ENUM] = ['can{0}'.format(i) for i in range(NUM_OBJS)]
    out[utils.TARG_ENUM] = []
    for i in range(n_aux):
        out[utils.TARG_ENUM] += ['aux_target_{0}'.format(i)]
    if FIX_TARGETS:
        for i in range(len(END_TARGETS)):
            out[utils.TARG_ENUM] += ['end_target_{0}'.format(i)]
    else:
        out[utils.TARG_ENUM] += ['can{0}_end_target'.format(i) for i in range(NUM_OBJS)]
    out[utils.GRASP_ENUM] = ['grasp{0}'.format(i) for i in range(N_GRASPS)]
    return out


def get_vector(config):
    state_vector_include = {
        'pr2': ['pose', 'gripper'] ,
    }
    for i in range(config['num_objs']):
        state_vector_include['can{0}'.format(i)] = ['pose']

    action_vector_include = {
        'pr2': ['pose', 'gripper']
    }

    target_vector_include = {
        'can{0}_end_target'.format(i): ['value'] for i in range(config['num_objs'])
    }
    for i in range(n_aux):
        target_vector_include['aux_target_{0}'.format(i)] = ['value']
    if FIX_TARGETS:
        for i in range(len(END_TARGETS)):
            target_vector_include['end_target_{0}'.format(i)] = ['value']


    return state_vector_include, action_vector_include, target_vector_include


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
# Information is track by the environment
    x0s = []
    targ_maps = []
    for i in range(conditions):
        x0 = np.zeros((dX,))

        # x0[state_inds['pr2', 'pose']] = np.random.uniform([-3, -4], [3, -2]) # [0, -2]
        # x0[state_inds['pr2', 'pose']] = np.random.uniform([-3, -1], [3, 1])
        can_locs = copy.deepcopy(possible_can_locs)
        # can_locs = copy.deepcopy(END_TARGETS)
        locs = []
        pr2_loc = None
        spacing = 2.5
        valid = [1 for _ in range(len(can_locs))]
        while len(locs) < config['num_objs'] + 1:
            locs = []
            random.shuffle(can_locs)
            pr2_loc = can_locs[0]
            locs.append(pr2_loc)
            valid = [1 for _ in range(len(can_locs))]
            valid[0] = 0
            for m in range(1, len(can_locs)):
                if np.linalg.norm(np.array(pr2_loc) - np.array(can_locs[m])) < spacing:
                    valid[m] = 0
            for j in range(config['num_objs']):
                for n in range(1, len(can_locs)):
                    if valid[n]:
                        locs.append(can_locs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_locs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_locs[n]) - np.array(can_locs[m])) < spacing:
                                valid[m] = 0
                        break
            spacing -= 0.1

        spacing = 2.5
        targs = []
        can_targs = [can_locs[i] for i in range(len(can_locs)) if (valid[i] or not NO_COL)]
        old_valid = copy.copy(valid)
        while not FIX_TARGETS and len(targs) < config['num_targs']:
            targs = []
            pr2_loc = locs[0]
            random.shuffle(can_targs)
            valid = [1 for _ in range(len(can_targs))]
            for m in range(0, len(can_targs)):
                if np.linalg.norm(np.array(pr2_loc) - np.array(can_targs[m])) < spacing:
                    valid[m] = 0
            for j in range(config['num_targs']):
                for n in range(0, len(can_targs)):
                    if valid[n]:
                        targs.append(can_targs[n])
                        valid[n] = 0
                        for m in range(n+1, len(can_targs)):
                            if not valid[m]: continue
                            if np.linalg.norm(np.array(can_targs[n]) - np.array(can_targs[m])) < spacing:
                                valid[m] = 0
                        break

            spacing -= 0.1

        for l in range(len(locs)):
            locs[l] = np.array(locs[l])
        x0[state_inds['pr2', 'pose']] = locs[0]
        for o in range(config['num_objs']):
            x0[state_inds['can{0}'.format(o), 'pose']] = locs[o+1]
        x0[state_inds['pr2', 'gripper']] = -1.
        x0s.append(x0)
        if FIX_TARGETS:
            targ_range = list(range(config['num_objs'] - config['num_targs']))
            inds = list(range(len(EMD_TARGETS))) if CONST_TARGETS else np.random.permutation(list(range(len(END_TARGETS))))
            next_map = {'can{0}_end_target'.format(no): END_TARGETS[o] for no, o in enumerate(inds[:config['num_objs']])}
            inplace = targ_range if CONST_ORDER else np.random.choice(list(range(config['num_objs'])), len(targ_range), replace=False)
            for n in targ_range:
                x0[state_inds['can{0}'.format(inplace[n]), 'pose']] = END_TARGETS[inds[inplace[n]]]
            next_map.update({'end_target_{0}'.format(i): END_TARGETS[i] for i in range(len(END_TARGETS))})
        else:
            inds = np.random.permutation(list(range(config['num_objs'])))
            next_map = {'can{0}_end_target'.format(o): targs[no] for no, o in enumerate(inds[:config['num_targs']])}
            if config['num_targs'] < config['num_objs']:
                next_map.update({'can{0}_end_target'.format(o): locs[o+1] for o in inds[config['num_targs']:config['num_objs']]})
        for a in range(n_aux):
            if a == 0:
                next_map['aux_target_{0}'.format(a)] = (0, 0)
            elif a % 2:
                next_map['aux_target_{0}'.format(a)] = (-int(a/2.+0.5), 0)
            else:
                next_map['aux_target_{0}'.format(a)] = (int(a/2.+0.5), 0)
        targ_maps.append(next_map)
    return x0s, targ_maps

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

def get_plans(use_tf=False):
    tasks = get_tasks(mapping_file)
    task_ids = sorted(list(tasks.keys()))
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    params = None
    sess = None
    for task in tasks:
        next_task_str = copy.deepcopy(tasks[task])
        for i in range(len(prim_options[utils.OBJ_ENUM])):
            for j in range(len(prim_options[utils.TARG_ENUM])):
                for k in range(len(prim_options[utils.GRASP_ENUM])):
                    obj = prim_options[utils.OBJ_ENUM][i]
                    targ = prim_options[utils.TARG_ENUM][j]
                    grasp = prim_options[utils.GRASP_ENUM][k]
                    new_task_str = []
                    for step in next_task_str:
                        new_task_str.append(step.format(obj, targ, grasp))
                    plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)
                    plan.params['pr2'].gripper[0,0] = -GRIP_VAL
                    params = plan.params
                    plans[(task_ids.index(task), i, j, k)] = plan
                    if env is None:
                        env = plan.env
                        for param in list(plan.params.values()):
                            if hasattr(param, 'geom'):
                                if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                                    param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                                openrave_bodies[param.name] = param.openrave_body
                    sess = plan.sess

    return plans, openrave_bodies, env

def get_end_targets(num_cans=NUM_OBJS, num_targs=NUM_OBJS, targs=None, randomize=False, possible_locs=END_TARGETS):
    raise Exception('Bad method call')
    target_map = {}
    inds = list(range(NUM_TARGS)) # np.random.permutation(range(num_targs))
    for n in range(num_cans):
        if n > num_targs and targs is not None:
            target_map['can{0}_end_target'.format(n)] = np.array(targs[n])
        else:
            if randomize:
                ind = inds[n]
            else:
                ind = n

            target_map['can{0}_end_target'.format(n)] = np.array(possible_locs[ind])

    if SORT_CLOSET:
        target_map['middle_target'] = np.array([0., 0.])
        target_map['left_target_1'] = np.array([-1., 0.])
        target_map['right_target_1'] = np.array([1., 0.])
        # target_map['left_target_2'] = np.array([-2., 0.])
        # target_map['right_target_2'] = np.array([2., 0.])
    return target_map


def setup(hyperparams):
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

    self.main_camera_id = 0
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.75, 0.25, 1], [0.75, 0.5, 0, 1], [0.25, 0.25, 0.5, 1], [0.5, 0, 0.25, 1], [0, 0.5, 0.75, 1], [0, 0, 0.5, 1]]

    items = config['include_items']
    prim_options = get_prim_choices()
    for name in prim_options[OBJ_ENUM]:
        if name =='pr2': continue
        cur_color = colors.pop(0)
        items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.4, 1.), 'rgba': tuple(cur_color)})
        # items.append({'name': '{0}_end_target'.format(name), 'type': 'cylinder', 'is_fixed': False, 'pos': (10, 10, 0.5), 'dimensions': (0.8, 0.2), 'rgba': tuple(cur_color)})
    for i in range(len(wall_dims)):
        dim, next_trans = wall_dims[i]
        next_trans[0,3] -= 3.5
        next_dim = dim # [dim[1], dim[0], dim[2]]
        pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
        items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

    generate_xml(BASE_XML, ENV_XML, include_files=config['include_files'], include_items=config['include_items'])











# CODE FROM OLDER VERSION OF PROB FILE BELOW THIS



def get_sorting_problem(can_locs, targets, pr2, grasp, failed_preds=[]):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for can in can_locs:
        hl_plan_str += " {0} - Can".format(can)

    for target in targets:
        hl_plan_str += " {0} - Target".format(target)

    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(can_locs, targets, pr2, grasp, failed_preds)

    goal_state = {}
    goal_str = "(:goal (and"
    for i in range(len(can_locs)):
        goal_str += " (CanAtTarget can{0} can{1}_end_target)".format(i, i)
        goal_state["(CanAtTarget can{0} can{1}_end_target)".format(i, i)] = True

    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(can_locs, targets, pr2, grasp, failed_preds=[]):
    hl_init_state = "(:init "
    for can1 in can_locs:
        loc1 = can_locs[can1]
        t1 = targets[can1+'_end_target']
        if loc1[1] < 3.5: continue
        for can2 in can_locs:
            if can2 == can1: continue
            loc2 = can_locs[can2]
            t2 = targets[can2+'_end_target']
            if loc2[1] < 3.5: continue
            if loc1[1] < loc2[1]:
                hl_init_state += " (CanObstructs {0} {1})".format(can1, can2)
                hl_init_state += " (WaitingOnCan {0} {1})".format(can2, can1)
            else:
                hl_init_state += " (CanObstructs {0} {1})".format(can2, can1)
                hl_init_state += " (WaitingOnCan {0} {1})".format(can1, can2)

    for t1 in targets:
        loc1 = targets[t1]
        if loc1[1] < 3.5 or np.abs(loc1[0]) > 0.5: continue
        for t2 in targets:
            if t1 == t2: continue
            loc2 = targets[t2]
            if loc2[1] < 3.5 or np.abs(loc2[0]) > 0.5: continue
            if loc2[1] > loc1[1]:
                hl_init_state += " (InFront {0} {1})".format(t1, t2)



    for can in can_locs:
        loc = can_locs[can]

        hl_init_state += " (CanInReach {0})".format(can)

        closest_target = None
        closest_dist = np.inf
        for target in targets:
            if targets[target][1] > 3.5 \
               and np.abs(targets[target][0]) < 0.5 \
               and loc[1] > 3.5 \
               and loc[1] < targets[target][1] \
               and np.abs(loc[0]) < 0.5 \
               and target[3] != can[3]:
                hl_init_state += " (CanObstructsTarget {0} {1})".format(can, target)

            dist = np.sum((targets[target] - loc)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target

        if closest_dist < 0.001:
            hl_init_state += " (CanAtTarget {0} {1})".format(can, closest_target)

        if np.all(np.abs(loc - pr2 + grasp) < 0.2):
            hl_init_state += " (CanInGripper {0})".format(can)

        if np.all(np.abs(loc - pr2 + grasp) < 1.0):
            hl_init_state += " (NearCan {0})".format(can)

    # Only mark the closest obstruction; it needs to be cleared first.
    for pred in failed_preds:
        if pred[0].get_type().lower() == 'obstructs':
            if " (CanObstructsTarget {0} {1})".format(pred[0].c.name, pred[2].name) not in hl_init_state:
                if pred[0].c.name != pred[1].name:
                    hl_init_state += " (CanObstructs {0} {1})".format(pred[0].c.name, pred[1].name)
                hl_init_state += " (CanObstructsTarget {0} {1})".format(pred[0].c.name, pred[2].name)
                break

        if pred[0].get_type().lower() == 'obstructsholding':
            if " (CanObstructsTarget {0} {1})".format(pred[0].obstr.name, pred[2].name) not in hl_init_state:
                if pred[0].obstr.name != pred[1].name:
                    hl_init_state += " (CanObstructs {0} {1})".format(pred[0].obstr.name, pred[1].name)
                hl_init_state += " (CanObstructsTarget {0} {1})".format(pred[0].obstr.name, pred[2].name)
                break


    hl_init_state += ")\n"
    # print hl_init_state
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
    can_locs = {}

    for param_name in param_map:
        param = param_map[param_name]
        if param_map[param_name]._type == 'Can':
            can_locs[param.name] = state[state_inds[param.name, 'pose']]

    prob, goal = get_sorting_problem(can_locs, targets, state[state_inds['pr2', 'pose']], param_map['grasp0'].value[:,0], failed_preds)
    hl_plan = get_hl_plan(prob, plan_id)
    if hl_plan == Plan.IMPOSSIBLE:
        # print 'Impossible HL plan for {0}'.format(prob)
        return []
    return parse_hl_plan(hl_plan)

def get_ll_plan_str(hl_plan, num_cans):
    tasks = get_tasks(mapping_file)
    ll_plan_str = []
    actions_per_task = []
    last_pose = "ROBOT_INIT_POSE"
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        can = act_params[2].lower()
        if len(act_params) > 3:
            target = act_params[3].upper()
        else:
            target = "CAN{}_INIT_TARGET".format(can[-1])

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(can, target)
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
    plan = plan_from_str(ll_plan_str, prob_file().format(num_cans), domain_file, None, {})
    for i in range(len(can_locs)):
        plan.params['can{0}'.format(i)].pose[:,0] = can_locs[i]
        plan.params['can{0}_init_target'.format(i)].value[:,0] = plan.params['can{0}'.format(i)].pose[:,0]
        plan.params['can{0}_end_target'.format(i)].value[:, 0] = end_targets[i]

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
        if plan.params[param]._Type == "Can":
            next_pos = random.choice(possible_can_locs)
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_can_locations(num_cans):
    locs = []
    stop = False
    while not len(locs):
        locs = []
        for _ in range(num_cans):
            next_loc = random.choice(possible_can_locs)
            start = time.time()
            while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.6):
                next_loc = random.choice(possible_can_locs)
                if time.time() - start > 10:
                    locs = []
                    start = time.time()
                    stop = True
                    break

            if stop: break
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
        if param._type == 'Can':
            for target_name in targets:
                pred_list.append('CanAtTarget {0} {1}'.format(param_name, target_name))

    state_encoding = dict(list(zip(pred_list, list(range(len(pred_list))))))
    hl_state = np.zeros((len(pred_list)))
    for param_name in plan.params:
        if plan.params[param_name]._type != 'Can': continue
        for target_name in targets:
            if np.all(np.abs(state[plan.state_inds[param_name, 'pose']] - targets[target_name]) < 0.1):
                hl_state[state_encoding['CanAtTarget {0} {1}'.format(param_name, target_name)]] = 1

    if task[0] is not None:
        for target_name in targets:
            hl_state[state_encoding['CanAtTarget {0} {1}'.format(task[1], target_name)]] = 0
        hl_state[state_encoding['CanAtTarget {0} {1}'.format(task[1], task[2])]] = 1

    return tuple(hl_state)

def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
    tasks = get_tasks(mapping_file)
    next_task_str = copy.deepcopy(tasks[task])
    for j in range(len(next_task_str)):
        next_task_str[j]= next_task_str[j].format(*targets)

    return plan_from_str(next_task_str, prob_file().format(num_cans), domain_file, env, openrave_bodies)
