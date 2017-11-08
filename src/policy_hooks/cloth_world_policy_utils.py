from core.parsing import parse_domain_config, parse_problem_config
from pma import hl_solver, robot_ll_solver

import numpy as np

import unittest, time, main


BASKET_POSE = [0.7, 0.35, 0.875]
BASKET_X_RANGE = [0.7, 0.85]
BASKET_Y_RANGE = [0.7, 0.8]
# CLOTH_INIT_X_RANGE = [-0.1, 0.9]
# CLOTH_INIT_Y_RANGE = [0.15, 1.15]
CLOTH_INIT_X_RANGE = [0.5, 0.9]
CLOTH_INIT_Y_RANGE = [-0.1, 0.5]
STEP_DELTA = 4
TABLE_POSE = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2] # XYZ
TABLE_TOP = 0.97 - 0.375 + 0.03
BASKET_HEIGHT_DELTA = 0.035

def generate_cond(num_cloths):
    i = 1
    act_num = 0 if num_cloths <= 1 else 4
    plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        '1: CLOTH_GRASP BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
        '3: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 INIT_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
    ]

    while i < num_cloths:
        plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i-1, i))
        act_num += 1
        plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, i))
        act_num += 1
        plan_str.append('{0}: CLOTH_PUTDOWN BAXTER CLOTH_{1} CLOTH_TARGET_END_{2}, CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        i += 1

    plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} ROBOT_END_POSE'.format(act_num, i-1))

    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_policy_{0}.prob'.format(num_cloths))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    # problem.init_state.params['cloth_target_begin_0'].value[:,0] = random_pose
    # problem.init_state.params['cloth_0'].pose[:,0] = random_pose

    plan = hls.get_plan(plan_str, domain, problem)

    basket = plan.params['basket']
    basket_target = plan.params['init_target']
    basket.pose[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket.rotation[:,:] = [[0], [0], [np.pi/2]]
    basket_target.value[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket_target.rotation[:,:] = [[0], [0], [np.pi/2]]

    plan.params['table'].pose[:,:] = np.array(TABLE_POSE).reshape(-1,1)
    plan.params['table'].rotation[:,:] = 0

    all_are_on_table = True # np.random.randInt(0, 2)
    on_table = num_cloths if all_are_on_table else np.random.randInt(0, num_cloths)

    possible_locs = np.random.choice(range(0, 10000, STEP_DELTA**2), on_table)

    for c in range(on_table-1, 0, -1):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 100) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 100) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        plan.params['cloth_{0}'.format(c)].pose[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_target_begin_{0}'.format(c)].value[:, 0] = [next_x, next_y, TABLE_TOP]

    possible_basket_locs = np.random.choice(range(0, 1800, STEP_DELTA**2), num_cloths-on_table)

    for c in range(num_cloths - on_table):
        next_x = (BASKET_POSE[0] - 0.15) + (possible_basket_locs[c] / 60) / 100.0
        next_y = (BASKET_POSE[1] - 0.3) + (possible_basket_locs[c] % 60) / 100.0
        plan.params['cloth_{0}'.format(c)].pose[:, 0] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        plan.params['cloth_target_begin_{0}'.format(c)].value[:, 0] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]

    for c in range(num_cloths):
        plan.params['cg_ee_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cg_ee_{0}'.format(c)].rotation[:,:] = np.nan
        
        plan.params['cp_ee_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cp_ee_{0}'.format(c)].rotation[:,:] = np.nan

        plan.params['cloth_grasp_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_grasp_end_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_putdown_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_putdown_end_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_target_end_{0}'.format(c)].value[:,:] = np.nan

    plan._determine_free_attrs()

    return plan

def get_randomized_initial_state(plan):
    num_cloths = 0
    while 'cloth_{0}'.format(num_cloths) in plan.params:
        num_cloths += 1

    X = np.zeros((plan.dX))

    basket = plan.params['basket']
    basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                        [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                        [BASKET_POSE[2]]]
    X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
    X[plan.state_inds[('basket', 'rotation')]] = [0, 0, np.pi/2]
    X[plan.state_inds[('table', 'pose')]] = TABLE_POSE
    X[plan.state_inds[('table', 'rotation')]] = [0, 0, 0]
    X[plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
    X[plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

    num_on_table = np.random.randint(1, num_cloths+1)
    possible_locs = np.random.choice(range(0, 40*60, STEP_DELTA**2), num_on_table).tolist()
    for c in range(num_cloths-1, num_on_table-2, -1):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 60) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 60) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

    possible_basket_locs = np.random.choice(range(0, 150, STEP_DELTA**2), num_cloths-num_on_table).tolist()

    stationary_params = ['basket', 'table']
    for c in range(num_cloths - num_on_table):
        next_x = (basket.pose[0,0] - 0.15) + (possible_basket_locs[c] / 10) / 100.0
        next_y = (basket.pose[1,0] - 0.3) + (possible_basket_locs[c] % 10) / 100.0
        X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        stationary_params.append('cloth_{0}'.format(c))

    discard_actions = (num_cloths - num_on_table) * 4
    return X, [discard_actions, len(plan.actions)-1], stationary_params
