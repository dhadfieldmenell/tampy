from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.baxter_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
from pma import hl_solver, robot_ll_solver

import numpy as np

import unittest, time, main, random


# BASKET_POSE = [0.7, 0.35, 0.875]
# BASKET_X_RANGE = [0.7, 0.85]
# BASKET_Y_RANGE = [0.7, 0.8]
# CLOTH_INIT_X_RANGE = [0.5, 0.9]
# CLOTH_INIT_Y_RANGE = [-0.15, 0.45]

BASKET_POSE = [0.65, 0.1, 0.875]
BASKET_X_RANGE = [0.65, 0.75]
BASKET_Y_RANGE = [-0.025, 0.075]
CLOTH_INIT_X_RANGE = [0.3, 0.7]
CLOTH_INIT_Y_RANGE = [0.5, 0.9]
CLOTH_XY = 40

STEP_DELTA = 3
BASKET_STEP_DELTA = 2
TABLE_POSE = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2] # XYZ
TABLE_TOP = 0.97 - 0.375 + .02
BASKET_HEIGHT_DELTA = 0.03 # 0.02

R_ARM_PUTDOWN_END = [0, -0.25, 0, 0, 0, 0, 0]
L_ARM_PUTDOWN_END = [-0.6, -1.49792454, -0.35878011, 1.63006026, 0.02577696, 1.44332767, -0.17578484] # [-1., -1.11049898, -0.29706795, 1.29338713, 0.13218013, 1.40690655, -0.50397199]

R_ARM_INIT = [0, -0.25, 0, 0, 0, 0, 0]
L_ARM_INIT = [-0.6, -1.49792454, -0.35878011, 1.63006026, 0.02577696, 1.44332767, -0.17578484]

# R_ARM_POSE = [-1.1, -0.7542201, 0.33281928, 0.81198007, 2.90107491, -1.54208563, 1.49564886]
R_ARM_POSE = [0, -1.14364607, -0.09527073, 0.71183977, 0.04343258, 2.00395428, -0.05391584]

# CLOTH_GRASP_END_LEFT = [-0.2, -1.10133617, -0.08624186, 1.05644701, 0.0390197, 1.61715084, 0.51025842]
# CLOTH_PUTDOWN_END_LEFT = [-0.7, -1.38083508, -0.01885368, 1.47083944, 0.00357422, 1.48082549, 0.06656424]

# CLOTH_GRASP_END_LEFT = [-0.5, -1.22381681, 0.15934068, 1.38479689, -0.05465265, 1.41410374, 0.44392776]
# CLOTH_GRASP_END_LEFT = [-0.8, -1.15478705, 0.61254287, 1.35995126, -0.23659335, 1.43852837, 0.58846723]
CLOTH_GRASP_END_LEFT = [-0.7, -1.20680768, 0.46547125, 1.26255527, -0.16050953, 1.55116649, 0.52742617]
CLOTH_PUTDOWN_END_LEFT = [-0.5, -1.28474597, -0.65578952, 1.40406676, 0.17325, 1.50943879, -0.36118326]

FOUR_CLOTH_LOCATIONS = [
    [[ 0.68 ,  0.39 ,  0.615], [ 0.68 ,  0.75 ,  0.615], [ 0.74 ,  0.63 ,  0.615], [ 0.58 ,  0.71 ,  0.615]],
    [[0.4, 0.6, 0.615], [0.5, 0.4, 0.615], [0.53, 0.7, 0.615], [0.62, 0.42, 0.615]],
    [[0.35, 0.7, 0.615], [0.48, 0.6, 0.615], [0.55, 0.4, 0.615], [0.6, 0.45, 0.615]],
    [[0.35, 0.55, 0.615], [0.41, 0.8, 0.615], [0.43, 0.67, 0.615], [0.59, 0.47, 0.615]],
    [[0.36, 0.7, 0.615], [0.4, 0.68, 0.615], [0.47, 0.46, 0.615], [0.8, 0.7, 0.615]],
    [[0.3, 0.9, 0.615], [0.33, 0.6, 0.615], [0.4, 0.7, 0.615], [0.46, 0.6, 0.615]]
]

FOUR_CLOTH_BASKET_LOCATIONS = [
    [[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]]
]


def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    cur_arm_pose = np.array(cur_arm_pose).flatten()
    for arm_pose in arm_poses:
        change = np.sum((np.array([1.75, 1.75, 2, 1.5, 2, 1, 1]) * (arm_pose - cur_arm_pose))**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

def generate_cond(num_cloths):
    i = 1
    act_num = 4
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
        plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} INIT_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        i += 1

    plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} ROBOT_END_POSE'.format(act_num, i-1))

    domain_fname = '../domains/laundry_domain/laundry_policy.domain'
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

    plan.params['robot_init_pose'].lArmPose[:,0] = L_ARM_INIT # L_ARM_PUTDOWN_END # [-0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].lGripper[:,0] = 0.02
    plan.params['robot_init_pose'].rArmPose[:,0] = R_ARM_PUTDOWN_END # [0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].rGripper[:,0] = 0.02

    plan.params['robot_end_pose'].lArmPose[:,0] = [1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].lGripper[:,0] = 0
    plan.params['robot_end_pose'].rArmPose[:,0] = [-1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].rGripper[:,0] = 0.015

    possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA**2)), num_cloths).tolist()

    for c in range(num_cloths):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 45) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 45) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        plan.params['cloth_{0}'.format(c)].pose[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_{0}'.format(c)].rotation[:, :] = 0
        plan.params['cloth_target_begin_{0}'.format(c)].value[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_target_begin_{0}'.format(c)].rotation[:, :] = 0

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

        plan.params['cloth_grasp_end_{0}'.format(c)].value[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].lArmPose[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].lGripper[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].rArmPose[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].rGripper[:,:] = 0

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
        # plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.array(L_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = 0.02
        # plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.array(R_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = 0.02

        plan.params['cloth_target_end_{0}'.format(c)].value[:,:] = 0
        plan.params['cloth_target_end_{0}'.format(c)].rotation[:,:] = 0

    plan._determine_free_attrs()

    return plan


def get_randomized_initial_state_left_pick_place_split(plan):
    num_cloths = 1
    # while 'cloth_{0}'.format(num_cloths) in plan.params:
    #     num_cloths += 1

    X = np.zeros((plan.dX))
    state_config = []

    success = False
    while not success:
        print('Searching for initial configuration...')
        mode = 0 if np.random.choice([0,1]) else 1
        X_0s = []

        X[plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
        X[plan.state_inds[('robot_end_pose', 'rArmPose')]] = np.zeros((7,))
        X[plan.state_inds[('robot_end_pose', 'rGripper')]] = 0

        # plan.params['baxter'].openrave_body.set_dof({'lArmPose': plan.params['robot_init_pose'].lArmPose.flatten(),
        #                                              'lGripper': plan.params['robot_init_pose'].lGripper.flatten()})

        # ee_trans = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
        # X[plan.state_inds[('baxter', 'ee_left_pos')]] = ee_trans[:3,3]
        # X[plan.state_inds[('baxter', 'ee_left_rot')]] = OpenRAVEBody._ypr_from_rot_matrix(ee_trans[:3,:3])

        X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = np.zeros((7,))
        X[plan.state_inds[('robot_init_pose', 'rGripper')]] = 0

        if not mode:
            X[plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
            X[plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper

        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        X[plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA)), num_cloths).tolist()
        possible_basket_locs = np.random.choice(list(range(0, 144, BASKET_STEP_DELTA)), num_cloths).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])
        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])
        for c in range(num_cloths-1, -1, -1):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 45) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 45) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.075], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            next_x = (basket.pose[0,0] - 0.12) + (possible_basket_locs[c] / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (possible_basket_locs[c] % 24) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [basket.pose[0,0], basket.pose[1,0], TABLE_TOP+BASKET_HEIGHT_DELTA]# [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]

            # arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.075], [0, np.pi/2, 0], "left_arm")
            # if not len(arm_poses): success = False

            height = np.random.uniform(0.175, 0.25)
            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose(X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] + [-0.025, 0.025, height], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses):
                success = False
                continue

            X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'lArmPose')]] = arm_poses[0]
            X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'lGripper')]] = 0.015
            X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'rArmPose')]] = np.zeros((7,))
            X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'rGripper')]] = 0.015

            if mode:
                X[plan.state_inds[('baxter', 'lArmPose')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'lArmPose')]]
                X[plan.state_inds[('baxter', 'lGripper')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(c), 'lGripper')]]
                X[plan.state_inds[('baxter', 'rGripper')]] = 0.015

            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lGripper')]] = 0.02
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rArmPose')]] = np.zeros((7,))
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rGripper')]] = 0.0

        num_on_table = np.random.randint(1, num_cloths + 1)

        for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
            if not mode:
                X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]
            else:
                X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] + [-0.025, 0.025, height]

        stationary_params = ['basket']
        for c in range(num_cloths-num_on_table):
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        plan_idx = num_cloths - num_on_table
        actions = [0,1] if not mode else [2,3]
        state_config = [X, actions, stationary_params]

    print("Found initial configuration.\n")
    return state_config















def get_random_initial_cloth_pick_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []

    success = False
    num_cloths_on_table = np.random.randint(1, num_cloths+1)
    num_cloths_in_basket = num_cloths - num_cloths_on_table
    next_cloth = num_cloths_in_basket
    actions = [4*next_cloth, 4*next_cloth+3]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    regions = [[[0.3, 0.5], [0.65, 0.85]], [[0.65, 0.85], [0.45, 0.7]], [[0.25, 0.45], [0.9, 1.0]], [[0.5, 0.65], [0.75, 0.95]]]

    while not success:
        print('Searching for initial configuration...')
        X_0s = []
        joint_angles = []
        stationary_params = ['basket']
        basket = plan.params['basket']
        # basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
        #                     [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
        #                     [BASKET_POSE[2]]]
        basket.pose[:,0] = [0.7, 0.025, 0.875]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [np.pi/2, 0, np.pi/2]

        possible_locs = np.sort(np.random.choice(list(range(0, CLOTH_XY**2, STEP_DELTA)), num_cloths_on_table, False)).tolist()
        possible_basket_locs = np.sort(np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket, False)).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])

        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])

        for c in range(num_cloths_in_basket, num_cloths):
            next_x = np.random.uniform(regions[c % 4][0][0], regions[c % 4][0][1])
            next_y = np.random.uniform(regions[c % 4][1][0], regions[c % 4][1][1])
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        target_cloth = X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]]
        grasp_pose = plan.params['baxter'].openrave_body.get_ik_from_pose([target_cloth[0], target_cloth[1], target_cloth[2]+0.1], [0, np.pi/2, 0], "left_arm")
        if not len(grasp_pose):
            success = False
            continue

        for c in range(0, num_cloths_in_basket):
            next_loc = possible_basket_locs.pop(0)
            next_x = (basket.pose[0,0] - 0.03) + (next_loc / 12) / 100.0
            next_y = (basket.pose[1,0] + 0.06) + (next_loc % 12) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        r_arm_pose = R_ARM_POSE

        X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'rGripper')]] = plan.params['robot_init_pose'].rGripper
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]] = CLOTH_GRASP_END_LEFT
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        if not next_cloth:
            plan.params['baxter'].openrave_body.set_dof({'lArmPose': plan.params['robot_init_pose'].lArmPose.flatten(),
                                                         'lGripper': plan.params['robot_init_pose'].lGripper.flatten(),
                                                         'rArmPose': plan.params['robot_init_pose'].rArmPose.flatten(),
                                                         'rGripper': plan.params['robot_init_pose'].rGripper.flatten()})
            if ('baxter', 'lArmPose') in plan.state_inds:
                X[plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
            elif ('baxter', 'ee_left_pos') in plan.state_inds:
                X[plan.state_inds['baxter', 'ee_left_pos']] = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()[:3,3]
            # X[plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
            if ('baxter', 'rArmPose') in plan.state_inds:
                X[plan.state_inds[('baxter', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
            elif ('baxter', 'ee_right_pos') in plan.state_inds:
                X[plan.state_inds['baxter', 'ee_right_pos']] = plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransform()[:3,3]
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            # X[plan.state_inds[('baxter', 'rGripper')]] = plan.params['robot_init_pose'].rGripper

            # Joint angles ordered by Mujoco Model joint order
            joint_angles = np.r_[0, X[plan.state_inds[('robot_init_pose', 'rArmPose')]], \
                                 X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                                 -X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                                 X[plan.state_inds[('robot_init_pose', 'lArmPose')]], \
                                 X[plan.state_inds[('robot_init_pose', 'lGripper')]], \
                                 -X[plan.state_inds[('robot_init_pose', 'lGripper')]]]
        else:
            next_x = np.random.uniform(0.6, 0.8)
            next_y = np.random.uniform(0.1, 0.4)
            height = np.random.uniform(0.25, 0.5)
            l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
            if not len(l_arm_poses):
                success = False
                continue

            # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
            # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
            # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lArmPose')]] = closest_arm_pose(l_arm_poses, CLOTH_PUTDOWN_END_LEFT)
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rArmPose')]] = r_arm_pose
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lArmPose')]] = closest_arm_pose(l_arm_poses, CLOTH_PUTDOWN_END_LEFT)
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            if ('baxter', 'lArmPose') in plan.state_inds:
                X[plan.state_inds[('baxter', 'lArmPose')]] = X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lArmPose')]]
            elif ('baxter', 'ee_left_pos') in plan.state_inds:
                # X[plan.state_inds[('baxter', 'ee_left_pos')]] = [next_x, next_y, TABLE_TOP+height]
                X[plan.state_inds[('baxter', 'ee_left_pos')]] = [next_x, next_y, 0.95]
            # X[plan.state_inds[('baxter', 'lGripper')]] = X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]]

            # Joint angles ordered by Mujoco Model joint order
            joint_angles = np.r_[0, X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rArmPose')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]], \
                                 -X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lArmPose')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]], \
                                 -X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]]]

            if ('baxter', 'rArmPose') in plan.state_inds:
                X[plan.state_inds[('baxter', 'rArmPose')]] = X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rArmPose')]]
            elif ('baxter', 'ee_right_pos') in plan.state_inds:
                X[plan.state_inds['baxter', 'ee_right_pos']] = [0, -1.0, 0.9]
            # X[plan.state_inds[('baxter', 'rGripper')]] = X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]]

        if ('baxter', 'ee_left_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_rot']] = [0, 0, 1, 0]

        if ('baxter', 'ee_right_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_rot']] = [0, 0, 1, 0]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config


def get_random_initial_cloth_place_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    joint_angles = []

    success = False

    num_cloths_on_table = np.random.randint(0, num_cloths)
    num_cloths_in_basket = num_cloths - num_cloths_on_table - 1
    next_cloth = num_cloths_in_basket

    actions = [4*next_cloth+2, 4*next_cloth+3]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    while not success:
        print('Searching for initial configuration...')
        X_0s = []
        stationary_params = ['basket']
        basket = plan.params['basket']
        # basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
        #                     [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
        #                     [BASKET_POSE[2]]]
        basket.pose[:,0] = [0.7, 0.025, 0.875]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [np.pi/2, 0, np.pi/2]

        possible_locs = np.sort(np.random.choice(list(range(0, CLOTH_XY**2, STEP_DELTA)), num_cloths_on_table, False)).tolist()
        possible_basket_locs = np.sort(np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket, False)).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])

        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])

        for c in range(num_cloths_in_basket+1, num_cloths):
            next_loc = possible_locs.pop(0)
            next_x = (next_loc / CLOTH_XY) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % CLOTH_XY) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        for c in range(0, num_cloths_in_basket):
            next_loc = possible_basket_locs.pop(0)
            next_x = (basket.pose[0,0] - 0.03) + (next_loc / 12) / 100.0
            next_y = (basket.pose[1,0] + 0.06) + (next_loc % 12) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        next_x = np.random.uniform(0.4, 0.8)
        next_y = np.random.uniform(0.3, 0.6)
        height = np.random.uniform(0.1, 0.35)
        l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
        if not len(l_arm_poses):
            success = False
            continue
        r_arm_pose = R_ARM_POSE

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]] = closest_arm_pose(l_arm_poses, CLOTH_GRASP_END_LEFT)
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lArmPose')]] = CLOTH_PUTDOWN_END_LEFT
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        if ('baxter', 'lArmPose') in plan.state_inds:
            X[plan.state_inds[('baxter', 'lArmPose')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]]
        elif ('baxter', 'ee_left_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_pos']] = [next_x, next_y, TABLE_TOP+height]
        if ('baxter', 'ee_left_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_rot']] = [0, 0, 1, 0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]]
        # X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]] = [next_x, next_y, TABLE_TOP+height]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(next_cloth), 'value')]] = X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]]

        if ('baxter', 'rArmPose') in plan.state_inds:
            X[plan.state_inds[('baxter', 'rArmPose')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]]
        elif ('baxter', 'ee_right_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_pos']] = [0, -1.0, 0.9]
        if ('baxter', 'ee_right_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_rot']] = [0, 0, 1, 0]
        # X[plan.state_inds[('baxter', 'rGripper')]] = X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]]

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]], \
                             -X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]]]
        plan.params['baxter'].openrave_body.set_dof({'lArmPose': X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]]})
        X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]] = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()[:3,3] # [0.7, 0.3, 1.0]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config

def get_random_initial_basket_grasp_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    stationary_params = []

    actions = [4*num_cloths, 4*num_cloths+1]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    success = False

    while not success:
        print('Searching for initial configuration...')

        height = np.random.uniform(0.35, 0.55)
        basket = plan.params['basket']
        # basket.pose[:,:] = [[np.random.uniform(0.45, 0.65)],
        #                     [np.random.uniform(-0.1, 0.1)],
        #                     [TABLE_TOP+height]]
        basket.pose[:,0] = [0.7, 0.025, 0.875]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [np.pi/2, 0, np.pi/2]

        possible_locs = np.sort(np.random.choice(list(range(0, 55*55, STEP_DELTA)), num_cloths_on_table, False)).tolist()
        possible_basket_locs = np.sort(np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket, False)).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose(basket.pose[:, 0])

        next_x = np.random.uniform(0.3, 0.8)
        next_y = np.random.uniform(-0.05, 0.4)
        height = np.random.uniform(0.1, 0.5)
        l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
        if not len(l_arm_poses):
            success = False
            continue
        r_arm_pose = R_ARM_POSE

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lArmPose')]] = l_arm_poses[0]
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]] = const.GRIPPER_OPEN_VALUE

        if ('baxter', 'lArmPose') in plan.state_inds:
            X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        elif ('baxter', 'ee_left_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_pos']] = [next_x, next_y, TABLE_TOP+height]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE

        if ('baxter', 'rArmPose'):
            X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
        elif ('baxter', 'ee_right_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_pos']] = [0, -1.0, 0.9]
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_OPEN_VALUE

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rArmPose')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lArmPose')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]]]

        for c in range(num_cloths):
            next_loc = possible_basket_locs.pop(0)
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 12) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 12) / 100.0
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, basket.pose[2,0]+BASKET_HEIGHT_DELTA-0.26]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config

def get_random_initial_rotate_with_basket_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    stationary_params = []

    actions = [4*num_cloths+2, 4*num_cloths+3]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    success = False

    while not success:
        print('Searching for initial configuration...')

        height = np.random.uniform(0.35, 0.55)
        basket = plan.params['basket']
        # basket.pose[:,:] = [[np.random.uniform(0.45, 0.65)],
        #                     [np.random.uniform(-0.1, 0.1)],
        #                     [TABLE_TOP+height]]
        basket.pose[:,0] = [0.7, 0.025, 0.875]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [np.pi/2, 0, np.pi/2]

        possible_locs = np.sort(np.random.choice(list(range(0, 65*65, STEP_DELTA)), num_cloths_on_table, False)).tolist()
        possible_basket_locs = np.sort(np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket, False)).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose(basket.pose[:, 0])

        ee_left = basket.pose[:, 0] + np.array([0, const.BASKET_OFFSET, 0])
        ee_right = basket.pose[:, 0] + np.array([0, -const.BASKET_OFFSET, 0])
        l_arm_poses = plan.params['baxter'].get_ik_from_pose(ee_left, [0, np.pi/2, 0], "left_arm")
        r_arm_poses = plan.params['baxter'].get_ik_from_pose(ee_right, [0, np.pi/2, 0], "right_arm")

        if not len(l_arm_poses) or not len(r_arm_poses):
            success = False
            continue

        plan.params['baxter'].openrave_body.set_dof({'lArmPose': l_arm_poses[0],
                                                     'lGripper': const.GRIPPER_CLOSE_VALUE,
                                                     'rArmPose': r_arm_poses[0],
                                                     'rGripper': const.GRIPPER_CLOSE_VALUE})

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_poses[0]
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('rotate_begin', 'lArmPose')]] = l_arm_poses[0]
        X[plan.state_inds[('rotate_begin', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('rotate_begin', 'rArmPose')]] = r_arm_poses[0]
        X[plan.state_inds[('rotate_begin', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE

        X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_poses[0]
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('rotate_begin', 'rArmPose')]], \
                             X[plan.state_inds[('rotate_begin', 'rGripper')]], \
                             -X[plan.state_inds[('rotate_begin', 'rGripper')]], \
                             X[plan.state_inds[('rotate_begin', 'lArmPose')]], \
                             X[plan.state_inds[('rotate_begin', 'lGripper')]], \
                             -X[plan.state_inds[('rotate_begin', 'lGripper')]]]

        for c in range(num_cloths):
            next_loc = possible_basket_locs.pop(0)
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 12) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 12) / 100.0
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, basket.pose[2,0]+BASKET_HEIGHT_DELTA-0.26]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config



























# New state gen functions

# What a plan would probably look like:
# For each cloth: pick (2 actions) then place (2 actions)
# Then grasp basket (2 actions)
# Then rotate with basket and putdown basket (2 actions)
# For each cloth: pick (2 actions) then place (2 actions)

def generate_full_cond(num_cloths):
    i = 1
    act_num = 4
    plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        '1: CLOTH_GRASP BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
        '3: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 BASKET_INIT_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
    ]

    while i < num_cloths:
        plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i-1, i))
        act_num += 1
        plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1

        plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, i))
        act_num += 1
        plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_INIT_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        i += 1

    # plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} BASKET_GRASP_BEGIN'.format(act_num, i-1))
    # act_num += 1
    # plan_str.append('{0}: BASKET_GRASP_WITH_CLOTH BAXTER BASKET BASKET_INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END'.format(act_num))
    # act_num += 1

    # plan_str.append('{0}: MOVEHOLDING_BASKET_WITH_CLOTH BAXTER BASKET_GRASP_END BASKET_ROTATE_BEGIN BASKET \n'.format(act_num))
    # act_num += 1
    # plan_str.append('{0}: ROTATE_HOLDING_BASKET_WITH_CLOTH BAXTER BASKET BASKET_ROTATE_BEGIN ROTATE_END_POSE REGION1 \n'.format(act_num))
    # act_num += 1

    # plan_str.append('{0}: MOVEHOLDING_BASKET_WITH_CLOTH BAXTER ROTATE_END_POSE BASKET_PUTDOWN_BEGIN BASKET \n'.format(act_num))
    # act_num += 1
    # plan_str.append('{0}: BASKET_PUTDOWN_WITH_CLOTH BAXTER BASKET BASKET_END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END'.format(act_num))


    domain_fname = '../domains/laundry_domain/laundry_policy.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_policy_{0}.prob'.format(num_cloths))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    # problem.init_state.params['cloth_target_begin_0'].value[:,0] = random_pose
    # problem.init_state.params['cloth_0'].pose[:,0] = random_pose

    plan = hls.get_plan(plan_str, domain, problem)

    basket = plan.params['basket']
    basket_target = plan.params['basket_init_target']
    basket.pose[:,0] = np.array(BASKET_POSE)
    basket.rotation[:,:1] = [[np.pi/2], [0], [np.pi/2]]
    basket_target.value[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket_target.rotation[:,:] = [[np.pi/2], [0], [np.pi/2]]

    plan.params['table'].pose[:,:] = np.array(TABLE_POSE).reshape(-1,1)
    plan.params['table'].rotation[:,:] = 0

    plan.params['robot_init_pose'].lArmPose[:,0] = CLOTH_PUTDOWN_END_LEFT
    plan.params['robot_init_pose'].lGripper[:,0] = 0.02
    plan.params['robot_init_pose'].rArmPose[:,0] = R_ARM_POSE # [0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].rGripper[:,0] = 0.02

    # plan.params['robot_end_pose'].lArmPose[:,0] = [1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    # plan.params['robot_end_pose'].lGripper[:,0] = 0
    # plan.params['robot_end_pose'].rArmPose[:,0] = [-1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    # plan.params['robot_end_pose'].rGripper[:,0] = 0.015

    possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA**2)), num_cloths).tolist()

    for c in range(num_cloths):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 45) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 45) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        plan.params['cloth_{0}'.format(c)].pose[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_{0}'.format(c)].rotation[:, :] = 0
        plan.params['cloth_target_begin_{0}'.format(c)].value[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_target_begin_{0}'.format(c)].rotation[:, :] = 0

    # for c in range(num_cloths):
    #     plan.params['cg_ee_{0}'.format(c)].value[:,:] = np.nan
    #     plan.params['cg_ee_{0}'.format(c)].rotation[:,:] = np.nan

    #     plan.params['cp_ee_{0}'.format(c)].value[:,:] = np.nan
    #     plan.params['cp_ee_{0}'.format(c)].rotation[:,:] = np.nan

    for c in range(1):

        plan.params['cloth_grasp_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_grasp_end_{0}'.format(c)].value[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].lArmPose[:,0] = CLOTH_GRASP_END_LEFT
        plan.params['cloth_grasp_end_{0}'.format(c)].lGripper[:,:] = 0
        plan.params['cloth_grasp_end_{0}'.format(c)].rArmPose[:,0] = R_ARM_POSE
        plan.params['cloth_grasp_end_{0}'.format(c)].rGripper[:,:] = 0

        plan.params['cloth_putdown_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_putdown_end_{0}'.format(c)].value[:,:] = 0
        plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,0] = CLOTH_PUTDOWN_END_LEFT
        plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = 0
        plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,0] = R_ARM_POSE
        plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = 0
        # plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.array(L_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = 0.02
        # plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.array(R_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = 0.02

    #     plan.params['cloth_target_end_{0}'.format(c)].value[:,:] = 0
    #     plan.params['cloth_target_end_{0}'.format(c)].rotation[:,:] = 0

    # plan.params['basket_grasp_end'].value[:,:] = 0
    # plan.params['basket_grasp_end'].lArmPose[:,:] = 0
    # plan.params['basket_grasp_end'].lGripper[:,:] = 0
    # plan.params['basket_grasp_end'].rArmPose[:,:] = 0
    # plan.params['basket_grasp_end'].rGripper[:,:] = 0

    # plan.params['basket_grasp_begin'].value[:,:] = 0
    # plan.params['basket_grasp_begin'].lArmPose[:,:] = 0
    # plan.params['basket_grasp_begin'].lGripper[:,:] = 0
    # plan.params['basket_grasp_begin'].rArmPose[:,:] = 0
    # plan.params['basket_grasp_begin'].rGripper[:,:] = 0

    plan.params['basket_grasp_begin'].value[:,:] = 0
    plan.params['basket_grasp_begin'].lArmPose[:,:] = np.nan
    plan.params['basket_grasp_begin'].lGripper[:,:] = np.nan
    plan.params['basket_grasp_begin'].rArmPose[:,:] = np.nan
    plan.params['basket_grasp_begin'].rGripper[:,:] = np.nan

    plan.params['basket_rotate_begin'].value[:,:] = 0
    plan.params['basket_rotate_begin'].lArmPose[:,:] = np.nan
    plan.params['basket_rotate_begin'].lGripper[:,:] = np.nan
    plan.params['basket_rotate_begin'].rArmPose[:,:] = np.nan
    plan.params['basket_rotate_begin'].rGripper[:,:] = np.nan

    plan.params['basket_putdown_begin'].value[:,:] = 0
    plan.params['basket_putdown_begin'].lArmPose[:,:] = np.nan
    plan.params['basket_putdown_begin'].lGripper[:,:] = np.nan
    plan.params['basket_putdown_begin'].rArmPose[:,:] = np.nan
    plan.params['basket_putdown_begin'].rGripper[:,:] = np.nan

    plan.params['rotate_end_pose'].value[:,:] = 0
    plan.params['rotate_end_pose'].lArmPose[:,:] = 0
    plan.params['rotate_end_pose'].lGripper[:,:] = 0
    plan.params['rotate_end_pose'].rArmPose[:,:] = 0
    plan.params['rotate_end_pose'].rGripper[:,:] = 0

    plan._determine_free_attrs()

    return plan

def get_random_initial_ee_cloth_pick_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []

    success = False
    num_cloths_on_table = np.random.randint(1, num_cloths+1)
    num_cloths_in_basket = num_cloths - num_cloths_on_table
    next_cloth = num_cloths_in_basket
    actions = [4*next_cloth, 4*next_cloth+1]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    while not success:
        print('Searching for initial configuration...')
        X_0s = []
        joint_angles = []
        stationary_params = ['basket']
        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA+STEP_DELTA*45)), num_cloths_on_table).tolist()
        possible_basket_locs = np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])

        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])

        for c in range(num_cloths_in_basket, num_cloths):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 45) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 45) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        for c in range(0, num_cloths_in_basket):
            next_loc = possible_basket_locs.pop()
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 24) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'rGripper')]] = plan.params['robot_init_pose'].rGripper
        if not next_cloth:
            plan.params['baxter'].openrave_body.set_dof({'lArmPose': plan.params['robot_init_pose'].lArmPose.flatten(),
                                                         'lGripper': plan.params['robot_init_pose'].lGripper.flatten(),
                                                         'rArmPose': plan.params['robot_init_pose'].rArmPose.flatten(),
                                                         'rGripper': plan.params['robot_init_pose'].rGripper.flatten()})
            ee_trans = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
            X[plan.state_inds[('baxter', 'ee_left_pos')]] = ee_trans[:3,3]
            X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE

            # Joint angles ordered by Mujoco Model joint order
            joint_angles = np.r_[0, X[plan.state_inds[('robot_init_pose', 'rArmPose')]], \
                                 X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                                 -X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                                 X[plan.state_inds[('robot_init_pose', 'lArmPose')]], \
                                 X[plan.state_inds[('robot_init_pose', 'lGripper')]], \
                                 -X[plan.state_inds[('robot_init_pose', 'lGripper')]]]
        else:
            next_x = np.random.uniform(0.3, 0.8)
            next_y = np.random.uniform(-0.1, 0.5)
            height = np.random.uniform(0.15, 0.5)
            l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
            if not len(l_arm_poses):
                success = False
                continue
            r_arm_pose = [-1.1, -0.7542201, 0.33281928, 0.81198007, 2.90107491, -1.54208563, 1.49564886]

            # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
            # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
            # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lArmPose')]] = l_arm_poses[-1]
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rArmPose')]] = r_arm_pose
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
            X[plan.state_inds[('baxter', 'ee_left_pos')]] = [next_x, next_y, TABLE_TOP+height]
            X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE

            # Joint angles ordered by Mujoco Model joint order
            joint_angles = np.r_[0, X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rArmPose')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]], \
                                 -X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'rGripper')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lArmPose')]], \
                                 X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]], \
                                 -X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth-1), 'lGripper')]]]

        X[plan.state_inds[('baxter', 'ee_right_pos')]] = [0, -1.0, 0.9]
        X[plan.state_inds[('baxter', 'ee_left_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'ee_right_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_OPEN_VALUE

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config


def get_random_initial_ee_cloth_place_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    joint_angles = []

    success = False

    num_cloths_on_table = np.random.randint(0, num_cloths)
    num_cloths_in_basket = num_cloths - num_cloths_on_table - 1
    next_cloth = num_cloths_in_basket

    actions = [4*next_cloth+2, 4*next_cloth+3]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    while not success:
        print('Searching for initial configuration...')
        X_0s = []
        stationary_params = ['basket']
        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA+STEP_DELTA*45)), num_cloths_on_table).tolist()
        possible_basket_locs = np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])

        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])

        for c in range(num_cloths_in_basket+1, num_cloths):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 45) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 45) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        for c in range(0, num_cloths_in_basket):
            next_loc = possible_basket_locs.pop()
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 24) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        next_x = np.random.uniform(0.2, 0.9)
        next_y = np.random.uniform(0.45, 1.0)
        height = np.random.uniform(0.15, 0.5)
        l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
        if not len(l_arm_poses):
            success = False
            continue
        r_arm_pose = [-1.1, -0.7542201, 0.33281928, 0.81198007, 2.90107491, -1.54208563, 1.49564886]

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]] = random.choice(l_arm_poses)
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('baxter', 'ee_left_pos')]] = [next_x, next_y, TABLE_TOP+height]
        X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]] = [next_x, next_y, TABLE_TOP+height]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(next_cloth), 'value')]] = X[plan.state_inds[('cloth_{0}'.format(next_cloth), 'pose')]]

        X[plan.state_inds[('baxter', 'ee_right_pos')]] = [0, -1.0, 0.9]
        X[plan.state_inds[('baxter', 'ee_left_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'ee_right_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]], \
                             X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]], \
                             -X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]]]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config

def get_random_initial_ee_basket_grasp_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    stationary_params = []

    actions = [4*num_cloths, 4*num_cloths+1]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    success = False

    while not success:
        print('Searching for initial configuration...')

        height = np.random.uniform(0.35, 0.55)
        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(0.45, 0.65)],
                            [np.random.uniform(-0.1, 0.1)],
                            [TABLE_TOP+height]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA+STEP_DELTA*45)), num_cloths_on_table).tolist()
        possible_basket_locs = np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose(basket.pose[:, 0])

        next_x = np.random.uniform(0.3, 0.8)
        next_y = np.random.uniform(-0.05, 0.4)
        height = np.random.uniform(0.1, 0.5)
        l_arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP+height], [0, np.pi/2, 0], "left_arm")
        if not len(l_arm_poses):
            success = False
            continue
        r_arm_pose = [-1.1, -0.7542201, 0.33281928, 0.81198007, 2.90107491, -1.54208563, 1.49564886]

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_pose
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lArmPose')]] = l_arm_poses[0]
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]] = const.GRIPPER_OPEN_VALUE
        X[plan.state_inds[('baxter', 'ee_left_pos')]] = [next_x, next_y, TABLE_TOP+height]
        X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_OPEN_VALUE

        X[plan.state_inds[('baxter', 'ee_right_pos')]] = [0, -1.0, 0.9]
        X[plan.state_inds[('baxter', 'ee_left_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'ee_right_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_OPEN_VALUE

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rArmPose')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'lArmPose')]], \
                             X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]], \
                             -X[plan.state_inds[('cloth_putdown_end_{0}'.format(num_cloths-1), 'rGripper')]]]

        for c in range(num_cloths):
            next_loc = possible_basket_locs.pop()
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 24) / 100.0
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, basket.pose[2,0]+BASKET_HEIGHT_DELTA-0.26]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config

def get_random_initial_ee_rotate_with_basket_state(plan, num_cloths):
    X = np.zeros((plan.dX))
    state_config = []
    stationary_params = []

    actions = [4*num_cloths+2, 4*num_cloths+3]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    success = False

    while not success:
        print('Searching for initial configuration...')

        height = np.random.uniform(0.35, 0.55)
        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(0.45, 0.65)],
                            [np.random.uniform(-0.1, 0.1)],
                            [TABLE_TOP+height]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(list(range(0, 45*45, STEP_DELTA+STEP_DELTA*45)), num_cloths_on_table).tolist()
        possible_basket_locs = np.random.choice(list(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12)), num_cloths_in_basket).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose(basket.pose[:, 0])

        ee_left = basket.pose[:, 0] + np.array([0, const.BASKET_OFFSET, 0])
        ee_right = basket.pose[:, 0] + np.array([0, -const.BASKET_OFFSET, 0])
        left_poses = plan.params['baxter'].get_ik_from_pose(ee_left, [0, np.pi/2, 0], "left_arm")
        right_poses = plan.params['baxter'].get_ik_from_pose(ee_right, [0, np.pi/2, 0], "right_arm")

        if not len(left_poses) or not len(right_poses):
            success = False
            continue

        plan.params['baxter'].openrave_body.set_dof({'lArmPose': l_arm_poses[0],
                                                     'lGripper': const.GRIPPER_CLOSE_VALUE,
                                                     'rArmPose': r_arm_poses[0],
                                                     'rGripper': const.GRIPPER_CLOSE_VALUE})

        # X[plan.state_inds[('baxter', 'lArmPose')]] = l_arm_poses[0]
        # X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        # X[plan.state_inds[('baxter', 'rArmPose')]] = r_arm_poses[0]
        # X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('rotate_begin', 'lArmPose')]] = l_arm_poses[0]
        X[plan.state_inds[('rotate_begin', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('rotate_begin', 'rArmPose')]] = r_arm_poses[0]
        X[plan.state_inds[('rotate_begin', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        X[plan.state_inds[('baxter', 'ee_left_pos')]] = ee_left
        X[plan.state_inds[('baxter', 'lGripper')]] = const.GRIPPER_CLOSE_VALUE

        X[plan.state_inds[('baxter', 'ee_right_pos')]] = ee_right
        X[plan.state_inds[('baxter', 'rGripper')]] = const.GRIPPER_CLOSE_VALUE

        X[plan.state_inds[('baxter', 'ee_left_rot')]] = [0, np.pi/2, 0]
        X[plan.state_inds[('baxter', 'ee_right_rot')]] = [0, np.pi/2, 0]

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('rotate_begin', 'rArmPose')]], \
                             X[plan.state_inds[('rotate_begin', 'rGripper')]], \
                             -X[plan.state_inds[('rotate_begin', 'rGripper')]], \
                             X[plan.state_inds[('rotate_begin', 'lArmPose')]], \
                             X[plan.state_inds[('rotate_begin', 'lGripper')]], \
                             -X[plan.state_inds[('rotate_begin', 'lGripper')]]]

        for c in range(num_cloths):
            next_loc = possible_basket_locs.pop()
            next_x = (basket.pose[0,0] - 0.12) + (next_loc / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (next_loc % 24) / 100.0
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, basket.pose[2,0]+BASKET_HEIGHT_DELTA-0.26]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config

def get_random_initial_pick_place_state(plan, num_cloths):
    # if np.random.choice([1]):
    #     return get_random_initial_cloth_pick_state(plan, num_cloths)
    # else:
    #     return get_random_initial_cloth_place_state(plan, num_cloths)
    return get_random_4_cloth_pick_state(plan)
    # return get_random_initial_cloth_pick_state(plan, num_cloths)

def state_vector_value(vec, plan, param_name, attr):
    return vec[plan.state_inds[param_name, attr]]



def get_random_4_cloth_pick_state(plan):
    X = np.zeros((plan.dX))
    state_config = []

    num_cloths = 4
    success = False
    num_cloths_on_table = 4 # np.random.randint(1, num_cloths+1)
    # num_cloths_in_basket = num_cloths - num_cloths_on_table
    next_cloth = 0 # num_cloths_in_basket
    actions = [0, len(plan.actions)-1]
    start_t = plan.actions[actions[0]].active_timesteps[0]

    regions = [[[0.3, 0.5], [0.65, 0.85]], [[0.65, 0.85], [0.45, 0.7]], [[0.25, 0.45], [0.9, 1.1]], [[0.5, 0.65], [0.75, 0.95]]]

    while not success:
        print('Searching for initial configuration...')
        X_0s = []
        joint_angles = []
        stationary_params = ['basket']
        basket = plan.params['basket']
        # basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
        #                     [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
        #                     [BASKET_POSE[2]]]
        basket.pose[:,0] = [0.65, 0.025, 0.875]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        if ('basket', 'rotation') in plan.state_inds:
            X[plan.state_inds[('basket', 'rotation')]] = basket.rotation[:,0]
        X[plan.state_inds[('basket_init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('basket_init_target', 'rotation')]] = [np.pi/2, 0, np.pi/2]

        possible_locs = np.sort(np.random.choice(list(range(0, CLOTH_XY**2, STEP_DELTA)), num_cloths_on_table, False)).tolist()
        # possible_basket_locs = np.sort(np.random.choice(range(0, 144, BASKET_STEP_DELTA+BASKET_STEP_DELTA*12), num_cloths_in_basket, False)).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])

        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])


        for c in range(num_cloths):
            bounds = regions[c]
            next_x = np.random.uniform(bounds[0][0], bounds[0][1])
            next_y = np.random.uniform(bounds[1][0], bounds[1][1])
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        for c in range(num_cloths):
            target_cloth = X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]]
            grasp_pose = plan.params['baxter'].openrave_body.get_ik_from_pose([target_cloth[0], target_cloth[1], target_cloth[2]+0.05], [0, np.pi/2, 0], "left_arm")
            if not len(grasp_pose):
                success = False
                continue
            grasp_pose = plan.params['baxter'].openrave_body.get_ik_from_pose([target_cloth[0], target_cloth[1], target_cloth[2]+0.3], [0, np.pi/2, 0], "left_arm")
            if not len(grasp_pose):
                success = False
                continue

        r_arm_pose = R_ARM_POSE

        X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'rGripper')]] = plan.params['robot_init_pose'].rGripper
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lArmPose')]] = CLOTH_GRASP_END_LEFT
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_CLOSE_VALUE
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_grasp_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        plan.params['baxter'].openrave_body.set_dof({'lArmPose': plan.params['robot_init_pose'].lArmPose.flatten(),
                                                     'lGripper': plan.params['robot_init_pose'].lGripper.flatten(),
                                                     'rArmPose': plan.params['robot_init_pose'].rArmPose.flatten(),
                                                     'rGripper': plan.params['robot_init_pose'].rGripper.flatten()})
        if ('baxter', 'lArmPose') in plan.state_inds:
            X[plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        elif ('baxter', 'ee_left_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_pos']] = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()[:3,3]
        # X[plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        if ('baxter', 'rArmPose') in plan.state_inds:
            X[plan.state_inds[('baxter', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        elif ('baxter', 'ee_right_pos') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_pos']] = plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransform()[:3,3]
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'lGripper')]] = const.GRIPPER_OPEN_VALUE
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rArmPose')]] = r_arm_pose
        X[plan.state_inds[('cloth_putdown_end_{0}'.format(next_cloth), 'rGripper')]] = const.GRIPPER_CLOSE_VALUE
        # X[plan.state_inds[('baxter', 'rGripper')]] = plan.params['robot_init_pose'].rGripper

        # Joint angles ordered by Mujoco Model joint order
        joint_angles = np.r_[0, X[plan.state_inds[('robot_init_pose', 'rArmPose')]], \
                             X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                             -X[plan.state_inds[('robot_init_pose', 'rGripper')]], \
                             X[plan.state_inds[('robot_init_pose', 'lArmPose')]], \
                             X[plan.state_inds[('robot_init_pose', 'lGripper')]], \
                             -X[plan.state_inds[('robot_init_pose', 'lGripper')]]]
        if ('baxter', 'ee_left_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_left_rot']] = [0, 0, 1, 0]

        if ('baxter', 'ee_right_rot') in plan.state_inds:
            X[plan.state_inds['baxter', 'ee_right_rot']] = [0, 0, 1, 0]

        state_config = [X, actions, stationary_params, joint_angles]

    print("Found initial configuration.\n")
    return state_config
