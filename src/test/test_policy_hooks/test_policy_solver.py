import unittest, time, main

import numpy as np

from core.parsing import parse_domain_config, parse_problem_config
from pma import hl_solver, robot_ll_solver
from policy_hooks import policy_solver, tamp_agent, policy_hyperparams, policy_solver_utils

def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    return domain, problem, params

def traj_retiming(plan, velocity):
    baxter = plan.params['baxter']
    rave_body = baxter.openrave_body
    body = rave_body.env_body
    lmanip = body.GetManipulator("left_arm")
    rmanip = body.GetManipulator("right_arm")
    left_ee_pose = []
    right_ee_pose = []
    for t in range(plan.horizon):
        rave_body.set_dof({
            'lArmPose': baxter.lArmPose[:, t],
            'lGripper': baxter.lGripper[:, t],
            'rArmPose': baxter.rArmPose[:, t],
            'rGripper': baxter.rGripper[:, t]
        })
        rave_body.set_pose([0,0,baxter.pose[:, t]])

        left_ee_pose.append(lmanip.GetTransform()[:3, 3])
        right_ee_pose.append(rmanip.GetTransform()[:3, 3])
    time = np.zeros(plan.horizon)
    # import ipdb; ipdb.set_trace()
    for t in range(plan.horizon-1):
        left_dist = np.linalg.norm(left_ee_pose[t+1] - left_ee_pose[t])
        right_dist = np.linalg.norm(right_ee_pose[t+1] - right_ee_pose[t])
        time_spend = max(left_dist, right_dist)/velocity[t]
        time[t+1] = time_spend
    return time

# Useful for creating sample plans
def get_random_cloth_init_poses(num_cloths, table_pos):
    cur_xy = [-.25, -.525]
    cloth_poses = []
    for i in range(num_cloths):
        if not (i+1) % 4:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(-0.4, -0.5), np.random.uniform(0.1, 0.15)])
            cur_xy[0] = max(cur_xy[0], -.25)
        else:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(0.1, 0.15), np.random.uniform(-0.025, 0.025)])
        pos = np.array(table_pos) + np.array([cur_xy[0], cur_xy[1], 0.05])
        cloth_poses.append(pos.tolist())
    return cloth_poses

def get_random_cloth_init_pose(table_pos):
    cur_xy = np.array([np.random.uniform(-0.2, 0.1), np.random.uniform(0.1, 0.5)])
    pos = np.array(table_pos) + np.array([cur_xy[0], cur_xy[1], 0.05])
    return pos

# Useful for creating sample plans
def get_random_cloth_end_poses(num_cloths, basket_init_pos):
    cur_xy = [-.11, .11]
    cloth_poses = []
    for i in range(num_cloths):
        if not (i+1) % 4:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(-0.21, -0.23), np.random.uniform(0.045, 0.055)])
            cur_xy[0] = max(cur_xy[0], -.11)
        else:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(0.045, 0.055), np.random.uniform(-0.01, 0.01)])
        pos = np.array(basket_init_pos) + np.array([cur_xy[0], cur_xy[1], 0.04])
        cloth_poses.append(pos.tolist())
    return cloth_poses


class TestPolicySolver(unittest.TestCase):
    def test_policy_solver(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/single_cloth_policy.prob')

        plans = []

        ll_solver = robot_ll_solver.RobotLLSolver()

        for i in range(50):

            plan_str = [
                '1: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
                '2: CLOTH_GRASP  BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
                '3: MOVEHOLDING_CLOTH  BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
                '4: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 END_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
                '5: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_END_POSE'
            ]

            ## Use this if multiple cloths in the plan
            # for i in range(1, 20):
            #     plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOTH_GRASP_BEGIN_{2}'.format((i-1)*3+1, i-1, i))
            #     plan_str.append('{0}: CLOTH_GRASP  BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format((i-1)*3+2, i, i, i, i, i))
            #     plan_str.append('{0}: CLOTH_PUTDOWN BAXTER CLOTH_{1} CLOTH_TARGET_END_{2}, CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format((i-1)*3+3, i, i, i, i, i))

            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            random_pose = get_random_cloth_init_pose(problem.init_state.params['table'].pose[:,0])
            problem.init_state.params['cloth_target_begin_0'].value[:,0] = random_pose
            problem.init_state.params['cloth_0'].pose[:,0] = random_pose

            plan = hls.get_plan(plan_str, domain, problem)
            result = ll_solver.backtrack_solve(plan)
            if not result:
                continue
            plan.time = np.ones((1, plan.horizon))
            baxter = plan.params['baxter']
            cloth = plan.params['cloth_0']
            basket = plan.params['basket']
            table = plan.params['table']
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, x_params=[baxter, cloth, basket, table], \
                                                                                                                                                                                                  u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plans.append(plan)

        solver = policy_solver.BaxterPolicySolver()
        solver.train_policy(plans)
        import ipdb; ipdb.set_trace()
