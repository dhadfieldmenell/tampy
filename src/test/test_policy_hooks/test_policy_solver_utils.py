import unittest, time, main

import numpy as np

from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
from pma import hl_solver, robot_ll_solver
from policy_hooks import policy_solver_utils

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

class TestPolicySolverUtils(unittest.TestCase):
    def test_vel_acc(self):
        # domain_fname = '../domains/laundry_domain/laundry.domain'
        # d_c = main.parse_file_to_dict(domain_fname)
        # domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        # hls = hl_solver.FFSolver(d_c)
        # p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/single_cloth_policy.prob')
        # problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        # plan_str = [
        #     '1: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        #     '2: CLOTH_GRASP  BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        #     '3: MOVEHOLDING_CLOTH  BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
        #     '4: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 END_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        #     '5: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_END_POSE'
        # ]
        # plan = hls.get_plan(plan_str, domain, problem)
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        # viewer.draw_plan_ts(plan, 0)
        # def callback(a): return viewer
        # solver = robot_ll_solver.RobotLLSolver()
        # result = solver.backtrack_solve(plan, callback = callback, verbose=False)

        # serializer = PlanSerializer()
        # serializer.write_plan_to_hdf5("vel_acc_test_plan.hdf5", plan)

        deserializer = PlanDeserializer()
        plan = deserializer.read_from_hdf5("vel_acc_test_plan.hdf5")

        plan.time = np.ones((1, plan.horizon))
        plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
        plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        vel, acc = policy_solver_utils.map_trajectory_to_vel_acc(plan)
        import ipdb; ipdb.set_trace()
