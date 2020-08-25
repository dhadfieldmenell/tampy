import numpy as np
import unittest, time, main
from pma import hl_solver, robot_ll_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes import baxter_sampling
from core.util_classes.baxter_predicates import BaxterCollides
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
# from ros_interface import action_execution
import core.util_classes.baxter_constants as const
from openravepy import matrixFromAxisAngle
import itertools
from collections import OrderedDict

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
        time_spend = max(left_dist  , right_dist)/velocity[t]
        time[t+1] = time[t] + time_spend
    return time


class TestBasketDomain(unittest.TestCase):
    def test_rotor_base_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0',
        '1: ROTATE BAXTER ROBOT_REGION_1_POSE_0 ROBOT_REGION_3_POSE_0 REGION3',
        '2: MOVETO BAXTER ROBOT_REGION_3_POSE_0 BASKET_GRASP_BEGIN_0',
        '3: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0',
        '4: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_3_POSE_1 REGION1',
        '5: ROTATE_HOLDING_BASKET BAXTER BASKET ROBOT_REGION_3_POSE_1 BASKET_PUTDOWN_BEGIN_0 REGION3',
        '6: BASKET_PUTDOWN BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0',
        '7: MOVETO BAXTER BASKET_PUTDOWN_END_0 ROBOT_INIT_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file rotate_basket_plan.hdf5...")

        serializer.write_plan_to_hdf5("rotate_basket_plan.hdf5", plan)
        self.assertTrue(result)

    def test_washer_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: ROTATE BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0 REGION1',
        '1: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOSE_DOOR_BEGIN_0',
        '2: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_0 CLOSE_DOOR_EE_APPROACH_0 CLOSE_DOOR_EE_RETREAT_0 CLOSE_DOOR_END_0 WASHER_OPEN_POSE_0 WASHER_CLOSE_POSE_0',
        '3: MOVETO BAXTER CLOSE_DOOR_END_0  OPEN_DOOR_BEGIN_0',
        '4: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 OPEN_DOOR_EE_APPROACH_0 OPEN_DOOR_EE_RETREAT_0 OPEN_DOOR_END_0 WASHER_CLOSE_POSE_0 WASHER_OPEN_POSE_0',
        '5: ROTATE BAXTER OPEN_DOOR_END_0 ROBOT_REGION_3_POSE_0 REGION3',
        '6: MOVETO BAXTER ROBOT_REGION_3_POSE_0 ROBOT_INIT_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file rotate_basket_plan.hdf5...")

        serializer.write_plan_to_hdf5("rotate_washer_manipulation.hdf5", plan)
        self.assertTrue(result)

    def test_cloth_pickup_putdown_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: ROTATE BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0 REGION2',
        '1: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOTH_GRASP_BEGIN_0',
        '2: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '3: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0',
        '4: CLOTH_PUTDOWN BAXTER CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        '5: ROTATE BAXTER CLOTH_PUTDOWN_END_0 ROBOT_REGION_3_POSE_0 REGION3',
        '6: MOVETO BAXTER ROBOT_REGION_3_POSE_0 ROBOT_INIT_POSE',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file cloth_pickup_putdown.hdf5...")

        serializer.write_plan_to_hdf5("cloth_pickup_putdown.hdf5", plan)
        self.assertTrue(result)

    def test_cloth_in_out_washer_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: ROTATE BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0 REGION2',
        '1: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOTH_GRASP_BEGIN_0',
        '2: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '3: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 REGION1',
        '4: MOVEHOLDING_CLOTH BAXTER CLOTH_PUTDOWN_BEGIN_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0',
        '5: MOVEHOLDING_CLOTH BAXTER LOAD_WASHER_INTERMEDIATE_POSE CLOTH_PUTDOWN_BEGIN_0 CLOTH0',
        '6: PUT_INTO_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        '7: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_REGION_1_POSE_1',
        '8: ROTATE BAXTER ROBOT_REGION_1_POSE_1 ROBOT_REGION_3_POSE_0 REGION3',
        '9: MOVETO BAXTER ROBOT_REGION_3_POSE_0 ROBOT_INIT_POSE',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file cloth_in_out_washer.hdf5...")

        serializer.write_plan_to_hdf5("cloth_in_out_washer.hdf5", plan)
        self.assertTrue(result)

    def test_rotor_base_stress_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN_0',
        '1: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0',
        '2: MOVETO BAXTER BASKET_GRASP_END_0 ROBOT_REGION_3_POSE_0',
        '3: ROTATE_HOLDING_BASKET BAXTER BASKET ROBOT_REGION_3_POSE_0 ROBOT_REGION_3_POSE_1 REGION1',
        '4: MOVETO BAXTER ROBOT_REGION_3_POSE_1 BASKET_PUTDOWN_BEGIN_0',
        '5: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_0 BG_EE_LEFT_1 BG_EE_RIGHT_1 BASKET_PUTDOWN_END_0',
        '6: MOVETO BAXTER BASKET_PUTDOWN_END_0 BASKET_GRASP_BEGIN_1',
        '7: BASKET_GRASP BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_1 BG_EE_LEFT_1 BG_EE_RIGHT_1 BASKET_GRASP_END_1',
        '8: MOVETO BAXTER BASKET_GRASP_END_1 ROBOT_REGION_1_POSE_0',
        '9: ROTATE_HOLDING_BASKET BAXTER BASKET ROBOT_REGION_1_POSE_0 ROBOT_REGION_1_POSE_1 REGION3',
        '10: MOVETO BAXTER ROBOT_REGION_1_POSE_1 BASKET_PUTDOWN_BEGIN_1',
        '11: BASKET_PUTDOWN BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_1 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_1',
        '12: MOVETO BAXTER BASKET_PUTDOWN_END_1 ROBOT_INIT_POSE'
        ]

        # '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN_0',
        # '1: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0',
        # '2: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_3_POSE_1 REGION1',
        # '3: ROTATE_HOLDING_BASKET BAXTER BASKET ROBOT_REGION_3_POSE_1 BASKET_PUTDOWN_BEGIN_0 REGION3',
        # '4: BASKET_PUTDOWN BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0',
        # '5: MOVETO BAXTER BASKET_PUTDOWN_END_0 ROBOT_INIT_POSE'
        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file rotate_basket_plan.hdf5...")

        serializer.write_plan_to_hdf5("rotate_basket_stress_plan.hdf5", plan)
        self.assertTrue(result)

    def test_cloth_pickup_putdown_in_basket_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_mitch_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: ROTATE BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0 REGION2',
        '1: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOTH_GRASP_BEGIN_0',
        '2: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '3: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 ROBOT_REGION_2_POSE_0 REGION3',
        '4: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET ROBOT_REGION_2_POSE_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        '5: MOVETO BAXTER CLOTH_PUTDOWN_END_0 CLOTH_GRASP_BEGIN_1',
        '6: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_END_0 CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '7: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 LOAD_BASKET_FAR CLOTH0',
        '8: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 LOAD_BASKET_FAR ROBOT_REGION_2_POSE_1 REGION2',
        '9: CLOTH_PUTDOWN BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_REGION_2_POSE_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '10: ROTATE BAXTER CLOTH_PUTDOWN_END_1 ROBOT_REGION_3_POSE_0 REGION3',
        '11: MOVETO BAXTER ROBOT_REGION_3_POSE_0 ROBOT_INIT_POSE',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print("solving basket domain problem...")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        solver = robot_ll_solver.RobotLLSolver()
        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print("Planning finished within {}s, displaying failed predicates...".format(end - start))

        print("Saving current plan to file cloth_pickup_putdown_in_basket.hdf5...")

        serializer.write_plan_to_hdf5("cloth_pickup_putdown_in_basket.hdf5", plan)
        self.assertTrue(result)
