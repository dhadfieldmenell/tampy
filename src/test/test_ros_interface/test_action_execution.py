import sys, unittest, time, main, rospy
import numpy as np
from pma import hl_solver, robot_ll_solver
import baxter_interface
from baxter_interface import CHECK_VERSION
from core.parsing import parse_domain_config, parse_problem_config
from ros_interface import action_execution, controllers
from core.util_classes.plan_hdf5_serialization import PlanDeserializer, PlanSerializer
from openravepy import Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
from core.util_classes import baxter_constants
from core.util_classes.viewer import OpenRAVEViewer


class TestActionExecute(unittest.TestCase):

    def test_execute(self):
        '''
        This will try to talk to the Baxter, so launch the sim or real robot
        first
        '''
        # import ipdb; ipdb.set_trace()
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("prototype_plan.hdf5")

        velocites = np.ones((plan.horizon, ))
        # slow_inds = np.array([range(19,39), range(58,78), range(97,117), range(136,156), range(175,195), range(214,234)]).flatten()
        # velocites[slow_inds] = 1.0
        baxter = plan.params['baxter']
        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))
        print("Initializing node... ")
        rospy.init_node("rsdk_joint_trajectory_client")
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        print("Enabling robot... ")
        rs.enable()
        print("Running. Ctrl-c to quit")
        baxter_interface.Gripper('left', CHECK_VERSION).calibrate()
        baxter_interface.Gripper('right', CHECK_VERSION).calibrate()
        action_execution.execute_plan(plan)
        # for action in plan.actions:
        #       import ipdb; ipdb.set_trace()
        #       action_execution_2.execute_action(action)

    def test_trajectory_controller(self):
        '''
        This will try to talk to the Baxter, so launch the sim or real robot
        first
        '''
        # import ipdb; ipdb.set_trace()
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("prototype2.hdf5")

        # slow_inds = np.array([range(19,39), range(58,78), range(97,117), range(136,156), range(175,195), range(214,234)]).flatten()
        # velocites[slow_inds] = 1.0
        baxter = plan.params['baxter']
        plan.time = np.ones((1, plan.horizon)) * 0.25
        print("Initializing node... ")
        rospy.init_node("rsdk_joint_trajectory_client")
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        print("Enabling robot... ")
        rs.enable()
        print("Running. Ctrl-c to quit")
        ctrl = trajectory_controller.TrajectoryController()
        baxter_interface.Gripper('left', CHECK_VERSION).calibrate()
        baxter_interface.Gripper('right', CHECK_VERSION).calibrate()
        ctrl.execute_plan(plan)
        # for action in plan.actions:
        #       import ipdb; ipdb.set_trace()
        #       action_execution_2.execute_action(action)

    def test_environment_updates(self):
        '''
        This will try to talk to the Baxter, so launch the sim or real robot
        first
        '''
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("prototype2.hdf5")
        baxter = plan.params['baxter']
        plan.time = np.ones((1, plan.horizon)) * 0.25
        print("Initializing node... ")
        rospy.init_node("rsdk_joint_trajectory_client")
        print("Running. Ctrl-c to quit")
        action_execution.execute_plan(plan)

    def test_prototype2(self):
        print("loading laundry domain...")
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/prototype2.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_INIT_TARGET CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 BASKET_GRASP_BEGIN',
        '5: BASKET_GRASP BAXTER BASKET BASKET_INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '6: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '7: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '8: MOVETO BAXTER BASKET_PUTDOWN_END CLOTH_GRASP_BEGIN_2',
        '9: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_2 CLOTH_GRASP_BEGIN_2 CG_EE_2 CLOTH_GRASP_END_2',
        '10: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_2 CLOTH_PUTDOWN_BEGIN_2 CLOTH',
        '11: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_2 CLOTH_PUTDOWN_BEGIN_2 CP_EE_2 CLOTH_PUTDOWN_END_2',
        '12: MOVETO BAXTER CLOTH_PUTDOWN_END_2 ROBOT_END_POSE'
        ]
        print("constructing plan object")
        plan = hls.get_plan(plan_str, domain, problem)

        print("Initializing node... ")
        rospy.init_node("rsdk_joint_trajectory_client")
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        print("Enabling robot... ")
        rs.enable()
        print("Running. Ctrl-c to quit")
        baxter_interface.Gripper('left', CHECK_VERSION).calibrate()
        baxter_interface.Gripper('right', CHECK_VERSION).calibrate()
        action_execution.execute_plan(plan)

        import ipdb; ipdb.set_trace()

    def test_basket_grasp(self):
        print("loading laundry domain...")
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/basket_move.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN',
        '1: BASKET_GRASP BAXTER BASKET BASKET_INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '2: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '3: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '4: MOVETO BAXTER BASKET_PUTDOWN_END ROBOT_INIT_POSE'
        ]
        print("constructing plan object")
        plan = hls.get_plan(plan_str, domain, problem)

        print("solving laundry domain problem...")
        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        success = solver.backtrack_solve(plan, callback = None, verbose=False)
        end = time.time()
        print("Planning finished within {}s.".format(end - start))

        import ipdb; ipdb.set_trace()

        print("Initializing node... ")
        rospy.init_node("rsdk_joint_trajectory_client")
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        print("Enabling robot... ")
        rs.enable()
        print("Running. Ctrl-c to quit")
        baxter_interface.Gripper('left', CHECK_VERSION).calibrate()
        baxter_interface.Gripper('right', CHECK_VERSION).calibrate()
        ctrl = trajectory_controller.TrajectoryController()
        ctrl.execute_plan(plan)

        import ipdb; ipdb.set_trace()

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
        time[t+1] = time[t] + time_spend
    return time
