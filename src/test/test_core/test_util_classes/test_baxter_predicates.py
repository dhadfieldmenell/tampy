import unittest
from core.util_classes import robot_predicates, baxter_predicates, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.param_setup import ParamSetup
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.baxter_constants as const
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.viewer import OpenRAVEViewer
import numpy as np
import main

def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    return domain, problem, params

class TestBaxterPredicates(unittest.TestCase):

    # Begin of the test

    def test_robot_at(self):

        # RobotAt, Robot, RobotPose

        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()

        pred = baxter_predicates.BaxterRobotAt("testRobotAt", [robot, rPose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "BaxterRobotAt")
        # Robot and RobotPose are initialized to the same pose
        self.assertTrue(pred.test(0))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")
        robot.pose = np.array([[3, 4, 5, 3]])
        rPose.value = np.array([[3, 4, 2, 6]])
        self.assertTrue(pred.is_concrete())
        robot.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        robot.lGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        rPose.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        robot.rArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        robot.lArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        rPose.rArmPose = np.array([[0,0,0,0,0,0,0]]).T
        rPose.lArmPose = np.array([[0,0,0,0,0,0,0]]).T
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")

        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_is_mp(self):

        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()

        pred = baxter_predicates.BaxterIsMP("test_isMP", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "BaxterIsMP")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (BaxterIsMP baxter)' at the timestep.")
        # Getting lowerbound and movement step
        llA_l, lA_m = pred.lower_limit[0:7], pred.joint_step[0:7]
        lrA_l, rA_m = pred.lower_limit[8:15], pred.joint_step[8:15]
        llG_l, lG_m = pred.lower_limit[7], pred.joint_step[7]
        lrG_l, rG_m = pred.lower_limit[15], pred.joint_step[15]

        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[0,2,3,4,5,6,7]])

        # Arm pose is valid in the timestep: 0,1,2,3
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))

        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))

        # Gripper pose is valid in the timestep: 0,1,3,4,5
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Thus only timestep 1 and 3 are valid
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertFalse(pred.test(4))
        self.assertFalse(pred.test(5))
        with self.assertRaises(PredicateException) as cm:
            pred.test(6)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (BaxterIsMP baxter)' at the timestep.")

    def test_within_joint_limit(self):

        joint_factor = const.JOINT_MOVE_FACTOR
        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()

        pred = baxter_predicates.BaxterWithinJointLimit("test_joint_limit", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "BaxterWithinJointLimit")
        # Getting lowerbound and movement step
        llA_l, lA_m = pred.lower_limit[0:7], pred.joint_step[0:7]
        lrA_l, rA_m = pred.lower_limit[8:15], pred.joint_step[8:15]
        llG_l, lG_m = pred.lower_limit[7], pred.joint_step[7]
        lrG_l, rG_m = pred.lower_limit[15], pred.joint_step[15]
        # Base pose is valid in the timestep: 0,1,2,3,4,5
        robot.pose = np.zeros((7,1))
        # timestep 0, 3, 6 should fail
        robot.rArmPose = np.hstack((lrA_l+(joint_factor+1)*rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l-1*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+(joint_factor*10)*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Thus timestep 1, 3, 6 should fail
        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertFalse(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))

    def test_in_contact(self):

        # InContact robot EEPose target

        robot = ParamSetup.setup_baxter()
        ee_pose = ParamSetup.setup_ee_pose()
        target = ParamSetup.setup_target()
        env = ParamSetup.setup_env()

        pred = baxter_predicates.BaxterInContact("testInContact", [robot, ee_pose, target], ["Robot", "EEPose", "Target"])
        self.assertEqual(pred.get_type(), "BaxterInContact")
        # EEPose and target is not yet initialized, thus pred will not pass
        self.assertFalse(pred.test(0))
        ee_pose.value = np.zeros((3, 1))
        target.value = np.zeros((3, 1))
        # Baxter gets initialized with Open gripper, thus will not pass
        self.assertFalse(pred.test(0))
        # Set baxter's gripper to fully Closed
        robot.rGripper = np.array([[0.00]])
        self.assertFalse(pred.test(0))
        # Set baxter's gripper to be just enough to grasp the can
        robot.rGripper = np.array([[0.015]])
        self.assertTrue(pred.test(0))

    def test_in_gripper(self):

        # InGripper, Robot, Can

        robot = ParamSetup.setup_baxter()
        can = ParamSetup.setup_blue_can()
        test_env = ParamSetup.setup_env()

        pred = baxter_predicates.BaxterInGripperPos("InGripper", [robot, can], ["Robot", "Can"], test_env)
        pred2 = baxter_predicates.BaxterInGripperRot("InGripper_rot", [robot, can], ["Robot", "Can"], test_env)
        # Since this predicate is not yet concrete
        pred._param_to_body[robot].set_transparency(.7)
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0,0,0]]).T

        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        # check the gradient of the implementations (correct)
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        # Now set can's pose and rotation to be the right things
        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(pred.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        # A new robot arm pose
        robot.rArmPose = np.array([[0,-np.pi/4,np.pi/4,np.pi/2,-np.pi/4,-np.pi/4,0]]).T
        self.assertFalse(pred.test(0))

        # Only the pos is correct, rotation is not yet right
        can.pose = np.array([[1.08769922, -0.31906039,  1.21028557]]).T
        self.assertTrue(pred.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        can.rotation = np.array([[-2.84786534,  0.25268026, -2.98976055]]).T
        self.assertTrue(pred.test(0))
        self.assertTrue(pred2.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        # now rotate robot basepose
        robot.pose = np.array([[np.pi/3]]).T
        self.assertFalse(pred.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        can.pose = np.array([[0.82016401,  0.78244496,  1.21028557]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred2.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)
        can.rotation = np.array([[-1.80066778,  0.25268026, -2.98976055]]).T
        self.assertTrue(pred2.test(0))
        self.assertTrue(pred.test(0))
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, const.TOL)

    def test_ee_reachable(self):

        # InGripper, Robot, Can

        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()
        rPose = ParamSetup.setup_baxter_pose()
        ee_pose = ParamSetup.setup_ee_pose()

        pred = baxter_predicates.BaxterEEReachablePos("ee_reachable", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        pred2 = baxter_predicates.BaxterEEReachableRot("ee_reachable_rot", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        baxter = pred._param_to_body[robot]
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))

        ee_pose.value = np.array([[1.2, -0.1, 0.925]]).T
        ee_pose.rotation = np.array([[0,0,0]]).T
        ee_targ = ParamSetup.setup_green_can()
        ee_body = OpenRAVEBody(test_env, "EE_Pose", ee_targ.geom)
        ee_body.set_pose(ee_pose.value[:, 0], ee_pose.rotation[:, 0])

        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        robot.rArmPose = np.zeros((7,7))
        # initialized pose value is not right
        self.assertFalse(pred.test(0))

        # Find IK Solution
        trajectory = []
        trajectory.append(baxter.get_ik_from_pose([1.2-3*const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])    #s=-3
        trajectory.append(baxter.get_ik_from_pose([1.2-2*const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])    #s=-2
        trajectory.append(baxter.get_ik_from_pose([1.2-const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])       #s=-1
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925], [0,0,0], "right_arm")[0])                     #s=0
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+const.RETREAT_DIST], [0,0,0], "right_arm")[0])        #s=1
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+2*const.RETREAT_DIST], [0,0,0], "right_arm")[0])      #s=2
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+3*const.RETREAT_DIST], [0,0,0], "right_arm")[0])      #s=3
        trajectory = np.array(trajectory)


        robot.rArmPose = trajectory.T
        # Predicate should succeed in the grasping post at t=3,
        # EEreachableRot should always pass since rotation is right all the time
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred2.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred2.test(2))
        self.assertTrue(pred.test(3))
        self.assertTrue(pred2.test(3))
        # Since finding ik introduced some error, we relax the tolerance to 1e-3
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(3), True, 1e-3)

        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(3), True, 1e-3)


    def test_obstructs(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can

        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        can = ParamSetup.setup_blue_can()
        test_env = ParamSetup.setup_env()

        pred = baxter_predicates.BaxterObstructs("test_obstructs", [robot, rPose, rPose, can], ["Robot", "RobotPose", "RobotPose", "Can"], test_env, tol=const.TOL)
        self.assertEqual(pred.get_type(), "BaxterObstructs")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # This gradient test passed
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-2)

        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # This gradient test passed
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, const.TOL)

        # Move can to the center of the gripper (touching -> should recognize as collision)

        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        robot.rGripper = np.matrix([[const.GRIPPER_CLOSE_VALUE]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # The gradient of collision check when can is in the center of the gripper is extremenly inaccurate, making gradients check fails.
        # if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-1)

        # Fully open the gripper, now Gripper shuold be fine
        robot.rGripper = np.matrix([[const.GRIPPER_OPEN_VALUE]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))

        # The gradient of collision check when can is in the center of the gripper is extremenly inaccurate, making gradients check fails.
        # if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)

        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-4)

        # Move can into the robot arm, should have collision
        can.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)

    def test_obstructs_holding(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can, Can

        TEST_GRAD = False
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        can = ParamSetup.setup_blue_can("can1", (0.02, 0.25))
        can_held = ParamSetup.setup_blue_can("can2", (0.02,0.25))
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterObstructsHolding("test_obstructs", [robot, rPose, rPose, can, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env, debug = True)
        self.assertEqual(pred.get_type(), "BaxterObstructsHolding")

        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        rPose.value = can.pose = np.array([[0],[0],[0]])
        can_held.pose = np.array([[.5],[.5],[0]])
        self.assertTrue(pred.test(0))
        # This Grandient test passes
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)

        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # This Grandient test passes
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move can to the center of the gripper (touching -> should recognize as collision)
        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        robot.rGripper = np.matrix([[const.GRIPPER_CLOSE_VALUE]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This Gradient test failed, because of discontinuity on gripper gradient
        # if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        robot.rGripper = np.matrix([[const.GRIPPER_OPEN_VALUE]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This Gradient test failed, because of discontinuity on gripper gradient
        # if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This Gradient test passed
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move caheldn into the robot arm, should have collision
        can.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient checks failed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        pred._plot_handles = []

        pred2 = baxter_predicates.BaxterObstructsHolding("test_obstructs_held", [robot, rPose, rPose, can_held, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env, debug = True)
        rPose.value = can_held.pose = can.pose = np.array([[0],[0],[0]])
        pred._param_to_body[can].set_pose(can.pose, can.rotation)
        self.assertTrue(pred2.test(0))
        can_held.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred2.test(0))
        # This Grandient test passed
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move can to the center of the gripper (touching -> should allow touching)
        robot.rGripper = np.matrix([[const.GRIPPER_CLOSE_VALUE]])
        can_held.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertFalse(pred2.test(0))
        self.assertTrue(pred2.test(0, negated = True))

        # This Gradient test fails ->failed link: l_finger_tip, r_finger_tip, r_gripper_palm
        # if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move can away from the gripper, no collision
        can_held.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred2.test(0))
        self.assertTrue(pred2.test(0, negated = True))
        # This Gradient test passed
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move caheldn into the robot arm, should have collision
        can_held.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This Gradient test failed -> failed link: r_gripper_l_finger, r_gripper_r_finger
        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

    def test_r_collides(self):

        # RCollides Robot Obstacle

        TEST_GRAD = False
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        table = ParamSetup.setup_box()
        test_env = ParamSetup.setup_env()
        # test_env.SetViewer("qtcoin")
        pred = baxter_predicates.BaxterRCollides("test_r_collides", [robot, table], ["Robot", "Table"], test_env, debug = True)
        # self.assertEqual(pred.get_type(), "RCollides")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        table.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        # Move can so that it collide with robot base
        table.pose = np.array([[0],[0],[1.5]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        # Move can away so there is no collision
        table.pose = np.array([[0],[2],[.75]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[0],[0],[3]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[0],[0],[-0.4]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test failed
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[1],[1],[.75]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.rotation = np.array([[.5,.5,-.5]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        table.pose = np.array([[.5],[.5],[2.2]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))

        table.pose = np.array([[.5],[1.45],[.5]])
        table.rotation = np.array([[0.8,0,0]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test is not passed
        # if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

    def test_eereachable_inv(self):
        # EEReachable Constants

        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()
        rPose = ParamSetup.setup_baxter_pose()
        ee_pose = ParamSetup.setup_ee_pose()

        pred = baxter_predicates.BaxterEEReachableInvPos("ee_reachable", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        pred2 = baxter_predicates.BaxterEEReachableInvRot("ee_reachable_rot", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        baxter = pred._param_to_body[robot]
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))

        ee_pose.value = np.array([[1.2, -0.1, 0.925]]).T
        ee_pose.rotation = np.array([[0,0,0]]).T
        ee_pos = ParamSetup.setup_green_can()
        ee_body = OpenRAVEBody(test_env, "EE_Pose", ee_pos.geom)
        ee_body.set_pose(ee_pose.value[:, 0], ee_pose.rotation[:, 0])

        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        robot.rArmPose = np.zeros((7,7))
        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        # Find IK Solution
        trajectory = []
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+3*const.RETREAT_DIST], [0,0,0], "right_arm")[0])       #s=3
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+2*const.RETREAT_DIST], [0,0,0], "right_arm")[0])       #s=2
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+const.RETREAT_DIST], [0,0,0], "right_arm")[0])         #s=1
        trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925], [0,0,0], "right_arm")[0])                      #s=0
        trajectory.append(baxter.get_ik_from_pose([1.2 - const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])               #s=-1
        trajectory.append(baxter.get_ik_from_pose([1.2-2*const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])             #s=-2
        trajectory.append(baxter.get_ik_from_pose([1.2-3*const.APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])             #s=-3
        trajectory = np.array(trajectory)

        robot.rArmPose = trajectory.T
        # Predicate should succeed in the grasping post at t=3,
        # EEreachableRot should always pass since rotation is right all the time
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred2.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred2.test(2))
        self.assertTrue(pred.test(3))
        self.assertTrue(pred2.test(3))
        # Since finding ik introduced some error, we relax the tolerance to 1e-3
        if const.TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(3), True, 1e-3)

        if const.TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(3), True, 1e-3)




    def test_basket_ee_reachable(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter_basket_grasp.domain',
                       '../domains/baxter_domain/baxter_probs/basket_env.prob')
        env = problem.env
        robot = params['baxter']
        basket = params['basket']
        table = params['table']
        robot_pose = params['robot_init_pose']
        ee_left = params['ee_left']
        ee_right = params['ee_right']

        left_pos_pred = baxter_predicates.BaxterEEReachableVerLeftPos("basket_ee_reachable_left_pos", [robot, robot_pose, ee_left], ["Robot", "RobotPose", "EEPose"], env)
        right_pos_pred = baxter_predicates.BaxterEEReachableVerRightPos("basket_ee_reachable_right_pos", [robot, robot_pose, ee_left], ["Robot", "RobotPose", "EEPose"], env)

        left_rot_pred = baxter_predicates.BaxterEEReachableVerLeftPos("basket_ee_reachable_left_rot", [robot, robot_pose, ee_left], ["Robot", "RobotPose", "EEPose"], env)
        right_rot_pred = baxter_predicates.BaxterEEReachableVerRightPos("basket_ee_reachable_right_rot", [robot, robot_pose, ee_left], ["Robot", "RobotPose", "EEPose"], env)

        # Predicates are not initialized
        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))
        # Sample Grasping Pose
        offset = [0,0.317,0]
        basket_pos = basket.pose.flatten()
        ee_left.value = (basket_pos + offset).reshape((3, 1))
        ee_left.rotation = np.array([[0,np.pi/2,0]]).T
        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rArmPose = np.zeros((7,7))
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        basket.pose = np.repeat(basket.pose, 7, axis=1)
        basket.rotation = np.repeat(basket.rotation, 7, axis=1)
        table.pose = np.repeat(basket.pose, 7, axis=1)
        table.rotation = np.repeat(basket.rotation, 7, axis=1)

        # initialized poses are not right
        self.assertFalse(left_pos_pred.test(3))
        self.assertFalse(right_pos_pred.test(3))
        self.assertFalse(left_rot_pred.test(3))
        self.assertFalse(right_rot_pred.test(3))
        # Find IK Solution of a proper EEReachable Trajectory
        left_trajectory = []
        baxter_body = robot.openrave_body
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory = np.array(left_trajectory)
        robot.lArmPose = left_trajectory.T

        right_trajectory = []
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory = np.array(right_trajectory)
        robot.rArmPose = right_trajectory.T

        # Predicate should succeed in the grasping post at t=3,
        # EEreachableRot should always pass since rotation is right all the time
        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_pos_pred.test(1))
        self.assertFalse(right_pos_pred.test(1))
        self.assertFalse(left_pos_pred.test(2))
        self.assertFalse(right_pos_pred.test(2))
        self.assertTrue(left_pos_pred.test(3))
        self.assertFalse(right_pos_pred.test(3))

        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))
        self.assertFalse(left_rot_pred.test(1))
        self.assertFalse(right_rot_pred.test(1))
        self.assertFalse(left_rot_pred.test(2))
        self.assertFalse(right_rot_pred.test(2))
        self.assertTrue(left_rot_pred.test(3))
        self.assertFalse(right_rot_pred.test(3))

        # Since finding ik introduced some error, we relax the tolerance to 1e-3
        if const.TEST_GRAD:
            left_pos_pred.expr.expr.grad(left_pos_pred.get_param_vector(3), True, 1e-3)
        if const.TEST_GRAD:
            right_pos_pred.expr.expr.grad(right_pos_pred.get_param_vector(3), True, 1e-3)

        if const.TEST_GRAD:
            left_rot_pred.expr.expr.grad(left_rot_pred.get_param_vector(3), True, 1e-3)
        if const.TEST_GRAD:
            right_rot_pred.expr.expr.grad(right_rot_pred.get_param_vector(3), True, 1e-3)

    def test_basket_in_gripper(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter_basket_grasp.domain',
                       '../domains/baxter_domain/baxter_probs/basket_env.prob')
        env = problem.env
        robot = params['baxter']
        basket = params['basket']
        table = params['table']
        robot_pose = params['robot_init_pose']
        ee_left = params['ee_left']
        ee_right = params['ee_right']
        baxter_body = robot.openrave_body

        pos_pred = baxter_predicates.BaxterBasketInGripperPos("BasketInGripper", [robot, basket], ["Robot", "Basket"], env)
        rot_pred = baxter_predicates.BaxterBasketInGripperRot("BasketInGripperRot", [robot, basket], ["Robot", "Basket"], env)
        self.assertFalse(pos_pred.test(0))
        self.assertFalse(rot_pred.test(0))

        offset = [0,0.317,0]
        basket_pos = basket.pose.flatten()
        ee_left.value = (basket_pos + offset).reshape((3, 1))
        ee_left.rotation = np.array([[0,np.pi/2,0]]).T
        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rArmPose = np.zeros((7,7))
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        basket.pose = np.repeat(basket.pose, 7, axis=1)
        basket.rotation = np.repeat(basket.rotation, 7, axis=1)
        table.pose = np.repeat(basket.pose, 7, axis=1)
        table.rotation = np.repeat(basket.rotation, 7, axis=1)

        self.assertFalse(pos_pred.test(3))
        self.assertFalse(rot_pred.test(3))

        left_trajectory = []
        baxter_body = robot.openrave_body
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory = np.array(left_trajectory)
        robot.lArmPose = left_trajectory.T

        self.assertFalse(pos_pred.test(3))
        self.assertFalse(rot_pred.test(3))

        right_trajectory = []
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory = np.array(right_trajectory)
        robot.rArmPose = right_trajectory.T

        self.assertFalse(pos_pred.test(0))
        self.assertFalse(pos_pred.test(1))
        self.assertFalse(pos_pred.test(2))
        self.assertTrue(pos_pred.test(3))
        self.assertFalse(pos_pred.test(4))
        self.assertFalse(pos_pred.test(5))
        self.assertFalse(pos_pred.test(6))

        self.assertTrue(rot_pred.test(0))
        self.assertTrue(rot_pred.test(1))
        self.assertTrue(rot_pred.test(2))
        self.assertTrue(rot_pred.test(3))
        self.assertTrue(rot_pred.test(4))
        self.assertTrue(rot_pred.test(5))
        self.assertTrue(rot_pred.test(6))

        if const.TEST_GRAD:
            pos_pred.expr.expr.grad(pos_pred.get_param_vector(3), True, 1e-3)

        if const.TEST_GRAD:
            rot_pred.expr.expr.grad(rot_pred.get_param_vector(3), True, 1e-3)
