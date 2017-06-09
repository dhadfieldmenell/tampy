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
        robot.time = np.zeros((1, 4))
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
        lrA_l, rA_m = pred.lower_limit[7:14], pred.joint_step[7:14]

        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[0,2,3,4,5,6,7]])
        robot.time = np.zeros((1, 7))
        # Arm pose is valid in the timestep: 0,1,2,3
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))

        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))

        # Gripper pose is valid in the timestep: 0,1,3,4,5
        robot.rGripper = np.matrix([0, 0.02, 0, 0.02, 0, 0.02, 0]).reshape((1, 7))

        robot.lGripper = np.matrix([0, 0.02, 0, 0.02, 0, 0.02, 0]).reshape((1, 7))
        # Thus only timestep 1, 2, and 3 are valid
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertTrue(pred.test(2))
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
        robot.pose = np.zeros((1,7))
        # timestep 0, 3, 6 should fail
        robot.rArmPose = np.hstack((lrA_l+(joint_factor+1)*rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l-1*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+(joint_factor*10)*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1, 7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        robot.time = np.zeros((1, 7))
        # Thus timestep 1, 3, 6 should fail
        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertFalse(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))

    def test_gripper_value_constraint(self):

        robot = ParamSetup.setup_baxter()
        ee_left = ParamSetup.setup_ee_pose("ee_left")
        ee_right = ParamSetup.setup_ee_pose("ee_right")
        target = ParamSetup.setup_target()
        env = ParamSetup.setup_env()
        # Creating predicates for all gripper constraints family
        left_close_pred = baxter_predicates.BaxterCloseGripperLeft("test_close_lGripper", [robot, ee_left, target], ["Robot", "EEPose", "Target"])
        self.assertEqual(left_close_pred.get_type(), "BaxterCloseGripperLeft")
        right_close_pred = baxter_predicates.BaxterCloseGripperRight("test_close_rGripper", [robot, ee_left, target], ["Robot", "EEPose", "Target"])
        self.assertEqual(right_close_pred.get_type(), "BaxterCloseGripperRight")
        left_open_pred = baxter_predicates.BaxterOpenGripperLeft("test_open_lGripper", [robot, ee_left, target], ["Robot", "EEPose", "Target"])
        self.assertEqual(left_open_pred.get_type(), "BaxterOpenGripperLeft")
        right_open_pred = baxter_predicates.BaxterOpenGripperRight("test_open_rGripper", [robot, ee_left, target], ["Robot", "EEPose", "Target"])
        self.assertEqual(right_open_pred.get_type(), "BaxterOpenGripperRight")
        both_close_pred = baxter_predicates.BaxterCloseGrippers("test_close_both", [robot, ee_left, ee_right, target], ["Robot", "EEPose", "EEPose", "Target"])
        self.assertEqual(both_close_pred.get_type(), "BaxterCloseGrippers")
        both_open_pred = baxter_predicates.BaxterOpenGrippers("test_open_both", [robot, ee_left, ee_right, target], ["Robot", "EEPose", "EEPose", "Target"])
        self.assertEqual(both_open_pred.get_type(), "BaxterOpenGrippers")

        # EEPose and target is not yet initialized, thus no pred shall pass
        self.assertFalse(left_close_pred.test(0))
        self.assertFalse(right_close_pred.test(0))
        self.assertFalse(left_open_pred.test(0))
        self.assertFalse(right_open_pred.test(0))
        self.assertFalse(both_close_pred.test(0))
        self.assertFalse(both_open_pred.test(0))

        ee_left.value = np.zeros((3, 1))
        ee_right.value = np.zeros((3, 1))
        target.value = np.zeros((3, 1))
        gClose = const.GRIPPER_CLOSE_VALUE
        gOpen = const.GRIPPER_OPEN_VALUE
        robot.lGripper = np.array([[gOpen,gOpen ,gClose ,gClose]])
        robot.rGripper = np.array([[gOpen,gClose ,gOpen ,gClose]])
        robot.lArmPose = np.zeros((7,4))
        robot.rArmPose = np.zeros((7,4))
        robot.pose = np.zeros((1,4))
        robot.time = np.zeros((1,4))

        self.assertFalse(left_close_pred.test(0))
        self.assertFalse(left_close_pred.test(1))
        self.assertTrue(left_close_pred.test(2))
        self.assertTrue(left_close_pred.test(3))

        self.assertFalse(right_close_pred.test(0))
        self.assertTrue(right_close_pred.test(1))
        self.assertFalse(right_close_pred.test(2))
        self.assertTrue(right_close_pred.test(3))

        self.assertTrue(left_open_pred.test(0))
        self.assertTrue(left_open_pred.test(1))
        self.assertFalse(left_open_pred.test(2))
        self.assertFalse(left_open_pred.test(3))

        self.assertTrue(right_open_pred.test(0))
        self.assertFalse(right_open_pred.test(1))
        self.assertTrue(right_open_pred.test(2))
        self.assertFalse(right_open_pred.test(3))

        self.assertFalse(both_close_pred.test(0))
        self.assertFalse(both_close_pred.test(1))
        self.assertFalse(both_close_pred.test(2))
        self.assertTrue(both_close_pred.test(3))

        self.assertTrue(both_open_pred.test(0))
        self.assertFalse(both_open_pred.test(1))
        self.assertFalse(both_open_pred.test(2))
        self.assertFalse(both_open_pred.test(3))

    def test_in_gripper_left(self):
        test_env = ParamSetup.setup_env()
        robot = ParamSetup.setup_baxter()
        robot_body = OpenRAVEBody(test_env, robot.name, robot.geom)
        robot.openrave_body = robot_body
        obj = ParamSetup.setup_blue_can()
        obj_body = OpenRAVEBody(test_env, obj.name, obj.geom)
        obj.openrave_body = obj_body

        in_gripper_left = baxter_predicates.BaxterInGripperLeft("test_in_gripper_left", [robot, obj], ["Robot", "Can"], test_env)
        self.assertEqual(in_gripper_left.get_type(), "BaxterInGripperLeft")
        self.assertFalse(in_gripper_left.test(0))
        obj.pose = np.array([[0,0,0]]).T
        self.assertFalse(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        obj.pose = np.array([[0.96897233, 1.10397558, 1.006976]]).T
        self.assertTrue(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        robot.lArmPose = np.array([[0,-np.pi/4,np.pi/4,np.pi/2,-np.pi/4,-np.pi/4,0]]).T
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        self.assertFalse(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        obj.pose = np.array([[ 0.1840567 ,  1.22269958,  1.21028557]]).T
        self.assertFalse(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        obj.rotation = np.array([[-1.27706534, 0.25268026, -2.98976055]]).T
        self.assertTrue(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)
        obj.pose = np.array([[0.2, 0.1, 1.21]]).T
        self.assertFalse(in_gripper_left.test(0))
        if const.TEST_GRAD: in_gripper_left.expr.expr.grad(in_gripper_left.get_param_vector(0), True, const.TOL)

    def test_in_gripper_right(self):

        # InGripper, Robot, Item

        test_env = ParamSetup.setup_env()
        robot = ParamSetup.setup_baxter()
        robot_body = OpenRAVEBody(test_env, robot.name, robot.geom)
        robot.openrave_body = robot_body
        obj = ParamSetup.setup_blue_can()
        obj_body = OpenRAVEBody(test_env, obj.name, obj.geom)
        obj.openrave_body = obj_body

        in_gripper_right = baxter_predicates.BaxterInGripperRight("test_in_gripper_right", [robot, obj], ["Robot", "Can"], test_env)
        self.assertEqual(in_gripper_right.get_type(), "BaxterInGripperRight")
        self.assertFalse(in_gripper_right.test(0))

        obj.pose = np.array([[0,0,0]]).T
        # initialized pose value is not right
        self.assertFalse(in_gripper_right.test(0))
        # check the gradient of the implementations (correct)
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        # Now set can's pose and rotation to be the right things
        obj.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        # A new robot arm pose
        robot.rArmPose = np.array([[0,-np.pi/4,np.pi/4,np.pi/2,-np.pi/4,-np.pi/4,0]]).T
        self.assertFalse(in_gripper_right.test(0))

        # Only the pos is correct, rotation is not yet right
        obj.pose = np.array([[1.08769922, -0.31906039,  1.21028557]]).T
        self.assertFalse(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        obj.rotation = np.array([[-2.84786534,  0.25268026, -2.98976055]]).T
        self.assertTrue(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        # now rotate robot basepose
        robot.pose = np.array([[np.pi/3]]).T
        self.assertFalse(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        obj.pose = np.array([[0.82016401,  0.78244496,  1.21028557]]).T
        self.assertFalse(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)
        obj.rotation = np.array([[-1.80066778,  0.25268026, -2.98976055]]).T
        self.assertTrue(in_gripper_right.test(0))
        if const.TEST_GRAD: in_gripper_right.expr.expr.grad(in_gripper_right.get_param_vector(0), True, const.TOL)

    def test_in_gripper_basket(self):
        test_env = ParamSetup.setup_env()
        robot = ParamSetup.setup_baxter()
        robot_body = OpenRAVEBody(test_env, robot.name, robot.geom)
        robot.openrave_body = robot_body
        basket = ParamSetup.setup_basket()
        basket_body = OpenRAVEBody(test_env, basket.name, basket.geom)
        basket.openrave_body = basket_body

        in_gripper_basket = baxter_predicates.BaxterBasketInGripper("test_in_gripper_basket", [robot, basket], ["Robot", "Basket"], test_env)
        self.assertEqual(in_gripper_basket.get_type(), "BaxterBasketInGripper")
        self.assertFalse(in_gripper_basket.test(0))

        offset = [0,0.317,0]
        basket.pose = np.array([[0.75, 0.02, 0.81]]).T
        basket.rotation = np.array([[np.pi/2, 0, np.pi/2]]).T
        basket_pos = basket.pose.flatten()
        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rArmPose = np.zeros((7,7))
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        robot.time = np.zeros((1,7))
        basket.pose = np.repeat(basket.pose, 7, axis=1)
        basket.rotation = np.repeat(basket.rotation, 7, axis=1)

        self.assertFalse(in_gripper_basket.test(3))

        left_trajectory = []
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(robot_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory = np.array(left_trajectory)
        robot.lArmPose = left_trajectory.T

        self.assertFalse(in_gripper_basket.test(0))
        self.assertFalse(in_gripper_basket.test(1))
        self.assertFalse(in_gripper_basket.test(2))
        self.assertFalse(in_gripper_basket.test(3))
        self.assertFalse(in_gripper_basket.test(4))
        self.assertFalse(in_gripper_basket.test(5))
        self.assertFalse(in_gripper_basket.test(6))

        right_trajectory = []
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(robot_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 3*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory = np.array(right_trajectory)
        robot.rArmPose = right_trajectory.T

        self.assertFalse(in_gripper_basket.test(0))
        self.assertFalse(in_gripper_basket.test(1))
        self.assertFalse(in_gripper_basket.test(2))
        self.assertTrue(in_gripper_basket.test(3))
        self.assertFalse(in_gripper_basket.test(4))
        self.assertFalse(in_gripper_basket.test(5))
        self.assertFalse(in_gripper_basket.test(6))

        basket.rotation[:, 3] = [0,0,0]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [np.pi/2,0,0]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [0,np.pi/2,0]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [0,0,np.pi/2]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [np.pi/2,np.pi/2,0]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [0,np.pi/2,np.pi/2]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [np.pi/2,np.pi/2,np.pi/2]
        self.assertFalse(in_gripper_basket.test(3))
        basket.rotation[:, 3] = [np.pi/2,0,np.pi/2]
        self.assertTrue(in_gripper_basket.test(3))


        if const.TEST_GRAD:
            in_gripper_basket.expr.expr.grad(in_gripper_basket.get_param_vector(3), True, 1e-3)


    def test_in_gripper_washer(self):
        test_env = ParamSetup.setup_env()
        robot = ParamSetup.setup_baxter()
        robot_body = OpenRAVEBody(test_env, robot.name, robot.geom)
        robot.openrave_body = robot_body
        washer = ParamSetup.setup_washer()
        washer_body = OpenRAVEBody(test_env, washer.name, washer.geom)
        washer.openrave_body = washer_body
        in_gripper_washer =  baxter_predicates.BaxterWasherInGripper("test_in_gripper_washer", [robot, washer], ["Robot", "Washer"], test_env)
        self.assertEqual(in_gripper_washer.get_type(), "BaxterWasherInGripper")
        self.assertFalse(in_gripper_washer.test(0))

        rel_pt = np.array([-0.035,0.055,-0.1])
        link = washer_body.env_body.GetLink("washer_handle")
        # test_env.SetViewer("qtcoin")

        washer.pose[:, 0] = [0.505, 0.961, 1.498]
        handle_pos = link.GetTransform().dot(np.r_[rel_pt, 1])[:3]
        robot.pose = np.array([[np.pi/4]])
        robot_body.set_pose([0,0,np.pi/4])
        arm_pose = robot_body.get_ik_from_pose(handle_pos, [np.pi/2,0,0], "left_arm")
        robot.lArmPose = arm_pose[0].reshape((7, 1))
        self.assertTrue(in_gripper_washer.test(0))

        def test_grasping_pose(washer_pose, pose, door):
            robot.pose[:, 0] = pose
            washer.pose[:, 0] = washer_pose
            washer.door = np.array([[door]])
            self.assertFalse(in_gripper_washer.test(0))
            handle_pos = link.GetTransform().dot(np.r_[rel_pt, 1])[:3]
            handle_rot = OpenRAVEBody.obj_pose_from_transform(link.GetTransform())[3:] + [-np.pi/2, 0, 0]
            arm_pose = robot_body.get_ik_from_pose(handle_pos, handle_rot, "left_arm")
            robot.lArmPose = arm_pose[0].reshape((7, 1))
            self.assertTrue(in_gripper_washer.test(0))
            robot_body.set_dof({'lArmPose': arm_pose[0]})

        test_grasping_pose([0.505, 1.261, 1.498], np.pi/4, 0)
        test_grasping_pose([0.505, 1.361, 1.498], np.pi/4, 0)
        test_grasping_pose([0.505, 1.461, 1.498], np.pi/4, 0)
        test_grasping_pose([0.605, 1.361, 1.498], np.pi/4, -np.pi/8)
        test_grasping_pose([0.805, 1.161, 1.498], np.pi/4, -4*np.pi/8)

        test_grasping_pose([0.605, 1.261, 1.498], np.pi/4, -0*np.pi/8)
        test_grasping_pose([0.605, 1.261, 1.498], np.pi/4, -1*np.pi/8)
        test_grasping_pose([0.605, 1.261, 1.498], np.pi/4, -2*np.pi/8)
        test_grasping_pose([0.605, 1.261, 1.498], np.pi/4, -3*np.pi/8)
        test_grasping_pose([0.605, 1.261, 1.498], np.pi/4, -4*np.pi/8)

        test_grasping_pose([0.730, 1.261, 1.498], np.pi/4, -0*np.pi/8)
        test_grasping_pose([0.730, 1.261, 1.498], np.pi/4, -1*np.pi/8)
        test_grasping_pose([0.730, 1.261, 1.498], np.pi/4, -2*np.pi/8)
        test_grasping_pose([0.730, 1.261, 1.498], np.pi/4, -3*np.pi/8)
        test_grasping_pose([0.730, 1.261, 1.498], np.pi/4, -4*np.pi/8)

        if const.TEST_GRAD:
            in_gripper_washer.expr.expr.grad(in_gripper_washer.get_param_vector(0), True, 1e-3)

    def test_ee_grasp_valid(self):
        test_env = ParamSetup.setup_env()
        ee_pose = ParamSetup.setup_ee_pose()
        washer = ParamSetup.setup_washer()
        washer_body = OpenRAVEBody(test_env, washer.name, washer.geom)
        washer.openrave_body = washer_body

        ee_grasp_valid =  baxter_predicates.BaxterEEGraspValid("test_ee_grasp_valid", [ee_pose, washer], ["EEPose", "Washer"], test_env)
        self.assertEqual(ee_grasp_valid.get_type(), "BaxterEEGraspValid")
        self.assertFalse(ee_grasp_valid.test(0))

        rel_pt = np.array([-0.035,0.055,-0.1])
        link = washer_body.env_body.GetLink("washer_handle")
        # test_env.SetViewer("qtcoin")

        washer.pose[:, 0] = [0.730, 1.261, 1.498]
        washer.door = np.array([[-np.pi/8]])
        washer_body.set_pose(washer.pose.flatten(), washer.rotation.flatten())
        washer_body.set_dof({"door": washer.door.flatten()})
        handle_pos = link.GetTransform().dot(np.r_[rel_pt, 1])[:3]
        ee_pose.value = handle_pos.reshape((3,1))
        ee_pose.rotation = np.zeros((3,1))
        if const.TEST_GRAD: ee_grasp_valid.expr.expr.grad(ee_grasp_valid.get_param_vector(0), True, const.TOL)
        ee_pose.rotation[:, 0] = [0, 0, np.pi/2]
        self.assertFalse(ee_grasp_valid.test(0))
        if const.TEST_GRAD: ee_grasp_valid.expr.expr.grad(ee_grasp_valid.get_param_vector(0), True, const.TOL)
        ee_pose.rotation[:, 0] = [0, np.pi/2 ,0]
        self.assertFalse(ee_grasp_valid.test(0))
        if const.TEST_GRAD: ee_grasp_valid.expr.expr.grad(ee_grasp_valid.get_param_vector(0), True, const.TOL)
        ee_pose.rotation[:, 0] = [np.pi/2 , 0, 0]
        self.assertTrue(ee_grasp_valid.test(0))
        if const.TEST_GRAD: ee_grasp_valid.expr.expr.grad(ee_grasp_valid.get_param_vector(0), True, const.TOL)
        ee_pose.value[:, 0] = [0.530, 1.261, 1.498]
        self.assertFalse(ee_grasp_valid.test(0))

        if const.TEST_GRAD: ee_grasp_valid.expr.expr.grad(ee_grasp_valid.get_param_vector(0), True, const.TOL)

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

    def test_basket_grasp(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter_basket_grasp.domain',
                       '../domains/baxter_domain/baxter_probs/basket_env.prob')
        env = problem.env
        robot = params['baxter']
        basket = params['basket']
        bask_targ = params['target']
        table = params['table']
        robot_pose = params['robot_init_pose']

        ee_left = params['ee_left']
        ee_right = params['ee_right']
        target = params['target']

        left_pos_pred = baxter_predicates.BaxterBasketGraspLeftPos("basket_grasp_left_pos", [ee_left, bask_targ], ["EEPose", "BasketTarget"], env)
        right_pos_pred = baxter_predicates.BaxterBasketGraspRightPos("basket_grasp_right_pos", [ee_right, bask_targ], ["EEPose", "BasketTarget"], env)
        left_rot_pred = baxter_predicates.BaxterBasketGraspLeftRot("basket_grasp_left_rot", [ee_left, bask_targ], ["EEPose", "BasketTarget"], env)
        right_rot_pred = baxter_predicates.BaxterBasketGraspRightRot("basket_grasp_right_rot", [ee_right, bask_targ], ["EEPose", "BasketTarget"], env)

        # Predicates are not initialized
        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))
        # Sample Grasping Pose
        offset = [0,0.317,0]
        basket_pos = basket.pose.flatten()

        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rArmPose = np.zeros((7,7))
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        basket.pose = np.repeat(basket.pose, 7, axis=1)
        basket.rotation = np.repeat(basket.rotation, 7, axis=1)
        table.pose = np.repeat(basket.pose, 7, axis=1)
        table.rotation = np.repeat(basket.rotation, 7, axis=1)

        ee_left.value = np.zeros((3,1))
        ee_left.rotation = np.zeros((3,1))

        ee_right.value = np.zeros((3,1))
        ee_right.rotation = np.zeros((3,1))
        # initialized poses are not right
        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))

        ee_left.value = (basket_pos + offset).reshape((3, 1))
        ee_left.rotation = np.array([[0,np.pi/2,0]]).T
        ee_right.value = (basket_pos - offset).reshape((3, 1))
        ee_right.rotation = np.array([[0,np.pi/2,0]]).T

        # EEreachableRot should always pass since rotation is right all the time
        self.assertTrue(left_pos_pred.test(0))
        self.assertTrue(right_pos_pred.test(0))
        self.assertTrue(left_rot_pred.test(0))
        self.assertTrue(right_rot_pred.test(0))

    def test_basket_grasp_valid(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter_basket_grasp.domain',
                       '../domains/baxter_domain/baxter_probs/basket_move.prob')

        env = problem.env
        robot = params['baxter']
        basket = params['basket']
        basket_targ = params['init_target']
        robot_pose = params['robot_init_pose']
        ee_left = params['grasp_ee_left']
        ee_right = params['grasp_ee_right']
        baxter_body = robot.openrave_body

        # viewer = OpenRAVEViewer.create_viewer(env)
        # objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        # viewer.draw(objLst, 0, 0.7)

        left_pos_pred = baxter_predicates.BaxterBasketGraspLeftPos("grasp_left_pos", [ee_left, basket_targ], ['EEPose', 'BasketTarget'])
        right_pos_pred = baxter_predicates.BaxterBasketGraspRightPos("grasp_right_pos", [ee_right, basket_targ], ['EEPose', 'BasketTarget'])

        left_rot_pred = baxter_predicates.BaxterBasketGraspLeftRot("grasp_left_rot", [ee_left, basket_targ], ['EEPose', 'BasketTarget'])
        right_rot_pred = baxter_predicates.BaxterBasketGraspRightRot("grasp_right_rot", [ee_right, basket_targ], ['EEPose', 'BasketTarget'])

        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))

        offset = [0,0.317,0]
        basket_pos = basket.pose.flatten()
        basket_targ.value = basket.pose
        basket_targ.rotation = basket.rotation

        ee_left.value = np.zeros((3,1))
        ee_left.rotation = np.zeros((3,1))
        ee_right.value = np.zeros((3,1))
        ee_right.rotation = np.zeros((3,1))
        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))

        ee_left.rotation = np.array([[0,np.pi/2,0]]).T
        ee_right.rotation = np.array([[0,np.pi/2,0]]).T

        self.assertFalse(left_pos_pred.test(0))
        self.assertFalse(right_pos_pred.test(0))
        self.assertTrue(left_rot_pred.test(0))
        self.assertTrue(right_rot_pred.test(0))

        ee_left.value = (basket_pos + offset).reshape((3, 1))
        ee_right.value = (basket_pos - offset).reshape((3, 1))
        ee_left.rotation = np.zeros((3,1))
        ee_right.rotation = np.zeros((3,1))

        self.assertTrue(left_pos_pred.test(0))
        self.assertTrue(right_pos_pred.test(0))
        self.assertFalse(left_rot_pred.test(0))
        self.assertFalse(right_rot_pred.test(0))

    def test_grippers_level(self):
        env = ParamSetup.setup_env()
        baxter = ParamSetup.setup_baxter()
        pred = baxter_predicates.BaxterGrippersLevel("test_grippers_level", [baxter], ["Robot"], env)
        # The Baxter begins with all joint angles set to 0
        self.assertTrue(pred.test(0))

        baxter_body = baxter.openrave_body
        baxter.lArmPose = np.array([baxter_body.get_ik_from_pose([.75, .05, .82], [0,np.pi/2,0], "left_arm")[0]]).T
        baxter.rArmPose = np.array([baxter_body.get_ik_from_pose([.75, .02, .79], [0,np.pi/2,0], "right_arm")[0]]).T
        self.assertFalse(pred.test(0))

        baxter.lArmPose = np.array([baxter_body.get_ik_from_pose([.3, 0, .8], [0,np.pi/2,0], "left_arm")[0]]).T
        baxter.rArmPose = np.array([baxter_body.get_ik_from_pose([.4, -.01, .8], [0,np.pi/2,0], "right_arm")[0]]).T
        self.assertTrue(pred.test(0))

        baxter.lArmPose = np.array([baxter_body.get_ik_from_pose([.65, .05, .83], [0,np.pi/2,0], "left_arm")[0]]).T
        baxter.rArmPose = np.array([baxter_body.get_ik_from_pose([.6, .05, .83], [0,np.pi/2,0], "right_arm")[0]]).T
        self.assertTrue(pred.test(0))

        baxter.lArmPose = np.array([baxter_body.get_ik_from_pose([.65, .5, .839], [0,np.pi/2,0], "left_arm")[0]]).T
        baxter.rArmPose = np.array([baxter_body.get_ik_from_pose([.65, -.2, .85], [0,np.pi/2,0], "right_arm")[0]]).T
        self.assertFalse(pred.test(0))

        baxter.lArmPose = np.array([baxter_body.get_ik_from_pose([.65, .14, .9], [0,np.pi/2,0], "left_arm")[0]]).T
        baxter.rArmPose = np.array([baxter_body.get_ik_from_pose([.65, .14, .9], [0,np.pi/2,0], "right_arm")[0]]).T
        self.assertTrue(pred.test(0))

        if const.TEST_GRAD:
            pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-3)

    def test_retiming(self):

        env = ParamSetup.setup_env()
        baxter = ParamSetup.setup_baxter()
        ee_vel = ParamSetup.setup_ee_vel()
        pred = baxter_predicates.BaxterEERetiming("test_retiming", [baxter, ee_vel], ["Robot", "EEVel"], env)
        # Since variables aren't defined yet
        self.assertFalse(pred.test(0))

        ee_vel.value = np.array([[0.1]]).T

        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_retiming: (BaxterEERetiming baxter ee_vel)' at the timestep.")

        body = baxter.openrave_body.env_body
        def get_ee_pose(body, arm):
            return body.GetManipulator(arm).GetTransform()[:3, 3]
        left_ee_pose = get_ee_pose(body, 'left_arm')
        right_ee_pose = get_ee_pose(body, 'right_arm')

        baxter.lArmPose = np.zeros((7, 5))
        baxter.lGripper = np.zeros((1, 5))
        baxter.rArmPose = np.zeros((7, 5))
        baxter.rGripper = np.zeros((1, 5))
        baxter.pose = np.zeros((1, 5))
        baxter.time = np.zeros((1, 5))
        self.assertTrue(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertTrue(pred.test(3))


        ee_vel.value = np.array([[const.APPROACH_DIST/2.]]).T
        basket_pos = np.array([0.65 , -0.283,  0.81])
        offset = [0, 0.317, 0]
        left_trajectory = []
        baxter_body = baxter.openrave_body
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST/2.], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST/2.], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 0*const.APPROACH_DIST/2.], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory = np.array(left_trajectory)
        baxter.lArmPose = left_trajectory.T

        right_trajectory = []
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.APPROACH_DIST/2.], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.APPROACH_DIST/2.], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 0*const.APPROACH_DIST/2.], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 1*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory.append(baxter_body.get_ik_from_pose(basket_pos - offset +
        [0,0, 2*const.RETREAT_DIST], [0,np.pi/2,0], "right_arm")[2])
        right_trajectory = np.array(right_trajectory)
        baxter.rArmPose = right_trajectory.T

        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertFalse(pred.test(3))
        # pred.expr.expr.grad(pred.get_param_vectbaxteror(3), True, 1e-3)

        baxter.time = np.array([[0, 1, 2, 3, 4]])
        self.assertTrue(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertTrue(pred.test(3))

        # pred.expr.expr.grad(pred.get_param_vector(3), True, 1e-3)
        ee_vel.value = np.array([[const.APPROACH_DIST/2.*2]]).T
        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertFalse(pred.test(3))

        # pred.expr.expr.grad(pred.get_param_vector(3), True, 1e-3)
        baxter.time = np.array([[0, 0.5, 1, 2, 3]])
        self.assertTrue(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertFalse(pred.test(3))
        # pred.expr.expr.grad(pred.get_param_vector(3), True, 1e-3)

        left_trajectory = []
        baxter_body = baxter.openrave_body
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 0*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 1*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory.append(baxter_body.get_ik_from_pose(basket_pos + offset +
        [0,0, 2*const.APPROACH_DIST], [0,np.pi/2,0], "left_arm")[4])
        left_trajectory = np.array(left_trajectory)
        baxter.lArmPose = left_trajectory.T

    def test_in_gripper_rot(self):
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

        viewer = OpenRAVEViewer.create_viewer(env)
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        viewer.draw(objLst, 0, 0.7)


        pred = baxter_predicates.BaxterBasketInGripperRot("BasketInGripperRot", [robot, basket], ["Robot", "Basket"], env)
        # self.assertFalse(pred.test(0))

        offset = [0,0.317,0]
        basket_pos = np.array([0.571, 0.017, 1.514])
        ee_left.value = (basket_pos + offset).reshape((3, 1))
        ee_left.rotation = np.array([[0,np.pi/2,0]]).T
        robot.lArmPose = np.zeros((7,1))
        robot.lGripper = np.ones((1, 1))*0.02
        robot.rArmPose = np.zeros((7,1))
        robot.rGripper = np.ones((1, 1))*0.02
        robot.pose = np.zeros((1,1))
        basket.pose = np.repeat(basket.pose, 1, axis=1)
        basket.rotation = np.repeat(basket.rotation, 1, axis=1)
        table.pose = np.repeat(table.pose, 1, axis=1)
        table.rotation = np.repeat(table.rotation, 1, axis=1)

        # self.assertFalse(pred.test(0))

        baxter_body = robot.openrave_body

        left_arm_pose = baxter_body.get_ik_from_pose(basket_pos + offset, [0,-np.pi/2,0], "left_arm")[0]

        right_arm_pose = baxter_body.get_ik_from_pose(basket_pos - offset, [0, -np.pi/2,0], "right_arm")[0]
        robot.lArmPose[:, 0] = left_arm_pose
        robot.rArmPose[:, 0] = right_arm_pose
        basket.openrave_body.set_pose(basket_pos, basket.rotation.flatten())

        self.assertFalse(pred.test(0))

        basket.pose = np.array([[0.571, 0.017,  0.90]]).T
        basket.openrave_body.set_pose([0.571, 0.017,  0.90], basket.rotation.flatten())
        left_arm_pose = baxter_body.get_ik_from_pose(basket.pose.flatten() + offset, [0,np.pi/2,0], "left_arm")[0]

        right_arm_pose = baxter_body.get_ik_from_pose(basket.pose.flatten() - offset , [0, np.pi/2,0], "right_arm")[0]
        robot.lArmPose[:, 0] = left_arm_pose
        robot.rArmPose[:, 0] = right_arm_pose
        baxter_body.set_dof({'lArmPose': left_arm_pose, 'rArmPose':right_arm_pose})
        import ipdb; ipdb.set_trace()
        self.assertTrue(pred.test(0))
