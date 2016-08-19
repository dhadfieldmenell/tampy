import unittest
from core.util_classes import robot_predicates, baxter_predicates, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.param_setup import ParamSetup
import numpy as np

class TestPR2Predicates(unittest.TestCase):

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
        robot.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        robot.lGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        rPose.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
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

    def test_baxter(self):
        # from core.util_classes.openrave_body import OpenRAVEBody
        # env = ParamSetup.setup_env()
        # env.SetViewer("qtcoin")
        # baxter = ParamSetup.setup_baxter()
        # body = OpenRAVEBody(env, baxter.name, baxter.geom)
        # import ipdb; ipdb.set_trace()
        pass

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
        # import ipdb; ipdb.set_trace()
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
        robot = ParamSetup.setup_pr2()
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterWithinJointLimit("test_joint_limit", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "BaxterWithinJointLimit")
        # Getting lowerbound and movement step
        llA_l, lA_m = pred.lower_limit[0:7], pred.joint_step[0:7]
        lrA_l, rA_m = pred.lower_limit[8:15], pred.joint_step[8:15]
        llG_l, lG_m = pred.lower_limit[7], pred.joint_step[7]
        lrG_l, rG_m = pred.lower_limit[15], pred.joint_step[15]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[0,2,3,4,5,6,7]])

        # timestep 6 should fail
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Thus timestep 1, 6 should fail
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))

    def test_in_gripper(self):
        tol = 1e-4
        TEST_GRAD = False
        # InGripper, Robot, Can
        robot = ParamSetup.setup_pr2()
        can = ParamSetup.setup_blue_can()
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterInGripperPos("InGripper", [robot, can], ["Robot", "Can"], test_env)
        pred2 = baxter_predicates.BaxterInGripperRot("InGripper_rot", [robot, can], ["Robot", "Can"], test_env)
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0,0,0]]).T
        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        # check the gradient of the implementations (correct)
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        # Now set can's pose and rotation to be the right things
        can.pose = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        # A new robot arm pose
        robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        # Only the pos is correct, rotation is not yet right
        can.pose = np.array([[0.59152062, -0.71105108,  1.05144139]]).T
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        can.rotation = np.array([[0.02484449, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        self.assertTrue(pred2.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        # now rotate robot basepose
        robot.pose = np.array([[0,0,np.pi/3]]).T
        self.assertFalse(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        can.pose = np.array([[0.91154861,  0.15674634,  1.05144139]]).T
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        can.rotation = np.array([[1.07204204, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred2.test(0))
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        can.rotation = np.array([[2.22529480e+00,   3.33066907e-16,  -5.23598776e-01]]).T
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(0))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)
        can.pose = np.array([[3.98707028e-01,   4.37093473e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations (correct)
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, tol)

        # testing example from grasp where predicate continues to fail,
        # confirmed that Jacobian is fine.
        robot.pose = np.array([-0.52014383,  0.374093  ,  0.04957286]).reshape((3,1))
        robot.backHeight = np.array([  2.79699865e-13]).reshape((1,1))
        robot.lGripper = np.array([ 0.49999948]).reshape((1,1))
        robot.rGripper = np.array([ 0.53268086]).reshape((1,1))
        robot.rArmPose = np.array([-1.39996414, -0.31404741, -1.42086452, -1.72304084, -1.16688324,
                                   -0.20148917, -3.33438558]).reshape((7,1))
        robot.lArmPose = np.array([ 0.05999948,  1.24999946,  1.78999946, -1.68000049, -1.73000049,
                                   -0.10000051, -0.09000051]).reshape((7,1))
        can.pose = np.array([-0.        , -0.08297436,  0.925     ]).reshape((3,1))
        can.rotation = np.array([-0., -0., -0.]).reshape((3,1))
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-4)