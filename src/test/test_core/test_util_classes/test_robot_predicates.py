import unittest
from core.internal_repr import parameter
from core.util_classes import robot_predicates, pr2_predicates_fixed, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.table import Table
from core.util_classes.robots import PR2
from core.util_classes.box import Box
from core.util_classes.param_setup import ParamSetup
from openravepy import Environment
from sco import expr
import numpy as np


class TestRobotPredicates(unittest.TestCase):

    # Begin of the test
    def test_at(self):

        # At, Can, Target

        can = ParamSetup.setup_blue_can()
        target = ParamSetup.setup_target()
        pred = robot_predicates.At("testpred", [can, target], ["Can", "Target"])
        self.assertEqual(pred.get_type(), "At")
        # target is a symbol and doesn't have a value yet
        self.assertFalse(pred.test(time=0))
        can.pose = np.array([[3, 3, 5, 6],
                                  [6, 6, 7, 8],
                                  [6, 6, 4, 2]])
        can.rotation = np.zeros((3, 4))
        target.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())
        # Test timesteps
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At blue_can target)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At blue_can target)'.")
        #
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = robot_predicates.At("testpred", [can, sym], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At blue_can sym)'.")
        # Test rotation
        can.rotation = np.array([[1,2,3,4],
                                      [2,3,4,5],
                                      [3,4,5,6]])

        target.rotation = np.array([[2],[3],[4]])

        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_robot_at(self):

        # RobotAt, Robot, RobotPose

        robot = ParamSetup.setup_pr2()
        rPose = ParamSetup.setup_pr2_pose()
        pred = pr2_predicates_fixed.PR2RobotAt("testRobotAt", [robot, rPose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "PR2RobotAt")
        # Robot and RobotPose are initialized to the same pose
        self.assertTrue(pred.test(0))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (PR2RobotAt pr2 pr2_pose)'.")
        robot.pose = np.array([[3, 4, 5, 3],
                               [6, 5, 7, 6],
                               [6, 3, 4, 6]])
        rPose.value = np.array([[3, 4, 5, 6],
                                [6, 5, 7, 1],
                                [6, 3, 9, 2]])
        self.assertTrue(pred.is_concrete())
        robot.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        robot.lGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        rPose.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        robot.backHeight = np.matrix([0.2, 0.29, 0.18, 0.2])
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
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (PR2RobotAt pr2 pr2_pose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (PR2RobotAt pr2 pr2_pose)'.")

        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_is_mp(self):
        robot = ParamSetup.setup_pr2()
        test_env = ParamSetup.setup_env()
        pred = pr2_predicates.IsMP("test_isMP", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "PR2IsMP")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (PR2IsMP pr2)' at the timestep.")
        # Getting lowerbound and movement step
        lbH_l, bH_m = pred.lower_limit[0], pred.joint_step[0]
        llA_l, lA_m = pred.lower_limit[1:8], pred.joint_step[1:8]
        lrA_l, rA_m = pred.lower_limit[9:16], pred.joint_step[9:16]
        llG_l, lG_m = pred.lower_limit[8], pred.joint_step[8]
        lrG_l, rG_m = pred.lower_limit[16], pred.joint_step[16]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[1,2,3,4,5,6,7],
                               [0,2,3,4,5,6,7],
                               [1,2,3,4,5,6,7]])

        # Arm pose is valid in the timestep: 0,1,2,3
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))

        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))

        # Gripper pose is valid in the timestep: 0,1,3,4,5
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Back height pose is always valid
        robot.backHeight = np.matrix([bH_m, bH_m, bH_m, bH_m, bH_m, bH_m, bH_m]).reshape((1,7))
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
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (PR2IsMP pr2)' at the timestep.")

    def test_within_joint_limit(self):
        robot = ParamSetup.setup_pr2()
        test_env = ParamSetup.setup_env()
        pred = pr2_predicates.WithinJointLimit("test_joint_limit", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "PR2WithinJointLimit")
        # Getting lowerbound and movement step
        lbH_l, bH_m = pred.lower_limit[0], pred.joint_step[0]
        llA_l, lA_m = pred.lower_limit[1:8], pred.joint_step[1:8]
        lrA_l, rA_m = pred.lower_limit[9:16], pred.joint_step[9:16]
        llG_l, lG_m = pred.lower_limit[8], pred.joint_step[8]
        lrG_l, rG_m = pred.lower_limit[16], pred.joint_step[16]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[1,2,3,4,5,6,7],
                               [0,2,3,4,5,6,7],
                               [1,2,3,4,5,6,7]])

        # timestep 6 should fail
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # timestep 3 shold fail
        robot.backHeight = np.matrix([bH_m, bH_m, bH_m, -bH_m, bH_m, bH_m, bH_m]).reshape((1,7))
        # Thus timestep 1, 3, 6 should fail
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertFalse(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))
