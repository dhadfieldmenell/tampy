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
