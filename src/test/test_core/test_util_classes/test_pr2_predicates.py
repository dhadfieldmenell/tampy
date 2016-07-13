import unittest
from core.internal_repr import parameter
from core.util_classes.matrix import Vector3d, PR2PoseVector
from core.util_classes import pr2_predicates
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.can import BlueCan
from core.util_classes.pr2 import PR2
from openravepy import Environment
from sco import expr
import numpy as np

## exprs for testing
e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestPR2Predicates(unittest.TestCase):


    def test_expr_at(self):

        # At, Can, Location

        attrs = {"name": ["can"], "geom": (1,1), "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)

        attrs = {"name": ["location"], "value": ["undefined"], "_type": ["Location"]}
        attr_types = {"name": str, "value": Vector3d, "_type": str}
        location = parameter.Symbol(attrs, attr_types)

        pred = pr2_predicates.At("testpred", [can, location], ["Can", "Location"])
        self.assertEqual(pred.get_type(), "At")

        self.assertFalse(pred.test(time=400))
        can.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8], [6, 3, 4, 2]])
        # location is a symbol and doesn't have a value yet
        #self.assertRaises(PredicateException, pred.test, time = 400)
        location.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())

        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = pr2_predicates.At("testpred", [can, sym], ["Can", "Location"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")

    def test_expr_robot_at(self):

        # RobotAt, Robot, RobotPose

        attrs = {"name": ["pr2"], "geom": ['../models/pr2/pr2.zae'], "pose": ["undefined"], "_type": ["Robot"]}
        attr_types = {"name": str, "geom": PR2, "pose": PR2PoseVector, "_type": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["funnyPose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": PR2PoseVector, "_type": str}
        robot_pose = parameter.Symbol(attrs, attr_types)

        pred = pr2_predicates.RobotAt("testRobotAt", [robot, robot_pose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "RobotAt")

        self.assertFalse(pred.test(time=400))
        robot.pose = np.array([[3, 4, 5, 6],
                            [6, 5, 7, 8],
                            [6, 3, 4, 2]])
        # p2 is a symbol and doesn't have a value yet
        #self.assertRaises(PredicateException, pred.test, time=400)
        robot_pose.value = np.array([[3, 4, 5, 6],
                            [6, 5, 7, 1],
                            [6, 3, 9, 2]])
        self.assertTrue(pred.is_concrete())

        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_expr_is_gp(self):

        # IsGP, Robot, RobotPose, Can

        attrs = {"name": ["robot"], "pose": ["undefined"], "_type": ["Robot"], "geom": ['../models/pr2/pr2.zae']}
        attr_types = {"name": str, "pose": PR2PoseVector, "_type": str, "geom": PR2}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["rPose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": PR2PoseVector, "_type": str}
        rPose = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["can"], "pose": ["undefined"], "_type": ["Can"], "geom": (1, 2)}
        attr_types = {"name": str, "pose": Vector3d, "_type": str, "geom": BlueCan}
        can = parameter.Object(attrs, attr_types)
        env = Environment()
        pred = pr2_predicates.IsGP("testgp", [robot, rPose, can], ["Robot", "RobotPose", "Can"], env)
        self.assertEqual(pred.get_type(), "IsGP")
        self.assertFalse(pred.test(time=400))



    def test_expr_is_pdp(self):

        # IsPDP, Robot, RobotPose, Can, Location


        attrs = {"name": ["robot"], "pose": ["undefined"], "_type": ["Robot"], "geom": ['../models/pr2/pr2.zae']}
        attr_types = {"name": str, "pose": PR2PoseVector, "_type": str, "geom": PR2}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["rPose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": PR2PoseVector, "_type": str}
        rPose = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["can"], "pose": ["undefined"], "_type": ["Can"], "geom": (1, 2)}
        attr_types = {"name": str, "pose": Vector3d, "_type": str, "geom": BlueCan}
        can = parameter.Object(attrs, attr_types)

        attrs = {"name": ["location"], "value": ["undefined"], "_type": ["Location"]}
        attr_types = {"name": str, "value": Vector3d, "_type": str}
        location = parameter.Symbol(attrs, attr_types)
        env = Environment()
        pred = pr2_predicates.IsPDP("testpdp", [robot, rPose, can, location], ["Robot", "RobotPose", "Can", "Location"], env)
        self.assertEqual(pred.get_type(), "IsPDP")



    def test_expr_is_in_gripper(self):

        # InGripper, Robot, Can, Grasp

        attrs = {"geom": ['../models/pr2/pr2.zae'], "pose": ["undefined"], "_type": ["Robot"], "name": ["pr2"]}
        attr_types = {"geom": PR2, "pose": Vector3d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"geom": (1, 1), "pose": ["undefined"], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": BlueCan, "pose": Vector3d, "_type": str, "name": str}
        can = parameter.Object(attrs, attr_types)

        attrs = {"value": ["undefined"], "_type": ["Grasp"], "name": ["grasp"]}
        attr_types = {"value": Vector3d, "_type": str, "name": str}
        grasp = parameter.Symbol(attrs, attr_types)

        pred = pr2_predicates.InGripper("InGripper", [robot, can, grasp], ["Robot", "Can", "Grasp"])
        self.assertEqual(pred.get_type(), "InGripper")


    # TODO: test other predicates

if __name__ == "__main__":
    unittest.main()
