import unittest
from core.internal_repr import parameter
from core.util_classes.matrix import Vector3d, PR2PoseVector
from core.util_classes import pr2_predicates
from errors_exceptions import PredicateException
from core.util_classes.can import BlueCan
from core.util_classes.pr2 import PR2
from sco import expr
import numpy as np

## exprs for testing
e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestPR2Predicates(unittest.TestCase):

    def test_expr_at(self):
        attrs = {"name": ["can"], "geom": (1,1), "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": Vector3d, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["location"], "value": ["undefined"], "_type": ["Location"]}
        attr_types = {"name": str, "value": Vector3d, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)

        pred = pr2_predicates.At("testpred", [p1, p2], ["Can", "Location"])
        self.assertEqual(pred.get_type(), "At")
        self.assertFalse(pred.test(time=400))
        p1.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8], [6, 3, 4, 2]])
        # p2 is a symbol and doesn't have a value yet
        self.assertRaises(PredicateException, pred.test, time = 400)
        p2.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(PredicateException) as cm:
            pred = pr2_predicates.At("testpred", [p1, p3], ["Can", "Location"])
        self.assertEqual(cm.exception.message, "attribute type not supported")


    def test_expr_robot_at(self):
        attrs = {"name": ["pr2"], "geom": [1], "pose": ["undefined"], "_type": ["Robot"]}
        attr_types = {"name": str, "geom": PR2, "pose": PR2PoseVector, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["funnyPose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": PR2PoseVector, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)

        pred = pr2_predicates.RobotAt("testRobotAt", [p1, p2], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "RobotAt")
        self.assertFalse(pred.test(time=400))
        p1.pose = np.array([[3, 4, 5, 6],
                            [6, 5, 7, 8],
                            [6, 3, 4, 2],
                            [7, 2, 4, 5],
                            [1, 4, 9, 4]])
        # p2 is a symbol and doesn't have a value yet
        self.assertRaises(PredicateException, pred.test, time=400)
        p2.value = np.array([[3, 4, 5, 6],
                            [6, 5, 7, 1],
                            [6, 3, 9, 2],
                            [7, 2, 4, 5],
                            [1, 4, 9, 4]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["loc"], "value": ["undefined"], "_type": ["Location"]}
        attr_types = {"name": str, "value": str, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(PredicateException) as cm:
            pred = pr2_predicates.At("testpred", [p1, p3], ["Robot", "Location"])
        self.assertEqual(cm.exception.message, "attribute type not supported")

    def test_expr_is_gp(self):
        attrs = {"name": ["rPose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": PR2PoseVector, "_type": str}
        rPose = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["can"], "geom": (1, 2), "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)

        pred = pr2_predicates.IsGP("testgp", [rPose, can], ["RobotPose", "Can"])
        self.assertEqual(pred.get_type(), "IsGP")
        self.assertFalse(pred.test(time=400))

        rPose.value = np.array([[3, 4, 5],
                                [2, 5, 7],
                                [6, 3, 4]])
        self.assertFalse(pred.test(time=400))
        can.pose = np.array([[1, 2, 5],
                             [2, 4, 3],
                             [5, 3, 4]])
        self.assertTrue(pred.test(time=0))

    def test_expr_is_pdp(self):
        pass



    def test_expr_is_in_gripper(self):
        pass


    # TODO: test other predicates
