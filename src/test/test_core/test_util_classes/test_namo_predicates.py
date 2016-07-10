import unittest
from core.internal_repr import parameter
from core.util_classes.matrix import Vector2d
from core.util_classes import common_predicates, namo_predicates
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import circle
from errors_exceptions import PredicateException
from sco import expr
import numpy as np
from openravepy import Environment

N = 10

## exprs for testing
e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestNamoPredicates(unittest.TestCase):

    def test_expr_at(self):
        radius = 1
        attrs = {"name": ["can"], "geom": [radius], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": Vector2d, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["target"], "geom": [radius], "pose": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": circle.BlueCircle, "pose": Vector2d, "_type": str}
        p2 = parameter.Object(attrs, attr_types)

        pred = namo_predicates.At("testpred", [p1, p2], ["Can", "Target"])
        self.assertEqual(pred.get_type(), "At")
        self.assertFalse(pred.test(time=400))
        p1.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8]])
        # p2 doesn't have a value yet
        self.assertFalse(pred.test(time=400))
        p2.pose = np.array([[3, 4, 5, 7], [6, 5, 8, 7]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(PredicateException) as cm:
            pred = namo_predicates.At("testpred", [p1, p3], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "attribute type not supported")

        attrs = {"name": ["target"], "geom": [radius], "pose": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": circle.BlueCircle, "pose": Vector2d, "_type": str}
        p3 = parameter.Object(attrs, attr_types)
        p3.pose = np.array([[3, 2], [6, 4]])

        pred = namo_predicates.At("testpred", [p1, p3], ["Can", "Target"])
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))

        # testing get_expr
        pred_dict = {"negated": False, "hl_info": "pre", "active_timesteps": (0,0), "pred": pred}
        self.assertTrue(isinstance(pred.get_expr(pred_dict, None), expr.EqExpr))
        pred_dict['hl_info'] = "hl_state"
        self.assertTrue(isinstance(pred.get_expr(pred_dict, None), expr.EqExpr))
        pred_dict['negated'] = True
        self.assertTrue(pred.get_expr(pred_dict, None) is None)
        pred_dict['hl_info'] = "pre"
        self.assertTrue(pred.get_expr(pred_dict, None) is None)

    def test_not_obstructs(self):
        radius = 1
        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        pred = namo_predicates.NotObstructs("obstructs", [green_can, blue_can], ["Can", "Can"])
        val, jac = pred.distance_from_obj(np.array([1.9,0,0,0]))
        self.assertTrue(np.allclose(np.array(val), .15, atol=1e-2))
        jac2 = np.array([[-0.95968306, -0., 0.95968306, 0.]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        green_can.pose = np.zeros((2,4))
        blue_can.pose = np.array([[2*(radius+pred.dsafe), 0, .1, 2*radius - pred.dsafe],
                                  [0, 2*(radius+pred.dsafe), 0, 0]])
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        """
        test below for checking gradient doesn't work well because the normal
        returned by the collision can be off by quite a bit
        """
        # le_expr = pred.expr
        # col_expr = le_expr.expr
        # for i in range(N):
        #     x = np.random.rand(4)
        #     print x[0:2] - x[2:4]
        #     print "x: ", x
        #     col_expr.grad(x, num_check=True, atol=1e-1)

    def test_is_gp(self):
        radius = 1
        attrs = {"geom": [radius], "pose": [(2, 0)], "_type": ["Robot"], "name": ["robot"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        pred = namo_predicates.IsGP("IsGP", [robot, robotPose, blue_can], ["Robot", "RobotPose", "Can"])
        self.assertTrue(np.allclose(pred.grasp(0), 0, atol = 1e-2))
        val, jac = pred.distance_from_obj(np.array([1.9, 0, 0, 0]))
        self.assertTrue(np.allclose(np.array(val), .15, atol=1e-2))
        jac2 = np.array([[-0.95968306, -0., 0.95968306, 0.]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        robotPose.value = np.zeros((2, 4))
        blue_can.pose = np.array([[2 * (radius + pred.dsafe), 0, .1, 2 * radius - pred.dsafe],
                              [0, 2 * radius + pred.dsafe, 0, 0]])
        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_is_pdp(self):
        radius = 1
        attrs = {"geom": [radius], "pose": [(2, 0)], "_type": ["Robot"], "name": ["robot"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Target"], "name": ["target1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        pred = namo_predicates.IsPDP("IsPdP", [robot, robotPose, blue_can], ["Robot", "RobotPose", "Target"])
        self.assertTrue(np.allclose(pred.grasp(0), 0, atol = 1e-2))
        val, jac = pred.distance_from_obj(np.array([1.9, 0, 0, 0]))
        self.assertTrue(np.allclose(np.array(val), .15, atol=1e-2))
        jac2 = np.array([[-0.95968306, -0., 0.95968306, 0.]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        robotPose.value = np.zeros((2, 4))
        blue_can.pose = np.array([[2 * (radius + pred.dsafe), 0, .1, 2 * radius - pred.dsafe],
                                  [0, 2 * radius + pred.dsafe, 0, 0]])
        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))
    def test_in_gripper(self):
        radius = 1
        attrs = {"geom": [radius], "pose": [(2.05, 0)], "_type": ["Robot"], "name": ["robot"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Target"], "name": ["target1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        target = parameter.Object(attrs, attr_types)
        pred = namo_predicates.InGripper("InGripper", [robot, target], ["Robot", "Target"])
        self.assertTrue(pred.test(time = 0))
        #TODO more test
