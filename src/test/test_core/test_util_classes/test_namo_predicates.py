import unittest
from core.internal_repr import parameter
from core.util_classes.matrix import Vector2d
from core.util_classes import common_predicates, namo_predicates
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import circle
from errors_exceptions import PredicateException, ParamValidationException
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
        can = parameter.Object(attrs, attr_types)
        attrs = {"name": ["target"], "geom": [radius], "value": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": circle.BlueCircle, "value": Vector2d, "_type": str}
        target = parameter.Symbol(attrs, attr_types)

        pred = namo_predicates.At("testpred", [can, target], ["Can", "Target"])
        self.assertEqual(pred.get_type(), "At")
        self.assertFalse(pred.test(time=400))
        can.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8]])
        # target doesn't have a value yet
        self.assertFalse(pred.test(time=400))
        target.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = namo_predicates.At("testpred", [can, sym], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")

        attrs = {"name": ["target"], "geom": [radius], "value": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": circle.BlueCircle, "value": Vector2d, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        sym.value = np.array([[3, 2], [6, 4]])

        pred = namo_predicates.At("testpred", [can, sym], ["Can", "Target"])
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))

        # testing get_expr
        pred_dict = {"negated": False, "hl_info": "pre", "active_timesteps": (0,0), "pred": pred}
        self.assertTrue(isinstance(pred.get_expr(pred_dict["negated"]), expr.EqExpr))
        pred_dict['hl_info'] = "hl_state"
        self.assertTrue(isinstance(pred.get_expr(pred_dict["negated"]), expr.EqExpr))
        pred_dict['negated'] = True
        self.assertTrue(pred.get_expr(pred_dict["negated"]) is None)
        pred_dict['hl_info'] = "pre"
        self.assertTrue(pred.get_expr(pred_dict["negated"]) is None)

    def test_robot_at(self):
        # RobotAt Robot RobotPose
        radius = 1
        attrs = {"name": ["robot"], "geom": [radius], "pose": ["undefined"], "_type": ["Robot"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": Vector2d, "_type": str}
        p1 = parameter.Object(attrs, attr_types)

        attrs = {"name": ["r_pose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": Vector2d, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)

        pred = namo_predicates.RobotAt("testpred", [p1, p2], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "RobotAt")
        self.assertFalse(pred.test(time=400))
        p1.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8]])
        p2.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (RobotAt robot r_pose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (RobotAt robot r_pose)'.")
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = namo_predicates.RobotAt("testpred", [p1, p3], ["Robot", "RobotPose"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (RobotAt robot sym)'.")

        pred = namo_predicates.RobotAt("testpred", [p1, p2], ["Robot", "RobotPose"])
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))


    def test_obstructs(self):
        #Obstructs, Robot, RobotPose, Can;
        radius = 1
        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Robot"], "name": ["robot"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can = parameter.Object(attrs, attr_types)

        env = Environment()
        pred = namo_predicates.Obstructs("obstructs", [robot, robotPose, can], ["Robot", "RobotPose", "Can"], env)
        val, jac = pred.distance_from_obj(np.array([1.9,0,0,0]))
        self.assertTrue(np.allclose(np.array(val), .20, atol=1e-2))
        jac2 = np.array([[-0.95968306, -0., 0.95968306, 0.]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        robot.pose = np.zeros((2,4))
        can.pose = np.array([[2*(radius+pred.dsafe), 0, .1, 2*radius - pred.dsafe],
                                  [0, 2*(radius+pred.dsafe), 0, 0]])
        self.assertFalse(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertTrue(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

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

    def test_in_contact(self):
        # InContact, Robot, RobotPose, Target
        radius = 1
        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Robot"], "name": ["pr2"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "value": [(0, 0)], "_type": ["Target"], "name": ["target"]}
        attr_types = {"geom": circle.BlueCircle, "value": Vector2d, "_type": str, "name": str}
        target = parameter.Symbol(attrs, attr_types)

        env = Environment()
        pred = namo_predicates.InContact("InContact", [robot, robotPose, target], ["Robot", "RobotPose", "Target"], env=env)
        #First test should fail because all objects's positions are in (0,0)
        self.assertFalse(pred.test(time = 0))
        val, jac = pred.distance_from_obj(np.array([1.9, 0, 0, 0]))
        self.assertTrue(np.allclose(np.array(val), .20, atol=1e-2))
        jac2 = np.array([[-0.95968306, -0., 0.95968306, 0.]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        robotPose.value = np.zeros((2,4))
        target.value = np.array([[2*radius+pred.dsafe, radius, 2*radius, 2*radius-pred.dsafe,  0],
                                 [0,                   0,      0,        0,                    0]])
        self.assertTrue(pred.test(time = 0))
        self.assertTrue(pred.test(time = 1))
        self.assertTrue(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))
        self.assertTrue(pred.test(time = 4))
        #since it symbol are assumed to be unchanged, test should always check distance with first traj vector
        robotPose.value = np.array([[radius, 3*radius + pred.dsafe, 0, -pred.dsafe, 2*radius+pred.dsafe],
                                    [0,      0,                     0, 0,           0]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertFalse(pred.test(time = 3))
        self.assertFalse(pred.test(time = 4))

    def test_obstructs_holding(self):
        # ObstructsHolding, Robot, RobotPose, Can, Can;
        radius = 1
        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Robot"], "name": ["pr2"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can1 = parameter.Object(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 2)], "_type": ["Can"], "name": ["can2"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can2 = parameter.Object(attrs, attr_types)

        env = Environment()
        pred = namo_predicates.ObstructsHolding("ObstructsHolding", [robot, robotPose, can1, can2], ["Robot", "RobotPose", "Can", "Can"], env)
        #First test should fail because all objects's positions are in (0,0)
        self.assertTrue(pred.test(time = 0))
        val, jac = pred.distance_from_obj(np.array([1.9,0,0,0,0,0]))
        self.assertTrue(np.allclose(np.array(val), 1.25, atol=1e-2))
        jac2 = np.array([[ 0, 0, 0.57735032,  0.57735032, -0.57735032, -0.57735032]])
        self.assertTrue(np.allclose(jac, jac2, atol=1e-2))

        robot.pose = np.zeros((2,4))
        can1.pose = np.array([[2*(radius+pred.dsafe)+0.1, 0,                    .1, 2*radius - pred.dsafe],
                              [0,                     2*(radius+pred.dsafe)+0.1, 0, 0]])
        can2.pose = np.zeros((2,4))
        self.assertFalse(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertTrue(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_in_gripper(self):
        # InGripper, Robot, Can, Grasp
        radius = 1
        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Robot"], "name": ["pr2"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can = parameter.Object(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["Grasp"], "name": ["grasp"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        grasp = parameter.Symbol(attrs, attr_types)

        pred = namo_predicates.InGripper("InGripper", [robot, can, grasp], ["Robot", "Can", "Grasp"])
        grasp.value = np.array([[1],[2]])
        #First test should fail because all objects's positions are in (0,0)
        self.assertFalse(pred.test(time = 0))

        #robot.pose - can.pose = grasp.value
        robot.pose = np.zeros((2,4))
        can.pose = np.array([[-1, 1, 3, 5],
                                [-2, 2, 4, 6]])
        self.assertTrue(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertFalse(pred.test(time = 3))

        robot.pose = np.array([ [0, 2, 4, 6],
                                [0, 4, 6, 8]])
        self.assertTrue(pred.test(time = 0))
        self.assertTrue(pred.test(time = 1))
        self.assertTrue(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

    def test_grasp_valid(self):
        # GraspValid RobotPose Target Grasp
        radius = 1

        attrs = {"value": [(0, 0)], "_type": ["RobotPose"], "name": ["r_pose"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        robotPose = parameter.Symbol(attrs, attr_types)

        attrs = {"geom": [radius], "value": [(0, 0)], "_type": ["Target"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "value": Vector2d, "_type": str, "name": str}
        target = parameter.Symbol(attrs, attr_types)

        attrs = {"value": [(0, 0)], "_type": ["Grasp"], "name": ["grasp"]}
        attr_types = {"value": Vector2d, "_type": str, "name": str}
        grasp = parameter.Symbol(attrs, attr_types)

        pred = namo_predicates.GraspValid("GraspValid", [robotPose, target, grasp], ["RobotPose", "Target", "Grasp"])
        #setting displacement to be <1, 2>
        grasp.value = np.array([[1],[2]])
        #First test should fail because all objects's positions are in (0,0)
        self.assertFalse(pred.test(time = 0))
        #robotPose.value - target.pose = grasp.value
        robotPose.value = np.zeros((2,4))
        target.value = np.array([[-1, 1, 3, 5],
                                [-2, 2, 4, 6]])
        #Since now target is a symbol, values are either all true or all false
        self.assertTrue(pred.test(time = 0))
        self.assertTrue(pred.test(time = 1))
        self.assertTrue(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

        robotPose.value = np.array([[4, 2, 1, 6],
                                    [6, 4, 0, 8]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertFalse(pred.test(time = 3))

    def test_stationary(self):
        # Stationary, Can
        attrs = {"geom": [1], "pose": [(0, 0)], "_type": ["Can"], "name": ["can"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can = parameter.Object(attrs, attr_types)

        pred = namo_predicates.Stationary("test_stay", [can], ["Can"])
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")

        can.pose = np.array([[1, 2],
                             [4, 4]])
        self.assertFalse(pred.test(time = 0))
        can.pose = np.array([[1, 1, 2],
                             [2, 2, 2]])
        self.assertTrue(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))

        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")

    def test_stationary_neq(self):
        # StationaryNEq, Can, Can
        attrs = {"geom": [1], "pose": [(0, 0)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can1 = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(0, 0)], "_type": ["Can"], "name": ["can2"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        can2 = parameter.Object(attrs, attr_types)

        pred1 = namo_predicates.StationaryNEq("test_stay_neq1", [can1, can1], ["Can", "Can"])

        with self.assertRaises(PredicateException) as cm:
            pred1.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay_neq1: (StationaryNEq can1 can1)' at the timestep.")

        can1.pose = np.array([[1, 2, 3],
                              [1, 2, 3]])
        # Since two objects in this predicate are the same one, test should always pass
        self.assertTrue(pred1.test(time = 0))
        self.assertTrue(pred1.test(time = 1))

        pred2 = namo_predicates.StationaryNEq("test_stay_neq2", [can1, can2], ["Can", "Can"])

        can2.pose = np.array([[1, 1, 1],
                              [1, 1, 1]])
        # Now that can1 is not can2, so it will check whether first can is stationary
        self.assertFalse(pred2.test(time = 0))
        self.assertFalse(pred2.test(time = 1))

        can1.pose = np.array([[2, 2, 3],
                              [2, 2, 3]])
        can2.pose = np.array([[1],
                              [1]])
        # no matter what kind of pose can2 has, it only checks whether first can is stationary
        self.assertTrue(pred2.test(time = 0))
        self.assertFalse(pred2.test(time = 1))

    def test_is_mp(self):
        # IsMP Robot
        attrs = {"geom": [1], "pose": [(0, 0)], "_type": ["Robot"], "name": ["pr2"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        robot = parameter.Object(attrs, attr_types)

        pred = namo_predicates.IsMP("IsMP", [robot], ["Robot"])

        #predicate only have 1 timestep
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'IsMP: (IsMP pr2)' at the timestep.")
        robot.pose = np.array([[1, 2,  8, 4],
                              [0, 1, -4, 9]])

        self.assertTrue([pred.test(time = 0)])
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))

        robot.pose = np.array([[1,2,3,4],
                               [5,6,7,8]])
        self.assertTrue(pred.test(time = 0))
        self.assertTrue(pred.test(time = 1))
        self.assertTrue(pred.test(time = 2))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=3)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'IsMP: (IsMP pr2)' at the timestep.")

if __name__ == "__main__":
    unittest.main()