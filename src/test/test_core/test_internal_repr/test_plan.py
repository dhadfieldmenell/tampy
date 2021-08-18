import unittest
from core.internal_repr import plan
from core.internal_repr import parameter
from core.internal_repr import action
from core.util_classes import namo_predicates_gurobi
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.util_classes import namo_predicates_gurobi
from errors_exceptions import PredicateException
import numpy as np


class TestPlan(unittest.TestCase):
    def setup(self):
        attrs = {"name": ["robot"], "geom": [1], "pose": [(0, 0)], "_type": ["Robot"]}
        attr_types = {
            "name": str,
            "geom": circle.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        self.robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["can1"], "geom": [1], "pose": [(3, 4)], "_type": ["Can"]}
        attr_types = {
            "name": str,
            "geom": circle.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        self.can1 = parameter.Object(attrs, attr_types)

        attrs = {"name": ["can2"], "geom": [1], "pose": [(3, 4)], "_type": ["Can"]}
        attr_types = {
            "name": str,
            "geom": circle.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        self.can2 = parameter.Object(attrs, attr_types)

        attrs = {
            "name": ["target"],
            "geom": [1],
            "value": [(3, 4)],
            "_type": ["Target"],
        }
        attr_types = {
            "name": str,
            "geom": circle.BlueCircle,
            "value": Vector2d,
            "_type": str,
        }
        self.target = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["rpose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": Vector2d, "_type": str}
        self.rpose = parameter.Symbol(attrs, attr_types)

        self.pred0 = namo_predicates_gurobi.At(
            "At_0", [self.can1, self.target], ["Can", "Target"]
        )
        self.pred1 = namo_predicates_gurobi.At(
            "At_1", [self.can2, self.target], ["Can", "Target"]
        )
        self.pred2 = namo_predicates_gurobi.RobotAt(
            "RobotAt_0", [self.robot, self.rpose], ["Robot", "RobotPose"]
        )

    def test_args(self):
        attrs = {"name": ["robot"], "geom": [1], "pose": [(0, 0)], "_type": ["Robot"]}
        attr_types = {
            "name": str,
            "geom": circle.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        robot = parameter.Object(attrs, attr_types)
        attrs = {"name": ["can"], "geom": [1], "pose": [(3, 4)], "_type": ["Can"]}
        attr_types = {
            "name": str,
            "geom": circle.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        can = parameter.Object(attrs, attr_types)
        param_map = {"robot": robot, "can": can}
        act_list = ["up", "down"]
        s = plan.Plan(param_map, act_list, 3, 4)
        self.assertEqual(s.params, param_map)
        self.assertEqual(s.actions, act_list)
        self.assertEqual(s.horizon, 3)
        self.assertEqual(s.env, 4)

    def test_get_failed_preds(self):
        # this function tests get_failed_pred() and satisfied()
        self.setup()
        params = [self.can1, self.target]
        pred_dict = [
            {
                "pred": self.pred0,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 1),
            }
        ]
        act0 = action.Action(2, "test_action0", (0, 10), params, pred_dict)
        params = [self.can2, self.target]
        pred_dict = [
            {
                "pred": self.pred1,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (1, 3),
            }
        ]
        act1 = action.Action(2, "test_action1", (0, 10), params, pred_dict)
        params = [self.robot, self.rpose]
        pred_dict = [
            {
                "pred": self.pred2,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (2, 3),
            }
        ]
        act2 = action.Action(2, "test_action2", (0, 10), params, pred_dict)
        plan_params = {
            "robot": self.robot,
            "can1": self.can1,
            "can2": self.can2,
            "target": self.target,
            "rpose": self.rpose,
        }
        plan_actions = [act0, act1, act2]
        test_plan = plan.Plan(plan_params, plan_actions, 10, 1)  # 1 is a dummy_env

        self.robot.pose = np.array([[3, 4, 7, 7], [2, 5, 8, 9]])
        self.can1.pose = np.array([[5, 5, 7, 9], [5, 5, 6, 8]])
        self.can2.pose = np.array([[9, 5, 5, 5], [4, 5, 5, 4]])
        self.rpose.value = np.array([[7], [9]])
        self.target.value = np.array([[5], [5]])
        self.assertFalse(test_plan.satisfied())
        self.assertEqual(test_plan.get_failed_pred(), (False, self.pred2, 2))
        self.assertEqual(
            test_plan.get_failed_preds(),
            [(False, self.pred1, 3), (False, self.pred2, 2)],
        )
        self.robot.pose = np.array([[3, 4, 7, 7], [2, 5, 9, 9]])
        self.assertEqual(test_plan.get_failed_pred(), (False, self.pred1, 3))
        self.assertEqual(test_plan.get_failed_preds(), [(False, self.pred1, 3)])
        self.assertFalse(test_plan.satisfied())
        self.can2.pose = np.array([[9, 5, 5, 5], [4, 5, 5, 5]])
        self.assertEqual(test_plan.get_failed_pred(), (False, None, 11))
        self.assertEqual(test_plan.get_failed_preds(), [])
        self.assertTrue(test_plan.satisfied())

    def test_get_active_preds(self):
        self.setup()
        params = [self.can1, self.target]
        pred_dict = [
            {
                "pred": self.pred0,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 5),
            }
        ]
        act0 = action.Action(2, "test_action0", (0, 5), params, pred_dict)
        params = [self.can2, self.target]
        pred_dict = [
            {
                "pred": self.pred1,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (3, 7),
            }
        ]
        act1 = action.Action(2, "test_action1", (3, 7), params, pred_dict)
        params = [self.robot, self.rpose]
        pred_dict = [
            {
                "pred": self.pred2,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (4, 9),
            }
        ]
        act2 = action.Action(2, "test_action2", (4, 9), params, pred_dict)
        plan_params = {
            "robot": self.robot,
            "can1": self.can1,
            "can2": self.can2,
            "target": self.target,
            "rpose": self.rpose,
        }
        plan_actions = [act0, act1, act2]
        test_plan = plan.Plan(plan_params, plan_actions, 10, 1)  # 1 is a dummy_env

        self.assertEqual(test_plan.get_active_preds(2), [self.pred0])
        self.assertEqual(
            test_plan.get_active_preds(4), [self.pred0, self.pred1, self.pred2]
        )
        self.assertEqual(test_plan.get_active_preds(6), [self.pred1, self.pred2])
        self.assertEqual(test_plan.get_active_preds(8), [self.pred2])
        self.assertEqual(test_plan.get_active_preds(10), [])


if __name__ == "__main__":
    unittest.main()
