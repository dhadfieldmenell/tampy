import unittest
from core.internal_repr import action
from core.util_classes import namo_predicates_gurobi
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.internal_repr import parameter
import numpy as np


class TestAction(unittest.TestCase):
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

    def test(self):
        s = action.Action(1, 2, 3, 4, 5)
        self.assertEqual(s.step_num, 1)
        self.assertEqual(s.name, 2)
        self.assertEqual(s.active_timesteps, 3)
        self.assertEqual(s.params, 4)
        self.assertEqual(s.preds, 5)

    def test_get_failed_preds(self):
        # This test tests get_failed_preds() and satisfied() functions
        self.setup()

        params = [self.robot, self.can1, self.can2, self.target, self.rpose]
        pred_dict = [
            {
                "pred": self.pred0,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 0),
            },
            {
                "pred": self.pred1,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 0),
            },
            {
                "pred": self.pred2,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 0),
            },
        ]
        act = action.Action(0, "test_action", (0, 0), params, pred_dict)
        # Since value of robot pose is not defined, RobotAt should fail
        self.assertFalse(act.satisfied())
        self.assertEqual(act.get_failed_preds(), [(False, self.pred2, 0)])
        # Now rpose.value is defined, test should all pass
        self.rpose.value = np.array([[0], [0]])
        self.assertTrue(act.satisfied())
        self.assertEqual(act.get_failed_preds(), [])

        self.robot.pose = np.array([[2, 2, 7], [5, 5, 9]])
        self.can1.pose = np.array([[3, 3, 9], [4, 4, 7]])
        self.can2.pose = np.array([[9, 3, 6], [4, 6, 7]])
        self.rpose.value = np.array([[2], [5]])

        pred_dict[0]["active_timesteps"] = (0, 1)
        pred_dict[1]["active_timesteps"] = (0, 1)
        pred_dict[1]["hl_info"] = "hl_state"
        pred_dict[2]["active_timesteps"] = (0, 1)
        act2 = action.Action(3, "test_action2", (0, 2), params, pred_dict)
        # since each preds is only defined between timestep 0 and timestep 1, it doesn't chect the timestep at 2
        # since hl_info for pred1 is hl_state, test doesn't check it
        self.assertTrue(act2.satisfied())
        self.assertEqual(act2.get_failed_preds(), [])

    def test_first_failed_ts(self):
        self.setup()

        params = [self.robot, self.can1, self.can2, self.target, self.rpose]
        # Precticates happened in sequences, pred0->pred1->pred2, pred1 and pred2, has a timestep of overlap
        pred_dict = [
            {
                "pred": self.pred0,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 1),
            },
            {
                "pred": self.pred1,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (1, 3),
            },
            {
                "pred": self.pred2,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (2, 3),
            },
        ]

        self.robot.pose = np.array([[2, 4, 7, 7], [5, 7, 8, 9]])
        self.can1.pose = np.array([[5, 5, 7, 9], [5, 5, 6, 8]])
        self.can2.pose = np.array([[9, 5, 5, 5], [4, 5, 5, 4]])
        self.rpose.value = np.array([[7], [9]])
        self.target.value = np.array([[5], [5]])
        # so that pred2 failed at timestep 2, pred1 failed at timestep3
        act = action.Action(4, "test_action", (0, 10), params, pred_dict)
        self.assertFalse(act.satisfied())
        self.assertEqual(act.first_failed_ts(), 2)
        self.robot.pose = np.array([[2, 4, 7, 7], [5, 7, 9, 9]])
        self.assertEqual(act.first_failed_ts(), 3)
        self.can2.pose = np.array([[9, 5, 5, 5], [4, 5, 5, 5]])
        self.assertEqual(act.first_failed_ts(), 10)

    def test_get_active_pred(self):
        self.setup()
        params = [self.robot, self.can1, self.can2, self.target, self.rpose]
        # Precticates happened in sequences, pred0->pred1->pred2, pred1 and pred2, has a timestep of overlap
        pred_dict = [
            {
                "pred": self.pred0,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (0, 5),
            },
            {
                "pred": self.pred1,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (4, 7),
            },
            {
                "pred": self.pred2,
                "negated": False,
                "hl_info": "pre",
                "active_timesteps": (3, 9),
            },
        ]
        act = action.Action(4, "test_action", (0, 10), params, pred_dict)
        self.assertEqual(act.get_active_preds(2), [self.pred0])
        self.assertEqual(act.get_active_preds(4), [self.pred0, self.pred1, self.pred2])
        self.assertEqual(act.get_active_preds(6), [self.pred1, self.pred2])
        self.assertEqual(act.get_active_preds(8), [self.pred2])
        self.assertEqual(act.get_active_preds(10), [])


if __name__ == "__main__":
    unittest.main()
