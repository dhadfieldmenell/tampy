import unittest
from core.internal_repr import plan
from core.internal_repr import parameter
from core.internal_repr import action
from core.util_classes import namo_predicates
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.util_classes import plan_hdf5_serialization
from errors_exceptions import PredicateException
import numpy as np


class TestPlanHDF5Serialization(unittest.TestCase):

    def setup(self):
        attrs = {"name": ["robot"], "geom": [1], "pose": [(0,0)], "_type": ["Robot"]}
        attr_types = {"name": str, "geom": circle.RedCircle,"pose": Vector2d, "_type": str}
        self.robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["can1"], "geom": [1], "pose": [(3,4)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": Vector2d, "_type": str}
        self.can1 = parameter.Object(attrs, attr_types)

        attrs = {"name": ["can2"], "geom": [1], "pose": [(3,4)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": Vector2d, "_type": str}
        self.can2 = parameter.Object(attrs, attr_types)

        attrs = {"name": ["target"], "geom": [1], "value": [(3,4)], "_type": ["Target"]}
        attr_types = {"name": str, "geom": circle.BlueCircle, "value": Vector2d, "_type": str}
        self.target = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["rpose"], "value": ["undefined"], "_type": ["RobotPose"]}
        attr_types = {"name": str, "value": Vector2d, "_type": str}
        self.rpose = parameter.Symbol(attrs, attr_types)

        self.pred0 = namo_predicates.At("At_0", [self.can1, self.target], ["Can", "Target"])
        self.pred1 = namo_predicates.At("At_1", [self.can2, self.target], ["Can", "Target"])
        self.pred2 = namo_predicates.RobotAt("RobotAt_0", [self.robot, self.rpose], ["Robot", "RobotPose"])

        params = [self.can1, self.target]
        pred_dict = [{'pred': self.pred0, 'negated': False, 'hl_info': 'pre', 'active_timesteps': (0, 5)}]
        act0 = action.Action(2, 'test_action0', (0,5), params, pred_dict)
        params = [self.can2, self.target]
        pred_dict = [{'pred': self.pred1, 'negated': False, 'hl_info': 'pre', 'active_timesteps': (3, 7)}]
        act1 = action.Action(2, 'test_action1', (3,7), params, pred_dict)
        params = [self.robot, self.rpose]
        pred_dict = [{'pred': self.pred2, 'negated': False, 'hl_info': 'pre', 'active_timesteps': (4, 9)}]
        act2 = action.Action(2, 'test_action2', (4,9), params, pred_dict)
        plan_params = {"robot": self.robot, "can1": self.can1, "can2": self.can2, "target": self.target, "rpose": self.rpose}
        plan_actions = [act0, act1, act2]
        self.plan = plan.Plan(plan_params, plan_actions, 10, 1) #1 is a dummy_env

        serializer = plan_hdf5_serialization.PlanSerializer()
        serializer.write_plan_to_hdf5("test/test_plan.hdf5", self.plan)


    def test_serialization(self):
        self.setup()
        deserializer = plan_hdf5_serialization.PlanDeserializer()
        test_plan = deserializer.read_from_hdf5("test/test_plan.hdf5")

        for param in test_plan.params:
            self.assertEqual(param, test_plan.params[param].name)
            self.assertTrue(param in self.plan.params)
            self.assertEqual(param, self.plan.params[param].name)
            self.assertEqual(test_plan.params[param].get_type(), self.plan.params[param].get_type())

        action_names = [a.name for a in self.plan.actions]
        action_timesteps = [a.active_timesteps for a in self.plan.actions]
        for i in range(len(self.plan.actions)):
            self.assertEqual(test_plan.actions[i].name, self.plan.actions[i].name)
            self.assertEqual(test_plan.actions[i].active_timesteps, self.plan.actions[i].active_timesteps)
            self.assertEqual(test_plan.actions[i].step_num, self.plan.actions[i].step_num)

        self.assertEqual(self.plan.horizon, test_plan.horizon)

        for t in range(test_plan.horizon):
            test_plan_active_preds = test_plan.get_active_preds(t)
            test_plan_pred_names = [p.name for p in test_plan_active_preds]
            active_preds =  self.plan.get_active_preds(t)
            for i in range(len(active_preds)):
                self.assertEqual(active_preds[i].name, test_plan_active_preds[i].name)
                self.assertEqual(active_preds[i].get_type(), test_plan_active_preds[i].get_type())


if __name__ == "__main__":
    unittest.main()
