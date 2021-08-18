import unittest
from core.internal_repr import state
from core.internal_repr import parameter
from core.util_classes import common_predicates, namo_predicates
from core.util_classes.matrix import Vector2d
from core.util_classes import items
from openravepy import Environment
import numpy as np


class TestState(unittest.TestCase):
    def setUp(self):
        attrs = {"name": ["robot"], "geom": [1], "pose": [(0, 0)], "_type": ["Robot"]}
        attr_types = {
            "name": str,
            "geom": items.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        self.robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["can"], "geom": [1], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {
            "name": str,
            "geom": items.RedCircle,
            "pose": Vector2d,
            "_type": str,
        }
        self.can = parameter.Object(attrs, attr_types)

        attrs = {
            "name": ["target"],
            "geom": [1],
            "value": ["undefined"],
            "_type": ["Target"],
        }
        attr_types = {
            "name": str,
            "geom": items.BlueCircle,
            "value": Vector2d,
            "_type": str,
        }
        self.target = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["gp"], "value": [(3, 6.0)], "_type": ["Sym"]}
        attr_types = {"name": str, "value": Vector2d, "_type": str}
        self.gp = parameter.Symbol(attrs, attr_types)

        self.at = namo_predicates_gurobi.At(
            "at", [self.can, self.target], ["Can", "Target"]
        )
        env = Environment()
        self.in_contact = namo_predicates_gurobi.InContact(
            "incontact",
            [self.robot, self.gp, self.target],
            ["Robot", "Sym", "Target"],
            env=env,
        )
        self.s = state.State(
            "state",
            {
                self.can.name: self.can,
                self.target.name: self.target,
                self.gp.name: self.gp,
            },
            [self.at, self.in_contact],
            timestep=0,
        )

        self.assertEqual(self.s.preds, set([self.at, self.in_contact]))
        other_state = state.State("state", [self.can, self.target, self.gp])
        self.assertEqual(other_state.params, [self.can, self.target, self.gp])
        self.assertEqual(other_state.preds, set())
        self.assertEqual(other_state.timestep, 0)

    def test_concrete(self):
        self.assertFalse(self.s.is_concrete())
        self.can.pose = np.array([3, 4])
        self.target.value = np.array([4, 5])
        self.assertTrue(self.s.is_concrete())

    def test_consistent(self):
        self.can.pose = np.array([[3, 0], [4, 2]])
        self.target.value = np.array([[3, 1], [3, 2]])
        self.assertFalse(self.s.is_consistent())
        self.target.value = np.array([[3, 0], [4, 2]])
        self.assertTrue(self.s.is_consistent())


if __name__ == "__main__":
    unittest.main()
