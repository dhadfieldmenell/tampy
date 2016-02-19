import unittest
from core.internal_repr import state
from core.internal_repr import parameter
from core.util_classes import common_predicates
import numpy as np

class TestState(unittest.TestCase):
    def setUp(self):
        attrs = {"name": "can", "pose": "undefined", "_type": "Can"}
        attr_types = {"name": str, "pose": int, "_type": str}
        self.can = parameter.Object(attrs, attr_types)
        attrs = {"name": "target", "pose": "undefined", "_type": "Target"}
        attr_types = {"name": str, "pose": int, "_type": str}
        self.target = parameter.Object(attrs, attr_types)
        attrs = {"name": "gp", "value": "undefined", "_type": "Sym"}
        attr_types = {"name": str, "value": int, "_type": str}
        self.gp = parameter.Symbol(attrs, attr_types)
        self.at = common_predicates.At("at", [self.can, self.target], ["Can", "Target"])
        self.isgp = common_predicates.IsGP("isgp", [self.gp, self.can], ["Sym", "Can"])
        self.s = state.State("state", [self.can, self.target, self.gp], [self.at, self.isgp], timestep=0)

    def test(self):
        self.assertEqual(self.s.name, "state")
        self.assertEqual(self.s.params, [self.can, self.target, self.gp])
        self.assertEqual(self.s.preds, set([self.at, self.isgp]))
        other_state = state.State("state", [self.can, self.target, self.gp])
        self.assertEqual(other_state.params, [self.can, self.target, self.gp])
        self.assertEqual(other_state.preds, set())
        self.assertEqual(other_state.timestep, 0)

    def test_concrete(self):
        self.assertFalse(self.s.is_concrete())
        self.can.pose = 3
        self.target.pose = 4
        self.assertTrue(self.s.is_concrete())

    def test_consistent(self):
        self.can.pose = np.array([[3, 0], [4, 2]])
        self.target.pose = np.array([[3, 1], [3, 2]])
        self.assertFalse(self.s.is_consistent())
        self.target.pose = np.array([[3, 1], [4, 2]])
        self.assertTrue(self.s.is_consistent())
