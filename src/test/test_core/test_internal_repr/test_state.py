import unittest
from core.internal_repr import state
from core.internal_repr import predicate
from core.internal_repr import parameter
import numpy as np

class TestState(unittest.TestCase):
    def setUp(self):
        self.can = parameter.Can("can")
        self.target = parameter.Target("target")
        self.gp = parameter.Symbol("gp")
        self.at = predicate.At("at", [self.can, self.target], ["Can", "Target"])
        self.isgp = predicate.IsGP("isgp", [self.gp, self.can], ["Symbol", "Can"])
        self.s = state.State("state", [self.can, self.target, self.gp], [self.at, self.isgp], timestep=0)
        self.assertEqual(self.s.name, "state")
        self.assertEqual(self.s.params, set([self.can, self.target, self.gp]))
        other_state = state.State("state", [self.can, self.target, self.gp], preds=None, timestep=0)
        self.assertEqual(other_state.params, set([self.can, self.target, self.gp]))
        self.assertEqual(other_state.preds, [])

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
