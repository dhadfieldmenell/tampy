import unittest
from core.internal_repr import problem
from core.internal_repr import parameter
from core.internal_repr import predicate
from core.internal_repr import state
import numpy as np

class TestProblem(unittest.TestCase):
    def setUp(self):
        self.can = parameter.Can("can")
        self.target = parameter.Target("target")
        self.gp = parameter.Symbol("gp")
        self.at = predicate.At("at", [self.can, self.target], ["Can", "Target"])
        self.isgp = predicate.IsGP("isgp", [self.gp, self.can], ["Symbol", "Can"])
        self.init_state = state.State("state", [self.at, self.isgp], timestep=0)

    def test_init_state(self):
        with self.assertRaises(Exception) as cm:
            problem.Problem(self.init_state, None, None, None)
        self.assertEqual(cm.exception.message, "Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?")
        self.can.pose = np.array([[3, 0], [4, 2]])
        self.target.pose = np.array([[3, 1], [3, 2]])
        with self.assertRaises(Exception) as cm:
            problem.Problem(self.init_state, None, None, None)
        self.assertEqual(cm.exception.message, "Initial state is not consistent (predicates are violated).")

    def test_goal_test(self):
        self.can.pose = np.array([[3, 0, 5], [4, 2, 1]])
        self.target.pose = np.array([[3, 0, 4], [4, 2, 0]])
        p = problem.Problem(self.init_state, [self.at], None, time_horizon=1)
        self.assertTrue(p.goal_test())
        p = problem.Problem(self.init_state, [self.at], None, time_horizon=2)
        self.assertFalse(p.goal_test())
