import unittest
from core.internal_repr import problem
from core.internal_repr import parameter
from core.util_classes import common_predicates
from core.internal_repr import state
from errors_exceptions import ProblemConfigException
import numpy as np

class TestProblem(unittest.TestCase):
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
        self.init_state = state.State("state", [self.can, self.target, self.gp], [self.at, self.isgp], timestep=0)

    def test_init_state(self):
        with self.assertRaises(ProblemConfigException) as cm:
            problem.Problem(self.init_state, None)
        self.assertEqual(cm.exception.message, "Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?")
        self.can.pose = np.array([[3, 0], [4, 2]])
        self.target.pose = np.array([[3, 1], [3, 2]])
        with self.assertRaises(ProblemConfigException) as cm:
            problem.Problem(self.init_state, None)
        self.assertEqual(cm.exception.message, "Initial state is not consistent (predicates are violated).")
        self.target.pose = np.array([[3, 1], [4, 2]])
        problem.Problem(self.init_state, None)

    def test_goal_test(self):
        self.can.pose = np.array([[3, 0, 5], [4, 1, 1]])
        self.target.pose = np.array([[3, 0, 4], [4, 2, 0]])
        p = problem.Problem(self.init_state, [self.at])
        # problems only consider timestep 0 in their goal test,
        # so this will be True even though the poses become different at timestep 1
        self.assertTrue(p.goal_test())
