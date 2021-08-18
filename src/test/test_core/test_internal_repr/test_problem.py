import unittest
from core.internal_repr import problem
from core.internal_repr import parameter
from core.util_classes import common_predicates, namo_predicates
from core.util_classes.matrix import Vector2d
from core.internal_repr import state
from errors_exceptions import ProblemConfigException
from core.util_classes import items
from openravepy import Environment
import numpy as np


class TestProblem(unittest.TestCase):
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
        self.init_state = state.State(
            "state",
            {
                self.can.name: self.can,
                self.target.name: self.target,
                self.gp.name: self.gp,
            },
            [self.at, self.in_contact],
            timestep=0,
        )

    def test_init_state(self):
        with self.assertRaises(ProblemConfigException) as cm:
            problem.Problem(self.init_state, None, None)
        self.assertEqual(
            cm.exception.message,
            "Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?",
        )
        self.can.pose = np.array([[3, 0], [4, 2]])
        self.target.value = np.array([[3, 1], [3, 2]])
        with self.assertRaises(ProblemConfigException) as cm:
            problem.Problem(self.init_state, None, None)
        self.assertEqual(
            cm.exception.message,
            "Initial state is not consistent (predicates are violated).",
        )
        self.target.value = np.array([[3, 0], [4, 2]])
        problem.Problem(self.init_state, None, None)

    def test_goal_test(self):
        self.can.pose = np.array([[3, 0, 5], [4, 1, 1]])
        self.target.value = np.array([[3, 0, 4], [4, 2, 0]])
        p = problem.Problem(self.init_state, [self.at], None)
        # problems only consider timestep 0 in their goal test,
        # so this will be True even though the poses become different at timestep 1
        self.assertTrue(p.goal_test())


if __name__ == "__main__":
    unittest.main()
