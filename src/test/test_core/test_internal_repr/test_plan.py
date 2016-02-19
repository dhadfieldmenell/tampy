import unittest
from core.internal_repr import plan

class TestPlan(unittest.TestCase):
    def test_args(self):
        s = plan.Plan(1, 2, 3)
        self.assertEqual(s.params, 1)
        self.assertEqual(s.actions, 2)
        self.assertEqual(s.horizon, 3)

    # TODO: add more as functionality develops
