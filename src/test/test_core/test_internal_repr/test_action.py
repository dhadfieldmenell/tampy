import unittest
from core.internal_repr import action

class TestAction(unittest.TestCase):
    def test(self):
        s = action.Action(1, 2, 3, 4, 5)
        self.assertEqual(s.step_num, 1)
        self.assertEqual(s.name, 2)
        self.assertEqual(s.active_timesteps, 3)
        self.assertEqual(s.params, 4)
        self.assertEqual(s.preds, 5)
