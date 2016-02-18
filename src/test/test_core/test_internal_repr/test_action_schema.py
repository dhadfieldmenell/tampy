import unittest
from core.internal_repr import action_schema

class TestActionSchema(unittest.TestCase):
    def test(self):
        s = action_schema.ActionSchema(1, 2, 3, 4, 5)
        self.assertEqual(s.name, 1)
        self.assertEqual(s.horizon, 2)
        self.assertEqual(s.params, 3)
        self.assertEqual(s.universally_quantified_params, 4)
        self.assertEqual(s.preds, 5)
