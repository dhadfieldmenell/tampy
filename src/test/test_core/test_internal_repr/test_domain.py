import unittest
from core.internal_repr import domain

class TestDomain(unittest.TestCase):
    def test(self):
        d = domain.Domain(1, 2, 3)
        self.assertEqual(d.param_schema, 1)
        self.assertEqual(d.pred_schema, 2)
        self.assertEqual(d.action_schema, 3)
