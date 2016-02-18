import unittest
from core.internal_repr import predicate_schema

class TestPredicateSchema(unittest.TestCase):
    def test(self):
        s = predicate_schema.PredicateSchema(1, 2, 3)
        self.assertEqual(s.pred_type, 1)
        self.assertEqual(s.pred_class, 2)
        self.assertEqual(s.expected_params, 3)
