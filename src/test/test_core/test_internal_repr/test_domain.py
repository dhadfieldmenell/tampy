import unittest
from core.internal_repr import domain
from core.internal_repr.parameter_schema import ParameterSchema
from core.internal_repr.predicate_schema import PredicateSchema
from core.internal_repr.action_schema import ActionSchema

class TestDomain(unittest.TestCase):
    def test(self):
        a = {"a": ParameterSchema(1, 2, 3)}
        b = {"b": PredicateSchema(1, 2, 3)}
        c = {"c": ActionSchema(1, 2, 3, 4, 5)}
        d = domain.Domain(a, b, c)
        self.assertEqual(d.param_schemas, a)
        self.assertEqual(d.pred_schemas, b)
        self.assertEqual(d.action_schemas, c)

    def test_failure(self):
        a = {"a": ParameterSchema(1, 2, 3)}
        b = {"b": PredicateSchema(1, 2, 3)}
        c = {"c": ActionSchema(1, 2, 3, 4, 5)}
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(1, 2, 3)
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(1, b, c)
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(a, 2, c)
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(a, b, 3)
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(b, a, c)
        with self.assertRaises(AssertionError) as cm:
            d = domain.Domain(c, b, a)
