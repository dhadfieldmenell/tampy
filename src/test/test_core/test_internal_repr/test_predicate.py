import unittest
from core.internal_repr import predicate
from core.internal_repr import parameter
from errors_exceptions import ParamValidationException
from core.util_classes.circle import GreenCircle, RedCircle, BlueCircle
from core.util_classes.matrix import Value
import numpy as np

class TestPredicate(unittest.TestCase):
    def setUp(self):
        attrs = {"name": ["can"], "geom": [1], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": RedCircle, "pose": Value, "_type": str}
        self.p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["target"], "geom": [1], "pose": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": BlueCircle, "pose": Value, "_type": str}
        self.p2 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": Value, "_type": str}
        self.p3 = parameter.Symbol(attrs, attr_types)
        # this one should work
        self.pred = predicate.Predicate("errorpred", [self.p1, self.p2, self.p3], ["Can", "Target", "Sym"])

    def test_param_validation(self):
        self.assertEqual(self.pred.get_type(), "Predicate")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [self.p1, self.p2, self.p3], ["Target", "Can", "Sym"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [self.p1, self.p2, self.p3], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [self.p1, self.p2, self.p3], ["Can", "Target", "Target", "Sym"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")

    def test_copy(self):
        param_copies = {}
        params = [self.p1, self.p2, self.p3]
        for p in params:
            param_copies[p] = p.copy_ts((1,3))
        pred_copy = self.pred.copy(param_copies)

        for p in params:
            self.assertTrue(p not in pred_copy.params)

        for p in param_copies.values():
            self.assertTrue(p in pred_copy.params)

        self.assertTrue(self.pred.name == pred_copy.name)
        self.assertTrue(self.pred.active_range == pred_copy.active_range)
        self.assertTrue(self.pred.priority == pred_copy.priority)
