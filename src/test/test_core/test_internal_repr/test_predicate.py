import unittest
from core.internal_repr import predicate
from core.internal_repr import parameter
from errors_exceptions import ParamValidationException
from core.util_classes.circle import GreenCircle, RedCircle, BlueCircle
import numpy as np

class TestPredicate(unittest.TestCase):
    def test_param_validation(self):

        attrs = {"name": ["can"], "geom": [1], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": RedCircle, "pose": int, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["target"], "geom": [1], "pose": ["undefined"], "_type": ["Target"]}
        attr_types = {"name": str, "geom": BlueCircle, "pose": int, "_type": str}
        p2 = parameter.Object(attrs, attr_types)
        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": int, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)
        # this one should work
        pred = predicate.Predicate("errorpred", [p1, p2, p3], ["Can", "Target", "Sym"])
        self.assertEqual(pred.get_type(), "Predicate")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [p1, p2, p3], ["Target", "Can", "Sym"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [p1, p2, p3], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")
        with self.assertRaises(ParamValidationException) as cm:
            predicate.Predicate("errorpred", [p1, p2, p3], ["Can", "Target", "Target", "Sym"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (Predicate can target sym)'.")
