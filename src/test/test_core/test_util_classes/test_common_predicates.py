import unittest
from core.internal_repr import parameter
from core.util_classes import common_predicates
import numpy as np

class TestCommonPredicates(unittest.TestCase):
    def test_at(self):
        attrs = {"name": "can", "pose": "undefined", "_type": "Can"}
        attr_types = {"name": str, "pose": int, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        attrs = {"name": "target", "pose": "undefined", "_type": "Target"}
        attr_types = {"name": str, "pose": int, "_type": str}
        p2 = parameter.Object(attrs, attr_types)
        attrs = {"name": "sym", "value": "undefined", "_type": "Sym"}
        attr_types = {"name": str, "value": int, "_type": str}
        p3 = parameter.Symbol(attrs, attr_types)

        pred = common_predicates.At("testpred", [p1, p2, p3], ["Can", "Target", "Sym"])
        self.assertEqual(pred.get_type(), "At")
        self.assertFalse(pred.test(time=400))
        p1.pose = np.array([[3, 4, 5], [6, 5, 7], [8, 9, 0]])
        # p2 doesn't have a value yet
        self.assertFalse(pred.test(time=400))
        p2.pose = np.array([[3, 4, 5], [6, 5, 8], [8, 9, 1]])
        self.assertTrue(pred.is_concrete())
        with self.assertRaises(Exception) as cm:
            pred.test(time=3)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target sym)'.")
        with self.assertRaises(Exception) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target sym)'.")
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))

    # TODO: test other predicates
