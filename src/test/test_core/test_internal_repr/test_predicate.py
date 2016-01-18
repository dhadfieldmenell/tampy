import unittest
from core.internal_repr import predicate
from core.internal_repr import parameter
import numpy as np

class TestPredicate(unittest.TestCase):
    # TODO
    def test_param_validation(self):
        p1 = parameter.Can("can")
        p2 = parameter.Target("target")
        with self.assertRaises(Exception) as cm:
            predicate.At("errorpred", [p1, p2], ["Target", "Can"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (At can target)'.")
        with self.assertRaises(Exception) as cm:
            predicate.At("errorpred", [p1, p2], ["Can"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (At can target)'.")
        with self.assertRaises(Exception) as cm:
            predicate.At("errorpred", [p1, p2], ["Can", "Target", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'errorpred: (At can target)'.")

    def test_at(self):
        p1 = parameter.Can("can")
        p2 = parameter.Target("target")
        pred = predicate.At("testpred", [p1, p2], ["Can", "Target"])
        p1.pose = np.array([[3, 4, 5], [6, 5, 7], [8, 9, 0]])
        # p2 doesn't have a value yet
        self.assertFalse(pred.test(time=400))

        p2.pose = np.array([[3, 4, 5], [6, 5, 8], [8, 9, 1]])
        with self.assertRaises(Exception) as cm:
            pred.test(time=3)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        with self.assertRaises(Exception) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
