import unittest
from IPython import embed as shell
from core.internal_repr import parameter
from core.util_classes import circle
from core.util_classes import matrix
from errors_exceptions import DomainConfigException
import numpy as np

class TestParameter(unittest.TestCase):
    def test_object(self):
        attrs = {"circ": 1, "test": 3.7, "test2": 5.3, "test3": 6.5, "pose": "undefined", "_type": "Can"}
        attr_types = {"test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str}
        with self.assertRaises(AssertionError) as cm:
            parameter.Object(attrs, attr_types)
        attrs["name"] = "param"
        with self.assertRaises(DomainConfigException) as cm:
            parameter.Object(attrs, attr_types)
        self.assertEqual(cm.exception.message, "Attribute 'name' for Object 'param' not defined in domain file.")
        attr_types["name"] = str
        param = parameter.Object(attrs, attr_types)
        self.assertEqual(param.name, "param")
        self.assertEqual(param.get_type(), "Can")
        self.assertFalse(param.is_symbol())
        self.assertFalse(param.is_defined())
        self.assertEqual(param.test, 3.7)
        self.assertEqual(param.test2, 5)
        self.assertEqual(param.test3, "6.5")
        self.assertEqual(param.circ.radius, 1)
        param.pose = matrix.Vector2d([2, 3])
        self.assertTrue(param.is_defined())
        self.assertEqual(param.pose.shape, (2, 1))

    def test_symbol(self):
        attrs = {"circ": 1, "test": 3.7, "test2": 5.3, "test3": 6.5, "pose": (3, 5), "_type": "CanSym", "name": "sym"}
        attr_types = {"test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
        with self.assertRaises(AssertionError) as cm:
            parameter.Symbol(attrs, attr_types)
        attrs["value"] = (4, 6)
        with self.assertRaises(DomainConfigException) as cm:
            parameter.Symbol(attrs, attr_types)
        self.assertEqual(cm.exception.message, "Attribute 'value' for Symbol 'sym' not defined in domain file.")
        attr_types["value"] = matrix.Vector2d
        param = parameter.Symbol(attrs, attr_types)
        self.assertEqual(param.name, "sym")
        self.assertEqual(param.get_type(), "CanSym")
        self.assertTrue(param.is_symbol())
        self.assertTrue(param.is_defined())
        self.assertEqual(param.test, 3.7)
        self.assertEqual(param.test2, 5)
        self.assertEqual(param.test3, "6.5")
        self.assertEqual(param.circ.radius, 1)
        param.pose = "undefined"
        self.assertTrue(param.is_defined())
        param.value = "undefined"
        self.assertFalse(param.is_defined())

    def test_abstract(self):
        # cannot instantiate Parameter directly
        with self.assertRaises(NotImplementedError) as cm:
            parameter.Parameter("can1")
        self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
        with self.assertRaises(NotImplementedError) as cm:
            parameter.Parameter("can2", 4, 76, 1)
        self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
        with self.assertRaises(NotImplementedError) as cm:
            parameter.Parameter()
        self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")

    def test_copy_object(self):
        attrs = {"name": "param", "circ": 1, "test": 3.7, "test2": 5.3, "test3": 6.5, "pose": [[3, 4, 5, 0], [6, 2, 1, 5], [1, 1, 1, 1]], "_type": "Can"}
        attr_types = {"name": str, "test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": np.array, "_type": str}
        p = parameter.Object(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertTrue(np.array_equal(p2.pose, [[3, 4, 5, 0, 0, 0, 0], [6, 2, 1, 5, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]]))
        p2 = p.copy(new_horizon=2)
        self.assertTrue(np.array_equal(p2.pose, [[3, 4], [6, 2], [1, 1]]))
        attrs["pose"] = "undefined"
        p = parameter.Object(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertEqual(p2.pose, "undefined")

    def test_copy_symbol(self):
        attrs = {"name": "param", "circ": 1, "test": 3.7, "test2": 5.3, "test3": 6.5, "value": [[3, 4, 5, 0], [6, 2, 1, 5], [1, 1, 1, 1]], "_type": "Can"}
        attr_types = {"name": str, "test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "value": np.array, "_type": str}
        p = parameter.Symbol(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertTrue(np.array_equal(p2.value, [[3, 4, 5, 0, 0, 0, 0], [6, 2, 1, 5, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]]))
        p2 = p.copy(new_horizon=2)
        self.assertTrue(np.array_equal(p2.value, [[3, 4], [6, 2], [1, 1]]))
        attrs["value"] = "undefined"
        p = parameter.Symbol(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertEqual(p2.value, "undefined")
