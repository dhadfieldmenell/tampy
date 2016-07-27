import unittest
from IPython import embed as shell
from core.internal_repr import parameter
from core.util_classes import circle
from core.util_classes import matrix
from errors_exceptions import DomainConfigException
import numpy as np

class TestParameter(unittest.TestCase):
    def test_object(self):
        attrs = {"circ": [1], "test": [3.7], "test2": [5.3], "test3": [6.5], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str}
        with self.assertRaises(AssertionError) as cm:
            parameter.Object(attrs, attr_types)
        attrs["name"] = ["param"]
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

        # test get_attr_type
        self.assertEqual(param.get_attr_type("test"), float)
        self.assertEqual(param.get_attr_type("test2"), int)
        self.assertEqual(param.get_attr_type("test3"), str)
        self.assertEqual(param.get_attr_type("circ"), circle.BlueCircle)
        self.assertEqual(param.get_attr_type("pose"), matrix.Vector2d)
        self.assertEqual(param.get_attr_type("_type"), str)
        self.assertEqual(param.get_attr_type("_attr_types"), dict)
        with self.assertRaises(KeyError):
            param.get_attr_type("does not exist")

    def test_symbol(self):
        attrs = {"circ": [1], "test": [3.7], "test2": [5.3], "test3": [6.5], "pose": [(3, 5)], "_type": ["CanSym"], "name": ["sym"]}
        attr_types = {"test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
        with self.assertRaises(AssertionError) as cm:
            parameter.Symbol(attrs, attr_types)
        attrs["value"] = [(4, 6)]
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

        # test get_attr_type
        self.assertEqual(param.get_attr_type("test"), float)
        self.assertEqual(param.get_attr_type("test2"), int)
        self.assertEqual(param.get_attr_type("test3"), str)
        self.assertEqual(param.get_attr_type("circ"), circle.BlueCircle)
        self.assertEqual(param.get_attr_type("pose"), matrix.Vector2d)
        self.assertEqual(param.get_attr_type("_type"), str)
        self.assertEqual(param.get_attr_type("_attr_types"), dict)
        with self.assertRaises(KeyError):
            param.get_attr_type("does not exist")

    """
    Now using the super class for saving openrave_bodies
    """
    # def test_abstract(self):
    #     # cannot instantiate Parameter directly
    #     with self.assertRaises(NotImplementedError) as cm:
    #         parameter.Parameter("can1")
    #     self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
    #     with self.assertRaises(NotImplementedError) as cm:
    #         parameter.Parameter("can2", 4, 76, 1)
    #     self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
    #     with self.assertRaises(NotImplementedError) as cm:
    #         parameter.Parameter()
    #     self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")

    def test_copy_object(self):
        attrs = {"name": ["param"], "circ": [1], "test": [3.7], "test2": [5.3], "test3": [6.5], "pose": [[[3, 4, 5, 0], [6, 2, 1, 5], [1, 1, 1, 1]]], "rotation": [[[1, 2, 3],[4, 5, 6]]], "_type": ["Can"]}
        attr_types = {"name": str, "test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "pose": matrix.Vector3d, "rotation": matrix.Vector2d, "_type": str}
        p = parameter.Object(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertTrue(np.allclose(p2.pose, [[3, 4, 5, 0, np.NaN, np.NaN, np.NaN], [6, 2, 1, 5, np.NaN, np.NaN, np.NaN], [1, 1, 1, 1, np.NaN, np.NaN, np.NaN]], equal_nan=True))
        self.assertTrue(np.allclose(p2.rotation, [[1, 2, 3, np.NaN, np.NaN, np.NaN, np.NaN], [4, 5, 6, np.NaN, np.NaN, np.NaN, np.NaN]], equal_nan=True))
        p2 = p.copy(new_horizon=2)
        self.assertTrue(np.array_equal(p2.pose, [[3, 4], [6, 2], [1, 1]]))
        self.assertTrue(np.array_equal(p2.rotation, [[1, 2], [4, 5]]))
        attrs["pose"] = "undefined"
        attrs["rotation"] = "undefined"
        p = parameter.Object(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        new_pose = np.empty((3, 7))
        new_pose[:] = np.NaN
        self.assertTrue(np.allclose(p2.pose, new_pose, equal_nan=True))
        new_pose = np.empty((2, 7))
        new_pose[:] = np.NaN
        self.assertTrue(np.allclose(p2.rotation, new_pose, equal_nan=True))

    def test_copy_symbol(self):
        attrs = {"name": ["param"], "circ": [1], "test": [3.7], "test2": [5.3], "test3": [6.5], "value": ["(3, 6)"], "rotation": ["(1,2,3)"], "_type": ["Can"]}
        attr_types = {"name": str, "test": float, "test3": str, "test2": int, "circ": circle.BlueCircle, "value": matrix.Vector2d, "rotation": matrix.Vector3d, "_type": str}
        p = parameter.Symbol(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        self.assertTrue(np.allclose(p2.value, [[3], [6]]))
        self.assertTrue(np.allclose(p2.rotation, [[1], [2], [3]]))
        p2 = p.copy(new_horizon=2)
        self.assertTrue(np.allclose(p2.value, [[3], [6]]))
        self.assertTrue(np.allclose(p2.rotation, [[1], [2], [3]]))
        attrs["value"] = ["undefined"]
        attrs["rotation"] = ["undefined"]
        p = parameter.Symbol(attrs, attr_types)
        p2 = p.copy(new_horizon=7)
        self.assertEqual(p2.name, "param")
        self.assertEqual(p2.test, 3.7)
        arr = np.empty((2, 1))
        arr[:] = np.NaN
        self.assertTrue(np.allclose(p2.value, arr, equal_nan=True))

        arr = np.empty((3, 1))
        arr[:] = np.NaN
        self.assertTrue(np.allclose(p2.rotation, arr, equal_nan=True))
