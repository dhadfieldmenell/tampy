import unittest
from core.util_classes import matrix
import numpy as np

class TestMatrix(unittest.TestCase):
    def test_abstract(self):
        with self.assertRaises(NotImplementedError) as cm:
            matrix.Matrix()
        self.assertEqual(cm.exception.message, "Override this.")

    def test_vector2d(self):
        with self.assertRaises(AssertionError) as cm:
            matrix.Vector2d("(3, 4, 5)")
        with self.assertRaises(AssertionError) as cm:
            matrix.Vector2d("(3, )")
        v1 = matrix.Vector2d("(7, 8)")
        v2 = matrix.Vector2d((7, 8))
        v3 = matrix.Vector2d("(7, 8")
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(np.array_equal(v1, v3))
        self.assertEqual(v1.shape, (2, 1))
        self.assertEqual(v1[1][0], [8])

    def test_vector3d(self):
        with self.assertRaises(AssertionError) as cm:
            matrix.Vector3d("(3, 4)")
        with self.assertRaises(AssertionError) as cm:
            matrix.Vector3d("(3, )")

        v1 = matrix.Vector3d("(4, 7, 8)")
        v2 = matrix.Vector3d((4, 7, 8))
        v3 = matrix.Vector3d("(4, 7, 8")
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(np.array_equal(v1, v3))
        self.assertEqual(v2.shape, (3, 1))
        self.assertEqual(v2[2][0], [8])

    def test_pr2posevector(self):
        v1 = matrix.Vector3d("(4, 7, 8)")
        v2 = matrix.Vector3d((4, 7, 8))
        v3 = matrix.Vector3d("(4, 7, 8")
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(np.array_equal(v1, v3))
        self.assertEqual(v2.shape, (3, 1))
        self.assertEqual(v2[2][0], [8])
