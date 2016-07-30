import unittest
from core.util_classes import box

class TestCan(unittest.TestCase):
    def test_basic(self):
        test_box = box.Box([1,1,1])
        self.assertEqual(test_box.dim, [1,1,1])
        self.assertEqual(test_box.length, 1)
        self.assertEqual(test_box.height, 1)
        self.assertEqual(test_box.width, 1)
