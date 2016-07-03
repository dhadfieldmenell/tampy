import unittest
from core.util_classes import can

class TestCan(unittest.TestCase):
    def test_basic(self):
        r = can.RedCan(3.5, 7.1)
        self.assertEqual(r.radius, 3.5)
        self.assertEqual(r.height, 7.1)
        self.assertEqual(r.color, "red")

        b = can.BlueCan(4, 9)
        self.assertEqual(b.radius, 4.0)
        self.assertEqual(b.height, 9.0)
        self.assertEqual(b.color, "blue")
