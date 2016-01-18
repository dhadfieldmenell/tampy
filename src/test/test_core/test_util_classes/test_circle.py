import unittest
from core.util_classes import circle

class TestCircle(unittest.TestCase):
    def test_basic(self):
        r = circle.RedCircle(3.5)
        self.assertEqual(r.radius, 3.5)
        self.assertEqual(r.color, "red")

        g = circle.GreenCircle("3.2")
        self.assertEqual(g.radius, 3.2)
        self.assertEqual(g.color, "green")

        b = circle.BlueCircle(4)
        self.assertEqual(b.radius, 4.0)
        self.assertEqual(b.color, "blue")
