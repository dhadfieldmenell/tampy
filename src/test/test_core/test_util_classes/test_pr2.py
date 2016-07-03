import unittest
from core.util_classes import pr2

class TestPR2(unittest.TestCase):

    def test_basic(self):
        r = pr2.PR2(5)
        self.assertEqual(r.geom, 5)

    pass