import unittest
from core.util_classes import table
import numpy as np


class TestViewer(unittest.TestCase):
    def test_table_dim(self):
        table_dim = [1, 2]
        thickness = 0.2
        leg_dim = [0.5, 0.5]
        leg_height = 0.6
        back = False
        dimension = [table_dim[0], table_dim[1], thickness, leg_dim[0], leg_dim[1], leg_height, back]
        test_table = table.Table(dimension)
        self.assertEqual(table_dim, test_table.table_dim)
        self.assertEqual(thickness, test_table.thickness)
        self.assertEqual(leg_dim, test_table.leg_dim)
        self.assertEqual(leg_height, test_table.leg_height)
        self.assertEqual(back, test_table.back)
