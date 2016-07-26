from core.util_classes.wall import Wall
import unittest

class TestWall(unittest.TestCase):
    def test_wall(self):
        wall = Wall("closet")
        self.assertEqual("closet", wall.wall_type)
