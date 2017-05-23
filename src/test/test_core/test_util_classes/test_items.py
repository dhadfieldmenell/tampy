from core.util_classes import items
import unittest
class TestItems(unittest.TestCase):
    def test_box(self):
        test_box = items.Box([1,1,1])
        self.assertEqual(test_box.dim, [1,1,1])
        self.assertEqual(test_box.length, 1)
        self.assertEqual(test_box.height, 1)
        self.assertEqual(test_box.width, 1)

    def test_can(self):
        r = items.RedCan(3.5, 7.1)
        self.assertEqual(r.radius, 3.5)
        self.assertEqual(r.height, 7.1)
        self.assertEqual(r.color, "red")

        b = items.BlueCan(4, 9)
        self.assertEqual(b.radius, 4.0)
        self.assertEqual(b.height, 9.0)
        self.assertEqual(b.color, "blue")

    def test_circle(self):
        r = items.RedCircle(3.5)
        self.assertEqual(r.radius, 3.5)
        self.assertEqual(r.color, "red")

        g = items.GreenCircle("3.2")
        self.assertEqual(g.radius, 3.2)
        self.assertEqual(g.color, "green")

        b = items.BlueCircle(4)
        self.assertEqual(b.radius, 4.0)
        self.assertEqual(b.color, "blue")

    def test_table_dim(self):
        table_dim = [1, 2]
        thickness = 0.2
        leg_dim = [0.5, 0.5]
        leg_height = 0.6
        back = False
        dimension = [table_dim[0], table_dim[1], thickness, leg_dim[0], leg_dim[1], leg_height, back]
        test_table = items.Table(dimension)
        self.assertEqual(table_dim, test_table.table_dim)
        self.assertEqual(thickness, test_table.thickness)
        self.assertEqual(leg_dim, test_table.leg_dim)
        self.assertEqual(leg_height, test_table.leg_height)
        self.assertEqual(back, test_table.back)

    def test_wall(self):
        wall = items.Wall("closet")
        self.assertEqual("closet", wall.wall_type)
