import unittest
from core.util_classes import viewer
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.internal_repr import parameter
import numpy as np
import time


class TestViewer(unittest.TestCase):

    def test_gridworldviewer(self):
        pass

    def test_openraveviewer_draw(self):
        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(2, 1)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        green_can.pose = np.array([[1,2,3,4,5],
                                    [3,4,5,6,7]])
        blue_can.pose = np.array([[3,4,5,6,7],
                                [1,2,3,4,5]])
        """
        To check whether this works uncomment the following
        """
        # testViewer = viewer.OpenRAVEViewer()
        # for t in range(5):
        #     testViewer.draw([green_can, blue_can], t)
        #     time.sleep(1)


    def test_openraveviewer_draw_traj(self):
        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(2, 1)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        green_can.pose = np.array([[1,2,3,4,5],
                                    [3,4,5,6,7]])
        blue_can.pose = np.array([[3,4,5,6,7],
                                [1,2,3,4,5]])
        """
        To check whether this works uncomment the following
        """
        # testViewer = viewer.OpenRAVEViewer()
        # testViewer.draw_traj([green_can, blue_can], range(5))
        # import ipdb; ipdb.set_trace()

    
