import unittest
from core.util_classes import viewer
from core.util_classes import circle
from core.util_classes import matrix
from core.internal_repr import parameter
from core.util_classes.robots import PR2
from core.util_classes.items import Box, Can, BlueCan, Table
from core.util_classes.plan_hdf5_serialization import PlanDeserializer, PlanSerializer
import numpy as np
import time


class TestViewer(unittest.TestCase):

    def setup_obstacle(self, name = "table"):
        attrs = {"name": [name], "geom": [[1.5, 0.94, 0.15, .2, 0.2, 0.6, False]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": Table, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        table = parameter.Object(attrs, attr_types)
        return table

    def setup_box(self, name = "box"):
        attrs = {"name": [name], "geom": [[1,.5,.5]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": Box, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        box = parameter.Object(attrs, attr_types)
        return box

    def setup_robot(self, name = "pr2"):
        attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
        return robot

    def setup_can(self, name = "can"):
        attrs = {"name": [name], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)
        return can


    def test_gridworldviewer(self):
        pass

    def test_openraveviewer_draw(self):
        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.GreenCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(2, 1)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
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
        attr_types = {"geom": circle.GreenCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(2, 1)], "_type": ["Can"], "name": ["can1"]}
        attr_types = {"geom": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
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

    def test_rotation_object(self):
        robot = self.setup_robot()
        can = self.setup_can()
        table = self.setup_obstacle()
        sTable = self.setup_box()

        robot.pose = np.array([[1,1,1]]).T
        can.pose = np.array([[0,0,0]]).T
        can.rotation = np.array([[1.57/2,0,0]]).T
        table.pose = np.array([[0,1,0]]).T
        table.rotation = np.array([[0,1.57/2,0]]).T
        sTable.pose = np.array([[0,0,2]]).T
        sTable.rotation = np.array([[0,0,1.57/2]]).T
        """
        To check whether this works uncomment the following
        """
        # testViewer = viewer.OpenRAVEViewer()
        # testViewer.draw_traj([robot, can, table, sTable], range(1))
        # import ipdb; ipdb.set_trace()

    def test_record(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("basket_plan.hdf5")
        view = viewer.OpenRAVEViewer.create_viewer(plan.env)
        # view.record_plan(plan, "basket_video")
        # view.animate_plan(plan, 1)
