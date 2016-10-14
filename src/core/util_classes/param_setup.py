from openravepy import Environment
from core.util_classes import matrix, robots, can, circle
from core.util_classes.table import Table
from core.util_classes.box import Box
from core.internal_repr import parameter
import numpy as np

class ParamSetup(object):
    """
        Example parameter setup for testing purposes
    """

    @staticmethod
    def setup_env():
        return Environment()

    @staticmethod
    def setup_pr2(name = "pr2"):
        attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": robots.PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
        return robot

    @staticmethod
    def setup_pr2_pose(name = "pr2_pose"):
        attrs = {"name": [name], "value": [(0, 0, 0)], "_type": ["RobotPose"], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "value": matrix.Vector3d, "_type": str, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        rPose = parameter.Symbol(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        rPose.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
        return rPose

    @staticmethod
    def setup_baxter(name = "baxter"):
        attrs = {"name": [name], "pose": [(0)], "_type": ["Robot"], "geom": [], "lGripper": [0.02], "rGripper": [0.02]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Value, "_type": str, "geom": robots.Baxter, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Object(attrs, attr_types)
        return robot

    @staticmethod
    def setup_baxter_pose(name = "baxter_pose"):
        attrs = {"name": [name], "value": [(0)], "_type": ["RobotPose"], "geom": [], "lGripper": [0.02], "rGripper": [0.02]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "value": matrix.Value, "_type": str, "geom": robots.Baxter, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Symbol(attrs, attr_types)
        return robot

    @staticmethod
    def setup_green_can(name = "green_can",  geom = (0.02,0.25)):
        attrs = {"name": [name], "geom": geom, "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": can.GreenCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_blue_can(name = "blue_can",  geom = (0.02,0.25)):
        attrs = {"name": [name], "geom": geom, "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": can.BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_red_can(name = "red_can", geom = (0.02,0.25)):
        attrs = {"name": [name], "geom": geom, "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": can.RedCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_target(name = "target"):
        # This is the target parameter
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Target"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        target = parameter.Symbol(attrs, attr_types)
        return target

    @staticmethod
    def setup_pr2_ee_pose(name = "pr2_ee_pose"):
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["EEPose"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        ee_pose = parameter.Symbol(attrs, attr_types)
        return ee_pose

    @staticmethod
    def setup_table(name = "table"):
        attrs = {"name": [name], "geom": [[1.5, 0.94, 0.15, .2, 0.2, 0.6, False]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": Table, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        table = parameter.Object(attrs, attr_types)
        return table

    @staticmethod
    def setup_box(name = "box"):
        attrs = {"name": [name], "geom": [[1,.5,.5]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": Box, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        box = parameter.Object(attrs, attr_types)
        return box
