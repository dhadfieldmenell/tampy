from openravepy import Environment
from core.util_classes import matrix, robots, items
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
        attrs = {"name": [name], "pose": [(0)], "_type": ["Robot"], "geom": [], "lGripper": [0.02], "time":[0], "rGripper": [0.02]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Value, "_type": str, "geom": robots.Baxter, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value, "time": matrix.Value}
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
        attr_types = {"name": str, "geom": items.GreenCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_blue_can(name = "blue_can",  geom = (0.02,0.25)):
        attrs = {"name": [name], "geom": geom, "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": items.BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_red_can(name = "red_can", geom = (0.02,0.25)):
        attrs = {"name": [name], "geom": geom, "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": items.RedCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can_obj = parameter.Object(attrs, attr_types)
        return can_obj

    @staticmethod
    def setup_sphere(name = "sphere", geom = (0.25,)):
        attrs = {"name": [name], "geom": geom, "pose": [(0, 0, 0)], "rotation": [(0, 0, 0)], "_type": ["Sphere"]}
        attr_types = {"name": str, "geom": items.Sphere, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        sphere_obj = parameter.Object(attrs, attr_types)
        return sphere_obj

    @staticmethod
    def setup_target(name = "target"):
        # This is the target parameter
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Target"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        target = parameter.Symbol(attrs, attr_types)
        return target


    @staticmethod
    def setup_ee_pose(name = "ee_pose"):
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["EEPose"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        ee_pose = parameter.Symbol(attrs, attr_types)
        return ee_pose

    @staticmethod
    def setup_table(name = "table"):
        attrs = {"name": [name], "geom": [[1.5, 0.94, 0.15, .2, 0.2, 0.6, False]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": items.Table, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        table = parameter.Object(attrs, attr_types)
        return table

    @staticmethod
    def setup_box(name = "box", geom = [1,.5,.5]):
        attrs = {"name": [name], "geom": [geom], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
        attr_types = {"name": str, "geom": items.Box, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        box = parameter.Object(attrs, attr_types)
        return box

    @staticmethod
    def setup_basket(name = "basket"):
        attrs = {"name": [name], "geom": [], "pose": ["undefined"], "rotation": [(0, 0, np.pi/2)], "_type": ["Basket"]}
        attr_types = {"name": str, "geom": items.Basket, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        basket = parameter.Object(attrs, attr_types)
        return basket

    @staticmethod
    def setup_basket_target(name = "basket_target"):
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0, 0, np.pi/2)], "_type": ["BasketTarget"]}
        attr_types = {"name": str, "geom": items.Basket, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        basket_target = parameter.Symbol(attrs, attr_types)
        return basket_target

    @staticmethod
    def setup_ee_vel(name = "ee_vel"):
        attrs = {"name": [name], "value": ["undefined"], "rotation": ["undefined"], "_type": ["EEVel"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        ee_vel = parameter.Symbol(attrs, attr_types)
        return ee_vel

    @staticmethod
    def setup_washer(name = "washer"):
        attrs = {"name": [name], "pose": [[0.505, 0.961, 1.498]], "door": [0.0], "rotation": [[3.141592653589793, 0, 0]], "geom": [True, False], "_type": ["Washer"]}
        attr_types = {"name": str, "pose": matrix.Vector3d, "door": matrix.Vector1d, "rotation": matrix.Vector3d, "geom": robots.Washer, "_type": str}
        washer = parameter.Object(attrs, attr_types)
        return washer

    @staticmethod
    def setup_washer_pose(name = "washer_pose"):
        attrs = {"name": [name], "value": [[0.505, 0.961, 1.498]], "door": [0], "rotation": [[3.141592653589793, 0, 0]], "geom": [True, False], "_type": ["WasherPose"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "door": matrix.Vector1d, "rotation": matrix.Vector3d, "geom": robots.Washer, "_type": str}
        washer_pose = parameter.Symbol(attrs, attr_types)
        return washer_pose

    @staticmethod
    def setup_cloth(name = 'cloth'):
        attrs = {"name": [name], "pose": [[0.571, 0.017,  0.90]], "rotation": [[0, 0, 0]], "geom": [], "_type": ["Cloth"]}
        attr_types = {"name": str, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "geom": items.Cloth, "_type": str}
        cloth = parameter.Object(attrs, attr_types)
        return cloth

    @staticmethod
    def setup_cloth_target(name = 'cloth_target'):
        attrs = {"name": [name], "value": ["undefined"], "rotation": ["undefined"], "_type": ["ClothTarget"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        cloth_target = parameter.Symbol(attrs, attr_types)
        return cloth_target
