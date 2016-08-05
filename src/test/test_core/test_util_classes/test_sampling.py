import unittest
import numpy as np
from openravepy import Environment, matrixFromAxisAngle
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2 import PR2
from core.util_classes import can, sampling, matrix
from core.internal_repr import parameter
import time

class TestSampling(unittest.TestCase):

    def test_sample_ee_from_target(self):
        solver = can_solver.CanSolver()
        env = setup_env()
        # env.SetViewer('qtcoin')
        target = setup_target()
        target.value = np.array([[0,0,0]]).T
        target.rotation = np.array([[1.1,.3,0]]).T

        dummy_targ_geom = can.BlueCan(0.04, 0.25)
        target_body = OpenRAVEBody(env, target.name, dummy_targ_geom)
        target_body.set_pose(target.value.flatten(), target.rotation.flatten())
        target_body.set_transparency(.7)

        robot = setup_robot
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        robot_body.set_transparency(.7)
        robot_body.set_pose(robot.pose.flatten())
        robot_body.set_dof(robot.backHeight, robot.lArmPose.flatten(), robot.lGripper, robot.rArmPose.flatten(), robot.rGripper)

        dummy_ee_pose_geom = GreenCan(.03,.3)
        ee_list = list(enumerate(sampling.get_ee_from_target(target)))
        for ee_pose in ee_list:
            ee_pos, ee_rot = ee_pose[1]
            body = OpenRAVEBody(env, "dummy"+str(ee_pose[0]), dummy_ee_pose_geom)
            body.set_pose(ee_pos, ee_rot)
            body.set_transparency(.9)

    def test_closest_arm_pose(self):
        env = setup_env()
        # env.SetViewer('qtcoin')
        can = setup_can()
        robot = setup_robot()
        can.pose = np.array([[0,-.2,.8]]).T
        can_body = OpenRAVEBody(env, can.name, can.geom)
        can_body.set_pose(can.pose.flatten(), can.rotation.flatten())
        can_body.set_transparency(.7)
        robot.pose = np.array([[-.5,0,0]]).T
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        robot_body.set_transparency(.7)
        robot_body.set_pose(robot.pose.flatten())
        robot_body.set_dof(robot.backHeight, robot.lArmPose.flatten(), robot.lGripper, robot.rArmPose.flatten(), robot.rGripper)
        can_trans = OpenRAVEBody.transform_from_obj_pose(can.pose, can.rotation)
        rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
        rot_mat = can_trans[:3, :3].dot(rot_mat[:3, :3])
        can_trans[:3, :3] = rot_mat
        torso_pose, arm_pose = sampling.get_torso_arm_ik(robot_body, can_trans, robot.rArmPose)
        robot_body.set_dof(robot.backHeight, robot.lArmPose.flatten(), robot.lGripper, arm_pose, robot.rGripper)
        # import ipdb; ipdb.set_trace()





# helper functions
def setup_env():
    return Environment()

def setup_robot(name = "pr2"):
    attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
    attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
    attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
    attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
    robot = parameter.Object(attrs, attr_types)
    # Set the initial arm pose so that pose is not close to joint limit
    robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
    robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
    return robot

def setup_robot_pose(name = "robot_Pose"):
    attrs = {"name": [name], "value": [(0, 0, 0)], "_type": ["RobotPose"], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
    attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
    attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
    attr_types = {"name": str, "value": matrix.Vector3d, "_type": str, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
    rPose = parameter.Symbol(attrs, attr_types)
    # Set the initial arm pose so that pose is not close to joint limit
    rPose.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
    rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
    return rPose

def setup_can(name = "can", geom = can.BlueCan):
    attrs = {"name": [name], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
    attr_types = {"name": str, "geom": can.BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
    can_obj = parameter.Object(attrs, attr_types)
    return can_obj

def setup_target(name = "target"):
    # This is the target parameter
    attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Target"]}
    attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
    target = parameter.Symbol(attrs, attr_types)
    return target

def setup_ee_pose(name = "ee_pose"):
    attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["EEPose"]}
    attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
    ee_pose = parameter.Symbol(attrs, attr_types)
    return ee_pose

def setup_obstacle(name = "table"):
    attrs = {"name": [name], "geom": [[1.5, 0.94, 0.15, .2, 0.2, 0.6, False]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
    attr_types = {"name": str, "geom": Table, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
    table = parameter.Object(attrs, attr_types)
    return table

def setup_box(name = "box"):
    attrs = {"name": [name], "geom": [[1,.5,.5]], "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Table"]}
    attr_types = {"name": str, "geom": Box, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
    box = parameter.Object(attrs, attr_types)
    return box
