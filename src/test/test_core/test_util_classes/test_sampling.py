import unittest
import numpy as np
from pma import can_solver
from openravepy import Environment, matrixFromAxisAngle
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.robots import PR2
from core.util_classes import items, pr2_sampling, matrix, param_setup
from core.util_classes.param_setup import ParamSetup
from core.internal_repr import parameter
import time

class TestSampling(unittest.TestCase):

    def test_sample_ee_from_target(self):
        solver = can_solver.CanSolver()
        env = ParamSetup.setup_env()
        # env.SetViewer('qtcoin')
        target = ParamSetup.setup_target()
        target.value = np.array([[0,0,0]]).T
        target.rotation = np.array([[1.1,.3,0]]).T

        dummy_targ_geom = items.BlueCan(0.04, 0.25)
        target_body = OpenRAVEBody(env, target.name, dummy_targ_geom)
        target_body.set_pose(target.value.flatten(), target.rotation.flatten())
        target_body.set_transparency(.7)

        robot = ParamSetup.setup_pr2()
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        robot_body.set_transparency(.7)
        robot_body.set_pose(robot.pose.flatten())
        dof_value_map = {"backHeight": robot.backHeight,
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": robot.rArmPose.flatten(),
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)

        dummy_ee_pose_geom = items.GreenCan(.03,.3)
        ee_list = list(enumerate(pr2_sampling.get_ee_from_target(target.value, target.rotation)))
        for ee_pose in ee_list:
            ee_pos, ee_rot = ee_pose[1]
            body = OpenRAVEBody(env, "dummy"+str(ee_pose[0]), dummy_ee_pose_geom)
            body.set_pose(ee_pos, ee_rot)
            body.set_transparency(.9)

    def test_closest_arm_pose(self):
        env = ParamSetup.setup_env()
        # env.SetViewer('qtcoin')
        can = ParamSetup.setup_blue_can()
        robot = ParamSetup.setup_pr2()
        can.pose = np.array([[0,-.2,.8]]).T
        can_body = OpenRAVEBody(env, can.name, can.geom)
        can_body.set_pose(can.pose.flatten(), can.rotation.flatten())
        can_body.set_transparency(.7)
        robot.pose = np.array([[-.5,0,0]]).T
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        robot_body.set_transparency(.7)
        robot_body.set_pose(robot.pose.flatten())
        dof_value_map = {"backHeight": robot.backHeight,
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": robot.rArmPose.flatten(),
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)
        can_trans = OpenRAVEBody.transform_from_obj_pose(can.pose, can.rotation)
        rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
        rot_mat = can_trans[:3, :3].dot(rot_mat[:3, :3])
        can_trans[:3, :3] = rot_mat
        torso_pose, arm_pose = pr2_sampling.get_torso_arm_ik(robot_body, can_trans, robot.rArmPose)
        dof_value_map = {"backHeight": robot.backHeight,
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": robot.rArmPose.flatten(),
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)
        # import ipdb; ipdb.set_trace()
