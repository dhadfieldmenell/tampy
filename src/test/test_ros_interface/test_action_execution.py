import sys
import unittest
import time
import main
import rospy
import numpy as np

import baxter_interface
from baxter_interface import CHECK_VERSION

from ros_interface import action_execution_2, action_execution
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from openravepy import Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
from core.util_classes import baxter_constants

class TestActionExecute(unittest.TestCase):

	def test_execute(self):
		'''
		This will try to talk to the Baxter, so launch the sim or real robot
		first
		'''
		# import ipdb; ipdb.set_trace()
		pd = PlanDeserializer()
		plan = pd.read_from_hdf5("cloth_manipulation_plan.hdf5")

		velocites = np.ones((plan.horizon, ))*1.5
		# slow_inds = np.array([range(19,39), range(58,78), range(97,117), range(136,156), range(175,195), range(214,234)]).flatten()
		# velocites[slow_inds] = 1.0
		baxter = plan.params['baxter']
		ee_time = traj_retiming(plan, velocites)
		baxter.time = ee_time.reshape((1, ee_time.shape[0]))
		print("Initializing node... ")
		rospy.init_node("rsdk_joint_trajectory_client")
		print("Getting robot state... ")
		rs = baxter_interface.RobotEnable(CHECK_VERSION)
		print("Enabling robot... ")
		rs.enable()
		print("Running. Ctrl-c to quit")
		baxter_interface.Gripper('left', CHECK_VERSION).calibrate()
		baxter_interface.Gripper('right', CHECK_VERSION).calibrate()
		import ipdb; ipdb.set_trace()
		action_execution.execute_plan(plan)
		# for action in plan.actions:
		# 	import ipdb; ipdb.set_trace()
		# 	action_execution_2.execute_action(action)

def traj_retiming(plan, velocity):
    baxter = plan.params['baxter']
    rave_body = baxter.openrave_body
    body = rave_body.env_body
    lmanip = body.GetManipulator("left_arm")
    rmanip = body.GetManipulator("right_arm")
    left_ee_pose = []
    right_ee_pose = []
    for t in range(plan.horizon):
        rave_body.set_dof({
            'lArmPose': baxter.lArmPose[:, t],
            'lGripper': baxter.lGripper[:, t],
            'rArmPose': baxter.rArmPose[:, t],
            'rGripper': baxter.rGripper[:, t]
        })
        rave_body.set_pose([0,0,baxter.pose[:, t]])

        left_ee_pose.append(lmanip.GetTransform()[:3, 3])
        right_ee_pose.append(rmanip.GetTransform()[:3, 3])
    time = np.zeros(plan.horizon)
    # import ipdb; ipdb.set_trace()
    for t in range(plan.horizon-1):
        left_dist = np.linalg.norm(left_ee_pose[t+1] - left_ee_pose[t])
        right_dist = np.linalg.norm(right_ee_pose[t+1] - right_ee_pose[t])
        time_spend = max(left_dist, right_dist)/velocity[t]
        time[t+1] = time[t] + time_spend
    return time
