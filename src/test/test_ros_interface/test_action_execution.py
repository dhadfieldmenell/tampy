import sys
import unittest
import time
import main
import rospy
import numpy as np
from ros_interface import action_execution
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
		plan = pd.read_from_hdf5("plan.hdf5")
		# baxter = plan.params['baxter']
		# natural_state = np.array([0., 1.42, 0., 0.02, 0., 0.22, -0.]).reshape((7, 1))
		# natural_traj = np.repeat(natural_state, 40, axis=1)
		# baxter.lArmPose = natural_traj
		# action = plan.actions[0]
		# action_execution.execute_action(plan)
		print("Initializing node... ")
		rospy.init_node("rsdk_joint_trajectory_client")
		print("Getting robot state... ")
		rs = baxter_interface.RobotEnable(CHECK_VERSION)
		print("Enabling robot... ")
		rs.enable()
		print("Running. Ctrl-c to quit")
		action_execution.execute_action(plan.actions[0])
