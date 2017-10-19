# I used the Trajectory class from one of the Baxter examples, so for now I need
# to leave this copyright info in place. I'll code my own later so we don't have
# to have this.

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from bisect import bisect
from copy import copy
import operator
import sys
import threading

from core.util_classes.plan_hdf5_serialization import PlanDeserializer, PlanSerializer
from pma.robot_ll_solver import RobotLLSolver
from ros_interface.environment_monitor import EnvironmentMonitor

import rospy

import baxter_interface
from baxter_interface import CHECK_VERSION
import baxter_dataflow

import actionlib

import threading
import traceback
import Queue

from control_msgs.msg import (
	FollowJointTrajectoryAction,
	FollowJointTrajectoryGoal,
	JointTolerance
)
from trajectory_msgs.msg import (
	JointTrajectoryPoint,
)

import time
import numpy as np

from core.util_classes.viewer import OpenRAVEViewer

joints = ['_s0', '_s1', '_e0', '_e1', '_w0', '_w1', '_w2']

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

class Trajectory(object):
	def __init__(self):
		#create our action server clients
		self._left_client = actionlib.SimpleActionClient(
			'robot/limb/left/follow_joint_trajectory',
			FollowJointTrajectoryAction,
		)
		self._right_client = actionlib.SimpleActionClient(
			'robot/limb/right/follow_joint_trajectory',
			FollowJointTrajectoryAction,
		)

		#verify joint trajectory action servers are available
		l_server_up = self._left_client.wait_for_server(rospy.Duration(10.0))
		r_server_up = self._right_client.wait_for_server(rospy.Duration(10.0))
		if not l_server_up or not r_server_up:
			msg = ("Action server not available."
				   " Verify action server availability.")
			rospy.logerr(msg)
			rospy.signal_shutdown(msg)
			sys.exit(1)
		#create our goal request
		self._l_goal = FollowJointTrajectoryGoal()
		self._r_goal = FollowJointTrajectoryGoal()

		for jnt in joints:
			left_tol = JointTolerance()
			left_tol.name = 'left'+jnt
			left_tol.position = .3
			self._l_goal.path_tolerance.append(left_tol)
			right_tol = JointTolerance()
			right_tol.name = 'right'+jnt
			right_tol.position = .3
			self._r_goal.path_tolerance.append(right_tol)

		#limb interface - current angles needed for start move
		self._l_arm = baxter_interface.Limb('left')
		self._r_arm = baxter_interface.Limb('right')

		#gripper interface - for gripper command playback
		self._l_gripper = baxter_interface.Gripper('left', CHECK_VERSION)
		self._r_gripper = baxter_interface.Gripper('right', CHECK_VERSION)

		#flag to signify the arm trajectories have begun executing
		self._arm_trajectory_started = False
		#reentrant lock to prevent same-thread lockout
		self._lock = threading.RLock()

		# Verify Grippers Have No Errors and are Calibrated
		if self._l_gripper.error():
			self._l_gripper.reset()
		if self._r_gripper.error():
			self._r_gripper.reset()
		if (not self._l_gripper.calibrated() and
			self._l_gripper.type() != 'custom'):
			self._l_gripper.calibrate()
		if (not self._r_gripper.calibrated() and
			self._r_gripper.type() != 'custom'):
			self._r_gripper.calibrate()

		#gripper goal trajectories
		self._l_grip = FollowJointTrajectoryGoal()
		self._r_grip = FollowJointTrajectoryGoal()

		# Timing offset to prevent gripper playback before trajectory has started
		self._slow_move_offset = 0.0
		self._trajectory_start_offset = rospy.Duration(0.0)
		self._trajectory_actual_offset = rospy.Duration(0.0)

		#param namespace
		self._param_ns = '/rsdk_joint_trajectory_action_server/'

		#gripper control rate
		self._gripper_rate = 4.0  # Hz

	def _execute_gripper_commands(self):
		r_cmd = self._r_grip.trajectory.points
		l_cmd = self._l_grip.trajectory.points
		pnt_times = [pnt.time_from_start.to_sec() for pnt in r_cmd]
		end_time = pnt_times[-1]
		rate = rospy.Rate(self._gripper_rate)
		start_time = rospy.get_time() - self._trajectory_actual_offset.to_sec()
		now_from_start = rospy.get_time() - start_time
		while(now_from_start < end_time + (1.0 / self._gripper_rate) and
			  not rospy.is_shutdown()):
			idx = bisect(pnt_times, now_from_start) - 1
			if self._r_gripper.type() != 'custom':
				self._r_gripper.command_position(r_cmd[idx].positions[0])
			if self._l_gripper.type() != 'custom':
				self._l_gripper.command_position(l_cmd[idx].positions[0])

			rate.sleep()
			now_from_start = rospy.get_time() - start_time

	def _add_point(self, positions, side, time):
		#creates a point in trajectory with time_from_start and positions
		point = JointTrajectoryPoint()
		point.positions = copy(positions)
		point.time_from_start = rospy.Duration(time)
		if side == 'left':
			self._l_goal.trajectory.points.append(point)
		elif side == 'right':
			self._r_goal.trajectory.points.append(point)
		elif side == 'left_gripper':
			self._l_grip.trajectory.points.append(point)
		elif side == 'right_gripper':
			self._r_grip.trajectory.points.append(point)

	def load_trajectory(self, action):
		baxter = filter(lambda p: p.name=='baxter', action.params)[0] # plan.params['robot']
		joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', \
					   'left_w1', 'left_w2', 'right_s0', 'right_s1', 'right_e0', \
					   'right_e1', 'right_w0', 'right_w1', 'right_w2']
		for name in joint_names:
			if 'left' == name[:-3]:
				self._l_goal.trajectory.joint_names.append(name)
			elif 'right' == name[:-3]:
				self._r_goal.trajectory.joint_names.append(name)

		def find_start_offset(pos):
			#create empty lists
			cur = []
			cmd = []
			dflt_vel = []
			vel_param = self._param_ns + "%s_default_velocity"
			#for all joints find our current and first commanded position
			#reading default velocities from the parameter server if specified
			for name in joint_names:
				if 'left' == name[:-3]:
					cmd.append(pos[name])
					cur.append(self._l_arm.joint_angle(name))
					prm = rospy.get_param(vel_param % name, 0.25)
					dflt_vel.append(prm)
				elif 'right' == name[:-3]:
					cmd.append(pos[name])
					cur.append(self._r_arm.joint_angle(name))
					prm = rospy.get_param(vel_param % name, 0.25)
					dflt_vel.append(prm)
			diffs = map(operator.sub, cmd, cur)
			diffs = map(operator.abs, diffs)
			#determine the largest time offset necessary across all joints
			offset = max(map(operator.div, diffs, dflt_vel))
			return offset

		ts = action.active_timesteps
		real_ts = 0
		for t in range(ts[0], ts[1]):
			cmd = {}
			for i in range(7):
				cmd['left'+joints[i]] = baxter.lArmPose[i][t]
				cmd['right'+joints[i]] = baxter.rArmPose[i][t]
			cmd['left_gripper'] = 100.0 if baxter.lGripper[0][t] > .015 else 0
			cmd['right_gripper'] = 100.0 if baxter.rGripper[0][t] > .015 else 0
			if t == ts[0]:
				cur_cmd = [self._l_arm.joint_angle(jnt) for jnt in self._l_goal.trajectory.joint_names]
				self._add_point(cur_cmd, 'left', 0.0)
				cur_cmd = [self._r_arm.joint_angle(jnt) for jnt in self._r_goal.trajectory.joint_names]
				self._add_point(cur_cmd, 'right', 0.0)
				# cur_cmd = [cmd['left_gripper']]
				# self._add_point(cur_cmd, 'left_gripper', 0.0)
				# cur_cmd = [cmd['right_gripper']]
				# self._add_point(cur_cmd, 'right_gripper', 0.0)
				start_offset = find_start_offset(cmd)
				self._slow_move_offset = start_offset
				self._trajectory_start_offset = rospy.Duration(start_offset)

			cur_cmd = [cmd[jnt] for jnt in self._l_goal.trajectory.joint_names]
			self._add_point(cur_cmd, 'left', real_ts + start_offset)
			cur_cmd = [cmd[jnt] for jnt in self._r_goal.trajectory.joint_names]
			self._add_point(cur_cmd, 'right', real_ts + start_offset)
			cur_cmd = [cmd['left_gripper']]
			self._add_point(cur_cmd, 'left_gripper', real_ts + start_offset)
			cur_cmd = [cmd['right_gripper']]
			self._add_point(cur_cmd, 'right_gripper', real_ts + start_offset)
			real_ts += 0.75 # action.ee_retiming[t-ts[0]]

	def _feedback(self, data):
		# Test to see if the actual playback time has exceeded
		# the move-to-start-pose timing offset
		if (not self._get_trajectory_flag() and
			  data.actual.time_from_start >= self._trajectory_start_offset):
			self._set_trajectory_flag(value=True)
			self._trajectory_actual_offset = data.actual.time_from_start

	def _set_trajectory_flag(self, value=False):
		with self._lock:
			# Assign a value to the flag
			self._arm_trajectory_started = value

	def _get_trajectory_flag(self):
		temp_flag = False
		with self._lock:
			# Copy to external variable
			temp_flag = self._arm_trajectory_started
		return temp_flag

	def start(self):
		"""
		Sends FollowJointTrajectoryAction request
		"""
		self._left_client.send_goal(self._l_goal, feedback_cb=self._feedback)
		self._right_client.send_goal(self._r_goal, feedback_cb=self._feedback)
		# Syncronize playback by waiting for the trajectories to start
		while not rospy.is_shutdown() and not self._get_trajectory_flag():
			rospy.sleep(0.05)
		self._execute_gripper_commands()

	def stop(self):
		"""
		Preempts trajectory execution by sending cancel goals
		"""
		if (self._left_client.gh is not None and
			self._left_client.get_state() == actionlib.GoalStatus.ACTIVE):
			self._left_client.cancel_goal()

		if (self._right_client.gh is not None and
			self._right_client.get_state() == actionlib.GoalStatus.ACTIVE):
			self._right_client.cancel_goal()

		#delay to allow for terminating handshake
		rospy.sleep(0.1)

	def wait(self):
		"""
		Waits for and verifies trajectory execution result
		"""
		#create a timeout for our trajectory execution
		#total time trajectory expected for trajectory execution plus a buffer
		last_time = self._r_goal.trajectory.points[-1].time_from_start.to_sec()
		time_buffer = rospy.get_param(self._param_ns + 'goal_time', 0.0) + 1 # 2.5
		timeout = rospy.Duration(self._slow_move_offset +
								 last_time +
								 time_buffer)

		l_finish = self._left_client.wait_for_result(timeout)
		r_finish = self._right_client.wait_for_result(timeout)
		l_result = (self._left_client.get_result().error_code == 0)
		r_result = (self._right_client.get_result().error_code == 0)

		#verify result
		if all([l_finish, r_finish, l_result, r_result]):
			return True
		else:
			msg = ("Trajectory action failed or did not finish before "
				   "timeout/interrupt.")
			rospy.logwarn(msg)
			return False


def enforce_joint_limits(plan):
	robot = plan.params['baxter'].openrave_body
	lb_limit, ub_limit = robot.env_body.GetDOFLimits()
	dof_map = robot._geom.dof_map
	dof_inds = np.r_[dof_map["lArmPose"], dof_map["lGripper"], dof_map["rArmPose"], dof_map["rGripper"]]
	active_ub = ub_limit[dof_inds].flatten()
	active_lb = lb_limit[dof_inds].flatten()
	for i in range(7):
		for j in range(plan.horizon):
			if plan.params['baxter'].lArmPose[i, j] < active_lb[i]:
				plan.params['baxter'].lArmPose[i, j] = active_lb[i] + .001*active_lb[i]
			if plan.params['baxter'].lArmPose[i,j] > active_ub[i]:
				plan.params['baxter'].lArmPose[i, j] = active_ub[i] - .001*active_ub[i]
	for i in range(7):
		for j in range(plan.horizon):
			if plan.params['baxter'].lArmPose[i, j] < active_lb[8+i]:
				plan.params['baxter'].lArmPose[i, j] = active_lb[8+i] +.001*active_lb[8+i]
			if plan.params['baxter'].lArmPose[i,j] > active_ub[8+i]:
				plan.params['baxter'].lArmPose[i, j] = active_ub[8+i] - .001*active_ub[8+i]

def execute_plan(plan):
	'''
	Pass in a plan on an initialized ros node and it will execute the
	trajectory of that plan for a single robot.
	'''
	env_monitor = EnvironmentMonitor()
	print "Updating parameter locations..."
	env_monitor.update_plan(plan, 0)

	# print "solving laundry domain problem..."
	# solver = RobotLLSolver()
	# start = time.time()
	# viewer = OpenRAVEViewer.create_viewer(plan.env)
	# success = solver.backtrack_solve(plan, callback = None, verbose=False)
	# end = time.time()
	# print "Planning finished within {}s.".format(end - start)

	# ps = PlanSerializer()
	# ps.write_plan_to_hdf5('prototype2.hdf5', plan)
	# pd = PlanDeserializer()
	# plan = pd.read_from_hdf5('washer_manipulation_plan.hdf5')
	# viewer = OpenRAVEViewer.create_viewer(plan.env)
	import ipdb; ipdb.set_trace()


	velocites = np.ones((plan.horizon, ))*2
	ee_time = traj_retiming(plan, velocites)
	for act in plan.actions:
		act_ts = act.active_timesteps
		act.ee_retiming = ee_time[act_ts[0]:act_ts[1]]

	enforce_joint_limits(plan)
	if True or success:
		for i in range(len(plan.actions)):
			action = plan.actions[i]
			print action.name
			# traj = Trajectory()
			# traj.load_trajectory(action)

			# rospy.on_shutdown(traj.stop)
			# result = True

			# traj.start()
			# result = traj.wait()

			env_monitor.update_plan(plan, action.active_timesteps[1], params=['basket'])

			import ipdb; ipdb.set_trace()

			if action.params[2].name is 'monitor_pose' and i < len(plan.actions) - 1:
				env_monitor.update_plan(plan, action.active_timesteps[1], params=['basket'])
				if len(plan.get_failed_preds((action.active_timesteps[1], plan.actions[i+1].active_timesteps[1]), tol=1e-3)):
					success = solver._backtrack_solve(plan, callback = callback, anum=i, verbose=False)
					success = solver.traj_smoother(plan, active_ts=(action.active_timesteps[0], plan.horizon-1))
					if not success:
						import ipdb; ipdb.set_trace()

		print("Exiting - Plan Completed")

	else:
		print ("Could not solve plan.")


def move_to_ts(action, ts):
	def get_joint_positions(limb, pos, i):
		return {limb + "_s0": pos[0][i], limb + "_s1": pos[1][i], \
				limb + "_e0": pos[2][i], limb + "_e1": pos[3][i], \
				limb + "_w0": pos[4][i], limb + "_w1": pos[5][i], \
				limb + "_w2": pos[6][i]}

	baxter = None
	for param in action.params:
		if param.name == 'baxter':
			baxter = param

	if not baxter:
		raise Exception("Baxter not found for action: %s" % action.name)

	l_arm_pos = baxter.lArmPose
	l_gripper = baxter.lGripper[0]
	r_arm_pos = baxter.rArmPose
	r_gripper = baxter.rGripper[0]

	print("Getting robot state... ")
	rs = baxter_interface.RobotEnable(CHECK_VERSION)
	init_state = rs.state().enabled

	def clean_shutdown():
		print("\nExiting example...")
		if not init_state:
			print("Disabling robot...")
			rs.disable()
	rospy.on_shutdown(clean_shutdown)

	print("Enabling robot... ")
	rs.enable()
	print("Running. Ctrl-c to quit")

	left = baxter_interface.limb.Limb("left")
	right = baxter_interface.limb.Limb("right")
	grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
	grip_right = baxter_interface.Gripper('right', CHECK_VERSION)

	left_queue = Queue.Queue()
	right_queue = Queue.Queue()
	rate = rospy.Rate(10)

	if grip_left.error():
		grip_left.reset()
	if grip_right.error():
		grip_right.reset()
	if (not grip_left.calibrated() and
		grip_left.type() != 'custom'):
		grip_left.calibrate()
	if (not grip_right.calibrated() and
		grip_right.type() != 'custom'):
		grip_right.calibrate()

	def move_thread(limb, gripper, angle, grip, queue, timeout=15.0):
			"""
			Threaded joint movement allowing for simultaneous joint moves.
			"""
			try:
				limb.move_to_joint_positions(angle, timeout)
				gripper.command_position(grip)
				queue.put(None)
			except Exception, exception:
				print "Exception raised in joint movement thread"
				queue.put(traceback.format_exc())
				queue.put(exception)

	left_thread = threading.Thread(
		target=move_thread,
		args=(left,
			grip_left,
			get_joint_positions("left", l_arm_pos, ts),
			l_gripper[ts],
			left_queue
			)
	)
	right_thread = threading.Thread(
		target=move_thread,
		args=(right,
		grip_right,
		get_joint_positions("right", r_arm_pos, ts),
		r_gripper[ts],
		right_queue
		)
	)

	left_thread.daemon = True
	right_thread.daemon = True
	left_thread.start()
	right_thread.start()
	baxter_dataflow.wait_for(
		lambda: not (left_thread.is_alive() or right_thread.is_alive()),
		timeout=20.0,
		timeout_msg=("Timeout while waiting for arm move threads to finish"),
		rate=10,
	)
	left_thread.join()
	right_thread.join()
	result = left_queue.get()
	if not result is None:
		raise left_queue.get()
	result = right_queue.get()
	if not result is None:
		raise right_queue.get()
