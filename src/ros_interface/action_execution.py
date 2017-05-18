import threading
import traceback
import Queue
import rospy
import baxter_dataflow
import baxter_interface
from baxter_interface import CHECK_VERSION

def execute_action(action):
	def get_joint_positions(limb, pos, i):

		return {limb + "_s0": pos[0][i], limb + "_s1": pos[1][i], \
				limb + "_e0": pos[2][i], limb + "_e1": pos[3][i], \
				limb + "_w0": pos[4][i], limb + "_w1": pos[5][i], \
				limb + "_w2": pos[6][i]}


	baxter = None
	for param in action.params:
		if param.name == 'baxter':
			baxter = param
			break

	if not baxter:
		raise Exception("Baxter not found for action: %s" % action.name)

	l_arm_pos = baxter.lArmPose
	l_gripper = baxter.lGripper[0]
	r_arm_pos = baxter.rArmPose
	r_gripper = baxter.rGripper[0]

	print ("Initialize baxter node...")
	baxter_node = rospy.init_node("baxter")
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
	rate = rospy.Rate(1000)

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

	for i in range(0, len(l_gripper)):
		left_thread = threading.Thread(
			target=move_thread,
			args=(left,
				grip_left,
				get_joint_positions("left", l_arm_pos, i),
				l_gripper[i],
				left_queue
				)
		)
		right_thread = threading.Thread(
			target=move_thread,
			args=(right,
			grip_right,
			get_joint_positions("right", r_arm_pos, i),
			r_gripper[i],
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

	left.move_to_neutral()
	right.move_to_neutral()
