import rospy
import sys
import baxter_interface

from ros_interface.controllers import TrajectoryController
from ros_interface.rotate_control import RotateControl
from core.util_classes.plan_hdf5_serialization import PlanDeserializer

filename=sys.argv[1]
if (len(sys.argv) == 3) 
	repetitions = sys.argv[2]

rospy.init_node(filename + ' plan')

pd = PlanDeserializer()
plan = pd.read_from_hdf5(filename)

rotate_control = RotateControl()
trajectory_control = TrajectoryController()

baxter_interface.gripper.Gripper("left").calibrate()
baxter_interface.gripper.Gripper("right").calibrate()

for iter in reptitions:
	for action in plan.actions:
	    if action.name.startswith("rotate"):
	        rotate_control.rotate_to_region(int(action.params[-1].name[-1]))
	    else:
	        trajectory_control.execute_plan(plan, active_ts=action.active_timesteps) 
	    raw_input("Press space for next action....")
	print("Iteration " + iter + " ended!")