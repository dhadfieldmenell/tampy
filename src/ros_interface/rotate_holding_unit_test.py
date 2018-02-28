import rospy
import sys
import baxter_interface

from ros_interface.controllers import TrajectoryController
from ros_interface.rotate_control import RotateControl
from core.util_classes.plan_hdf5_serialization import PlanDeserializer

filename=sys.argv[1]
repetitions = 1
debug = False

if (len(sys.argv) >= 3): 
	repetitions = int(sys.argv[2])

if (len(sys.argv) >= 4):
    debug = True

rospy.init_node(filename + '_plan')

pd = PlanDeserializer()
plan = pd.read_from_hdf5(filename + ".hdf5")

rotate_control = RotateControl()
trajectory_control = TrajectoryController()

baxter_interface.gripper.Gripper("left").calibrate()
baxter_interface.gripper.Gripper("right").calibrate()

for iter in range(repetitions):
	for action in plan.actions:
	    if action.name.startswith("rotate"):
	        rotate_control.rotate_to_region(int(action.params[-1].name[-1]))
	    else:
	        trajectory_control.execute_plan(plan, active_ts=action.active_timesteps) 

	    if debug:
                raw_input("Press enter for next action....")
	print("Iteration " + str(iter) + " ended!")
