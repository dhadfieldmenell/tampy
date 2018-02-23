import rospy

import baxter_interface

from ros_interface.controllers import TrajectoryController
from ros_interface.rotate_control import RotateControl
from core.util_classes.plan_hdf5_serialization import PlanDeserializer

rospy.init_node('cloth_pickup_putdown_plan')

pd = PlanDeserializer()
plan = pd.read_from_hdf5('cloth_pickup_putdown.hdf5')

rotate_control = RotateControl()
trajectory_control = TrajectoryController()

baxter_interface.gripper.Gripper("left").calibrate()
baxter_interface.gripper.Gripper("right").calibrate()

for action in plan.actions:
    if action.name.startswith("rotate"):
        rotate_control.rotate_to_region(int(action.params[-1].name[-1]))
    else:
        trajectory_control.execute_plan(plan, active_ts=action.active_timesteps) 
