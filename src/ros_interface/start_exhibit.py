import baxter_interface
import rospy

from ros_interface.laundry_env_monitor import LaundryEnvironmentMonitor

rospy.init_node('laundry_exeuction_node')

rs = baxter_interface.robot_enable.RobotEnable()
rs.enable()

left = baxter_interface.limb.Limb('left')
left.move_to_joint_positions({'left_s0':-0.75, 'left_s1':-0.75, 'left_e0':0, 'left_e1':0, 'left_w0':0, 'left_w1':0, 'left_w2':0})
right = baxter_interface.limb.Limb('right')
right.move_to_joint_positions({'right_s0':-0.75, 'right_s1':-0.75, 'right_e0':0, 'right_e1':0, 'right_w0':0, 'right_w1':0, 'right_w2':0})

lem = LaundryEnvironmentMonitor()
lem.run_baxter()
