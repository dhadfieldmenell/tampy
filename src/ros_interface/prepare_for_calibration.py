import baxter_interface
import rospy

rospy.init_node('calibration_begin_node')

rs = baxter_interface.robot_enable.RobotEnable()

raw_input("Verify both robot arms are in the documented position and both grippers have been removed. Once they are hit enter. ")

left = baxter_interface.limb.Limb('left')
left.move_to_joint_positions({'left_s0':0, 'left_s1':-0.75, 'left_e0':0, 'left_e1':0, 'left_w0':0, 'left_w1':0, 'left_w2':0})
right = baxter_interface.limb.Limb('right')
right.move_to_joint_positions({'right_s0':0, 'right_s1':-0.75, 'right_e0':0, 'right_e1':0, 'right_w0':0, 'right_w1':0, 'right_w2':0})
