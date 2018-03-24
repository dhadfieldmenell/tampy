import baxter_interface
import rospy

rospy.init_node('calibration_begin_node')

rs = baxter_interface.robot_enable.RobotEnable()

left = baxter_interface.limb.Limb('left')
left.move_to_joint_positions({'left_s0':0, 'left_s1':-0.75, 'left_e0':0, 'left_e1':0, 'left_w0':0, 'left_w1':0, 'left_w2':0})
right = baxter_interface.limb.Limb('right')
right.move_to_joint_positions({'right_s0':0, 'right_s1':-0.75, 'right_e0':0, 'right_e1':0, 'right_w0':0, 'right_w1':0, 'right_w2':0})

[-0.24083498369800996, -0.4759175394414496, -0.17640779060682257, 1.7169079968407492, 0.5054466696082438, -1.1550875332777166, -0.03106311095467963]
[0.06366020269724466, -0.45405831321408247, 0.18829614171293454, 1.791306065053192, -0.6389029981542749, -1.2908448330055757, -0.13000487177328882]
