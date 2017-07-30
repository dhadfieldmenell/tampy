from ros_interface.basket.BasketNet import BasketNet
from ros_interface.cloth.ClothNet import ClothNet

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

import cv_bridge

import numpy as np

import matplotlib.pyplot as plt

basket_net = BasketNet()
cloth_net = ClothNet()

bridge = cv_bridge.CvBridge()

last_basket_time = -1
last_cloth_time = -1

def predict_basket(msg):
	global last_basket_time
	if msg.header.stamp.secs > last_basket_time + 5:
		depth_im = bridge.imgmsg_to_cv2(msg, 'passthrough')
		depth_im = np.array(depth_im, dtype=np.float32).reshape((1,480, 640, 1)) / 1000.0
		depth_im = depth_im[:, 56:265, 86:505, :]
		np.save('basket_pose.npy', basket_net.predict(depth_im)[0])
		last_basket_time = msg.header.stamp.secs

def predict_cloth(msg):
	global last_cloth_time
	if msg.header.stamp.secs > last_cloth_time + 5:
		color_im = bridge.imgmsg_to_cv2(msg, 'passthrough')
		color_im = np.array(color_im, dtype=np.float32).reshape((1,480, 640, 3))
		color_im = color_im[:, 65:260, 98:288, :]
		np.save('cloth_pose.npy', cloth_net.predict(color_im)[0])
		last_cloth_time = msg.header.stamp.secs

print 'Running prediction script...'
rospy.init_node('env_monitor')
rospy.Subscriber('/camera/depth/image_raw', Image, predict_basket, queue_size=1)
rospy.Subscriber('/camera/rgb/image_raw', Image, predict_cloth, queue_size=1)
rospy.spin()
