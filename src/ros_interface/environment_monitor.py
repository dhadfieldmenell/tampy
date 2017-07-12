import rospy
from std_msgs.msg import Int32MultiArray

import cv_bridge

class EnvironmentMonitor:
	def __init__(self):
		self.basket_net = BasketNet()
		self.cloth_net = ClothNet()
		self.basket_pose = []
		self.cloth_pose = []
		self.build()
		self.subscribe_to_image_topics()

	def subscribe_to_image_topics(self):
		rospy.Subscriber('/camera/depth/image', Image, self.predict_basket)
		rospy.Subscriber('/camera/rgb/image', Image, self.predict_cloth)

	def build(self):
		self.basket_net.build()
		self.cloth_net.build()

	def predict_basket(self, msg):
		depth_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
		depth_im = np.array(depth_im, dtype=np.float32).reshape((1,-1)) / 1000.0
		self.basket_pose = self.basket_net.predict(depth_im)[0]

	def predict_cloth(self, msg):
		color_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
		color_im = np.array(dcolor_im, dtype=np.float32).reshape((1,-1)) / 1000.0
		self.cloth_pose = self.cloth_net.predict(color_im)[0]

	def get_basket_pose(self):
	    return self.basket_pose

	def get_cloth_pose(self):
	    return self.cloth_pose

	def update_plan(self, plan, t):
		basket = plan.params['basket']
		cloth = plans.params['cloth']

		basket.pose[:2, t] = self.basket_pose[:2]
		basket.rotation[0,t] = self.basket_pose[2]
		cloth.pose[:2,t] = self.cloth_pose[:2]
		cloth.rotation[0,t] = self.coth_pose[2]
