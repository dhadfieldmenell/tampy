from basket.BasketNet import BasketNet
from cloth.ClothNet import ClothNet

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
		self.updating = True

	def subscribe_to_image_topics(self):
		rospy.Subscriber('/camera/depth/image', Image, self.predict_basket)
		rospy.Subscriber('/camera/rgb/image', Image, self.predict_cloth)

	def build(self):
		self.basket_net.build()
		self.cloth_net.build()

	def predict_basket(self, msg):
		self.updating = True
		depth_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
		depth_im = np.array(depth_im, dtype=np.float32).reshape((1,480, 640, 1)) / 1000.0
		depth_im = depth_im[:, 56:265, 86:505, :]
		self.basket_pose = self.basket_net.predict(depth_im)[0]
		self.updating = False

	def predict_cloth(self, msg):
		self.updating = True
		color_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
		color_im = np.array(color_im, dtype=np.float32).reshape((1,480, 640, 3))
		color_im = color_im[:, 65:260, 98:288, :]
		self.cloth_pose = self.cloth_net.predict(color_im)[0]
		self.updating = False

	def get_basket_pose(self):
	    return self.basket_pose

	def get_cloth_pose(self):
	    return self.cloth_pose

	def update_plan(self, plan, t, params=[]):
		basket = plan.params['basket']
		cloth = plans.params['cloth']
		basket_init_pose = plan.params['basket_init_pose']
		cloth_init_pose = plan.params['cloth_init_pose']
		table = plan.params['table']
		table_pose = table.pose[:,0]

		if not params or 'basket' in params:
			basket.pose[0, t] = table_pose[0] - self.basket_pose[0] 
			basket.pose[1, t] = table_pose[1] +  self.basket_pose[1]
			basket.rotation[0,t] = self.basket_pose[2]
			basket_init_pose.pose[0, t] = table_pose[0] - self.basket_pose[0] 
			basket_init_pose.pose[1, t] = table_pose[1] +  self.basket_pose[1]
			basket_init_pose.rotation[0,t] = self.basket_pose[2]

		if not params or 'cloth' in params:
			cloth.pose[0,t] = table_pose[0] - self.cloth_pose[0]
			cloth.pose[1,t] = tale_pose[1] + self.cloth_pose[1]
			cloth.rotation[0,t] = self.coth_pose[2]
			cloth_init_pose.value[0,t] = table_pose[0] - self.cloth_pose[0]
			cloth_init_pose.value[1,t] = table_pose[1] + self.cloth_pose[1]
			cloth_init_pose.rotation[0,t] = self.coth_pose[2]
