from ros_interface.basket.BasketNet import BasketNet
from ros_interface.cloth.ClothNet import ClothNet

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

import cv_bridge

import numpy as np
from collections import OrderedDict
import time

def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    # param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]

class EnvironmentMonitor:
	def __init__(self):
		# self.basket_net = BasketNet()
		# self.cloth_net = ClothNet()
		self.basket_pose = []
		self.cloth_pose = []
		# self.bridge = cv_bridge.CvBridge()
		# self.build()
		# self.subscribe_to_image_topics()

	# def subscribe_to_image_topics(self):
	# 	rospy.Subscriber('/camera/depth/image', Image,  lambda msg: self.predict_basket(msg), queue_size=1)
	# 	rospy.Subscriber('/camera/rgb/image', Image, lambda msg: self.predict_cloth(msg), queue_size=1)

	# def build(self):
	# 	self.basket_net.build()
	# 	self.cloth_net.build()

	# def predict_basket(self, msg):
	# 	print 'Predicting basket'
	# 	depth_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
	# 	depth_im = np.array(depth_im, dtype=np.float32).reshape((1,480, 640, 1)) / 1000.0
	# 	depth_im = depth_im[:, 56:265, 86:505, :]
	# 	self.basket_pose = self.basket_net.predict(depth_im)[0]
	# 	raise Exception()

	# def predict_cloth(self, msg):
	# 	print 'Predicting cloth'
	# 	color_im = self.bridge.imgmsg_to_cv2(msg.data, 'passthrough')
	# 	color_im = np.array(color_im, dtype=np.float32).reshape((1,480, 640, 3))
	# 	color_im = color_im[:, 65:260, 98:288, :]
	# 	self.cloth_pose = self.cloth_net.predict(color_im)[0]
	# 	raise Exception()

	# def get_basket_pose(self):
	#     return self.basket_pose

	# def get_cloth_pose(self):
	#     return self.cloth_pose

	def update(self):
		try:
			self.basket_pose = np.load('basket_pose.npy')
			self.cloth_pose = np.load('cloth_pose.npy')
			return True
		except:
			return False

	def update_plan(self, plan, t, params=[], read_file=True):
		if read_file:
			while not self.update():
				time.sleep(5)
		basket = plan.params['basket']
		cloth = plan.params['cloth']
		basket_init_target = plan.params['basket_init_target']
		cloth_init_target = plan.params['cloth_init_target']
		table = plan.params['table']
		table_pose = table.pose[:,0]
		# import ipdb; ipdb.set_trace()

		updated_values = []

		if not params or 'basket' in params:
			attr_inds, res = OrderedDict(), OrderedDict()
			basket.pose[0, t] = table_pose[0] - self.basket_pose[0]
			basket.pose[1, t] = table_pose[1] +  self.basket_pose[1]
			# basket.rotation[0,t] = self.basket_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket.pose[:,t]), ('rotation', basket.rotation[:,t])])
			updated_values.append((res, attr_inds))

			attr_inds, res = OrderedDict(), OrderedDict()
			basket_init_target.value[0, 0] = table_pose[0] - self.basket_pose[0]
			basket_init_target.value[1, 0] = table_pose[1] +  self.basket_pose[1]
			# basket_init_pose.rotation[0,t] = 1.57self.basket_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket_init_target.value[:,0]), ('rotation', basket_init_target.rotation[:,0])])
			updated_values.append((res, attr_inds))



		if not params or 'cloth' in params:
			attr_inds, res = OrderedDict(), OrderedDict()
			cloth.pose[0,t] = table_pose[0] - self.cloth_pose[0]
			cloth.pose[1,t] = table_pose[1] + self.cloth_pose[1]
			# cloth.rotation[0,t] = self.coth_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, cloth, [('pose', cloth.pose[:,t]), ('rotation', cloth.rotation[:,t])])
			updated_values.append((res, attr_inds))

			attr_inds, res = OrderedDict(), OrderedDict()
			cloth_init_target.value[0,0] = table_pose[0] - self.cloth_pose[0]
			cloth_init_target.value[1,0] = table_pose[1] + self.cloth_pose[1]
			# cloth_init_pose.rotation[0,t] = self.coth_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, cloth_init_target, [('value', cloth_init_target.value[:,0]), ('rotation', cloth_init_target.rotation[:,0])])
			updated_values.append((res, attr_inds))
		return updated_values
