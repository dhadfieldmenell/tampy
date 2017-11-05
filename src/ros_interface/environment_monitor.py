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
		self.net_dict = {}
		self.state_dict = {}
		self.add_predictor('cloth_pose', ClothNet, 3)
		self.add_predictor('basket_pose', BasketNet, 3)
		# self.add_predictor('basket_wrist_error', BasketWristNet, 3)
		# self.add_predictor('cloth_wrist_error', ClothWristNet, 2)
		# self.add_predictor('handle_wrist_error', HandleWristNet, 1)
		self.add_subscriber('/zed/depth/depth_registered', Image, self.predict_basket)
		self.add_subscriber('/camera/rgb/image_raw', Image, predict_cloth)
		# self.add_subscriber('/cameras/right_hand_camera/image', Image, predict_right_error)
		# self.add_subscriber('/cameras/left_hand_camera/image', Image, predict_left_error)

		self.bridge = cv_bridge.CvBridge()

		self.last_basket_time = -1
		self.last_cloth_time = -1

	def add_predictor(self, name, net, dim):
		self.net_dict[name] = net()
		self.state_dict[name] = np.zeros((dim,))

	def add_subscriber(self, topic, msg_type, callback):
		rospy.Subscriber(topic, msg_type, callback, queue_size=1)

	def predict_basket(self, msg):
		if msg.header.stamp.secs > self.last_basket_time + 5:
			depth_im = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
			depth_im = np.array(depth_im, dtype=np.float32).reshape((1,480, 640, 1)) / 1000.0
			depth_im = depth_im[:, 56:265, 86:505, :]
			self.state_dict['basket_pose'] = basket_net.predict(depth_im)[0]
			self.last_basket_time = msg.header.stamp.secs

	def predict_cloth(self, msg):
		if msg.header.stamp.secs > self.last_cloth_time + 5:
			color_im = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
			color_im = np.array(color_im, dtype=np.float32).reshape((1,480, 640, 3))
			color_im = color_im[:, 65:260, 98:288, :]
			self.state_dict['cloth_pose'] = cloth_net.predict(color_im)[0]
			self.last_cloth_time = msg.header.stamp.secs


	def update_plan(self, plan, t, params=[], read_file=True):
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
			print basket.pose[:, t]
			basket.pose[0, t] = table_pose[0] - self.state_dict['basket_pose'][0]
			basket.pose[1, t] = table_pose[1] +  self.state_dict['basket_pose'][1]
			# basket.rotation[0,t] = self.state_dict['basket_pose'][2]
			add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket.pose[:,t]), ('rotation', basket.rotation[:,t])])
			updated_values.append((res, attr_inds))

			attr_inds, res = OrderedDict(), OrderedDict()
			basket_init_target.value[0, 0] = table_pose[0] - self.state_dict['basket_pose'][0]
			basket_init_target.value[1, 0] = table_pose[1] +  self.state_dict['basket_pose'][1]
			# basket_init_pose.rotation[0,t] = 1.57self.state_dict['basket_pose'][2]
			add_to_attr_inds_and_res(t, attr_inds, res, basket_init_target, [('value', basket_init_target.value[:,0]), ('rotation', basket_init_target.rotation[:,0])])
			updated_values.append((res, attr_inds))

		if not params or 'cloth' in params:
			attr_inds, res = OrderedDict(), OrderedDict()
			cloth.pose[0,t] = table_pose[0] - self.state_dict['cloth_pose'][0]
			cloth.pose[1,t] = table_pose[1] + self.state_dict['cloth_pose'][1]
			# cloth.rotation[0,t] = self.coth_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, cloth, [('pose', cloth.pose[:,t]), ('rotation', cloth.rotation[:,t])])
			updated_values.append((res, attr_inds))

			attr_inds, res = OrderedDict(), OrderedDict()
			cloth_init_target.value[0,0] = table_pose[0] - self.state_dict['cloth_pose'][0]
			cloth_init_target.value[1,0] = table_pose[1] + self.state_dict['cloth_pose'][1]
			# cloth_init_pose.rotation[0,t] = self.coth_pose[2]
			add_to_attr_inds_and_res(t, attr_inds, res, cloth_init_target, [('value', cloth_init_target.value[:,0]), ('rotation', cloth_init_target.rotation[:,0])])
			updated_values.append((res, attr_inds))
		return updated_values
