# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

import tensorflow as tf

with tf.device("/cpu:0"):
    from keras import backend as K
    from keras.models import load_model

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image

import ros_interface.utils as utils

box_color = [0, 0, 255]
box_width = 5
zed_threshold = 0.8
wrist_threshold = 0.6

TRAINED_MODEL = 'ros_interface/cloth/April30thTrained.h5'

# def get_session():
#     with tf.device("/cpu:0"):
#         return tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

# K.tensorflow_backend.set_session(get_session())

class ClothGridPredict:
    def __init__(self):
        self.bridge = CvBridge()
        self.cur_im = None
        self.cur_grip_im = None
        self.image_sub = rospy.Subscriber("/zed/rgb/image_rect_color", Image, self.callback)
        self.grip_im_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, self.grip_callback)
        with tf.device("/cpu:0"):
            self.net = load_model(TRAINED_MODEL)

    def callback(self, data):
      try:
        self.cur_im = data
      except CvBridgeError, e:
          print e

    def grip_callback(self, data):
      try:
        self.cur_grip_im = data
      except CvBridgeError, e:
          print e

    # def predict(self):
    #     locs = []
    #     if self.cur_im: 
    #         im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
    #         im = np.array(im, dtype=np.float32)
    #         for loc in utils.cloth_grid_coordinates:
    #             region = im[loc[0][0]-utils.cloth_grid_window:loc[0][0]+utils.cloth_grid_window, loc[0][1]-utils.cloth_grid_window:loc[0][1]+utils.cloth_grid_window]
    #             region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
    #             region = (region - utils.cloth_net_mean) / utils.cloth_net_std
    #             prediction = self.net.predict(region)

    #             if prediction > 0.9:
    #                 ref = utils.cloth_grid_ref
    #                 disp = np.array(ref[0] - loc[0]) / utils.pixels_per_cm
    #                 locs.append(((ref[1] + disp) / 100.0, loc[1]))

    #     return locs

    def predict(self):
        locs = []
        if self.cur_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
            im = np.array(im, dtype=np.float32)
            for loc in utils.cloth_grid_coordinates:
                cm_offset = [utils.cloth_grid_ref[1][0] - 100*loc[0][0], utils.cloth_grid_ref[1][1] - 100*loc[0][1]]
                im_offset = utils.cloth_grid_rot_mat.dot(cm_offset)*utils.pixels_per_cm
                pixel_val_x = utils.cloth_grid_ref[0][0] + int(im_offset[0])
                pixel_val_y = utils.cloth_grid_ref[0][1] + int(im_offset[1])

                region = im[pixel_val_x-utils.cloth_grid_window:pixel_val_x+utils.cloth_grid_window, pixel_val_y-utils.cloth_grid_window:pixel_val_y+utils.cloth_grid_window]
                region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                prediction = self.net.predict(region)

                if prediction > zed_threshold:
                    locs.append(loc)

        return locs

    def test_predict(self):
        import matplotlib.pyplot as plt

        if self.cur_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
            im = np.array(im, dtype=np.float64)

            bounds = [[[260, 1080], [40, 300]], [[260, 520], [40, 680]], [[800, 1080], [40, 600]], [[260, 1080], [580, 680]]]

            for ((lower_y, upper_y), (lower_x, upper_x)) in bounds:
                for y in range(lower_y, upper_y, utils.cloth_grid_window*2):
                    for x in range(lower_x, upper_x, utils.cloth_grid_window*2):
                        region = im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y-utils.cloth_grid_window:y+utils.cloth_grid_window]
                        region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                        region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                        prediction = self.net.predict(region)
                        if prediction > zed_threshold:
                            im[x-utils.cloth_grid_window-box_width:x-utils.cloth_grid_window, y-utils.cloth_grid_window:y+utils.cloth_grid_window] = box_color
                            im[x+utils.cloth_grid_window:x+utils.cloth_grid_window+box_width, y-utils.cloth_grid_window:y+utils.cloth_grid_window] = box_color
                            im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y-utils.cloth_grid_window-box_width:y-utils.cloth_grid_window] = box_color
                            im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y+utils.cloth_grid_window:y+utils.cloth_grid_window+box_width] = box_color

            plt.imshow(im)
            plt.show()

    def test_wrist_predict(self):
        import matplotlib.pyplot as plt

        if self.cur_grip_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_grip_im, 'passthrough')
            im = np.array(im, dtype=np.float64)[:,:,:3]

            bounds = [[[30, 290], [30, 170]]]

            for ((lower_y, upper_y), (lower_x, upper_x)) in bounds:
                for y in range(lower_y, upper_y, utils.cloth_grid_window*2):
                    for x in range(lower_x, upper_x, utils.cloth_grid_window*2):
                        region = im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y-utils.cloth_grid_window:y+utils.cloth_grid_window]
                        region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                        region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                        prediction = self.net.predict(region)
                        if prediction > wrist_threshold:
                            im[x-utils.cloth_grid_window-box_width:x-utils.cloth_grid_window, y-utils.cloth_grid_window:y+utils.cloth_grid_window] = box_color
                            im[x+utils.cloth_grid_window:x+utils.cloth_grid_window+box_width, y-utils.cloth_grid_window:y+utils.cloth_grid_window] = box_color
                            im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y-utils.cloth_grid_window-box_width:y-utils.cloth_grid_window] = box_color
                            im[x-utils.cloth_grid_window:x+utils.cloth_grid_window, y+utils.cloth_grid_window:y+utils.cloth_grid_window+box_width] = box_color

            plt.imshow(im)
            plt.show()

    def predict_washer(self):
        locs = []
        if self.cur_grip_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_grip_im, 'passthrough')
            im = np.array(im[:,:,:3], dtype=np.float32)
            for loc in utils.washer_im_locs:
                region = im[loc[0][0]-utils.cloth_grid_window:loc[0][0]+utils.cloth_grid_window, loc[0][1]-utils.cloth_grid_window:loc[0][1]+utils.cloth_grid_window]
                region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                prediction = self.net.predict(region)

                if prediction > wrist_threshold:
                    locs.append(loc[1])
                    break

        return locs

    def predict_wrist_center(self, offset=[0,0], threshold=wrist_threshold):
        prediction = False
        if self.cur_grip_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_grip_im, 'passthrough')
            im = np.array(im[:,:,:3], dtype=np.float32)
            x1, x2 = offset[0]+85, offset[0]+115
            y1, y2 = offset[1]+185, offset[1]+215
            region = np.expand_dims(cv2.resize(im[x1:x2, y1:y2, :3], ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
            region = (region - utils.cloth_net_mean) / utils.cloth_net_std
            prediction = self.net.predict(region) >= threshold

        return prediction
        