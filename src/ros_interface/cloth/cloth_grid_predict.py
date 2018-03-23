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


# TRAINED_MODEL = 'ros_interface/cloth/clothGridEvalJan4.h5'
# TRAINED_MODEL = 'ros_interface/cloth/feb7TrainedClothGrid.h5'
TRAINED_MODEL = 'ros_interface/cloth/Feb24TrainedClothGrid.h5'

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
        self.grip_im_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image, self.grip_callback)
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
                pixel_val_x = utils.cloth_grid_ref[0][0] + int(utils.pixels_per_cm*(utils.cloth_grid_ref[1][0] - 100*loc[0][0]))
                pixel_val_y = utils.cloth_grid_ref[0][1] + int(utils.pixels_per_cm*(utils.cloth_grid_ref[1][1] - 100*loc[0][1]))

                region = im[pixel_val_x-utils.cloth_grid_window:pixel_val_x+utils.cloth_grid_window, pixel_val_y-utils.cloth_grid_window:pixel_val_y+utils.cloth_grid_window]
                region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                prediction = self.net.predict(region)

                if prediction > 0.95:
                    locs.append(loc)

        return locs

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

                if prediction > 0.95:
                    locs.append(loc[1])
                    break

        return locs

    def predict_wrist_center(self):
        prediction = False
        if self.cur_grip_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_grip_im, 'passthrough')
            im = np.array(im[:,:,:3], dtype=np.float32)
            region = np.expand_dims(cv2.resize(im[90:110, 160:180, :3], ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
            prediction = self.net.predict(region) > 0.9

        return prediction
        