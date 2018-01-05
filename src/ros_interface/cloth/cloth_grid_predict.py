import numpy as np

from keras.models import load_model

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image

import ros_interface.cloth.grid_utils as utils


TRAINED_MODEL = 'clothGridEvalJan4.h5'

class ClothGridPredict:
    def __init__(self):
        self.bridge = CvBridge()
        self.cur_im = None
        self.image_sub = rospy.Subscriber("/zed/rgb/image_rect_color", Image, self.callback)
        self.net = load_model(TRAINED_MODEL)

    def callback(self, data):
      try:
        self.cur_im = data
      except CvBridgeError, e:
          print e

    def predict(self):
        locs = []
        if self.cur_im: 
            im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
            im = np.array(im, dtype=np.float32)
            for loc in utils.cloth_grid_coordinates:
                region = im[loc[0][0]-utils.cloth_grid_window:loc[0][0]+utils.cloth_grid_window, loc[0][1]-utils.cloth_grid_window:loc[0][1]+utils.cloth_grid_window]
                region = np.expand_dims(cv2.resize(region, ((utils.cloth_grid_input_dim, utils.cloth_grid_input_dim))), 0)
                region = (region - utils.cloth_net_mean) / utils.cloth_net_std
                prediction = self.net.predict(region)

                if prediction > 0.5:
                    ref = utils.cloth_grid_ref
                    disp = np.array(ref[0] - loc[0]) / utils.pixels_per_cm
                    locs.append(ref[1] + disp)

        return locs
