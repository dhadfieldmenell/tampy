import numpy as np

from keras.models import load_model

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image

import ros_interface.utils as utils


TRAINED_MODEL = 'jan7TrainedBasketSim.h5'

class BasketPredict:
    def __init__(self):
        self.bridge = CvBridge()
        self.cur_im = None
        self.image_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self.callback)
        # TODO: Add catch in case of stream failure
        self.net = load_model(TRAINED_MODEL)

    def callback(self, data):
      try:
        self.cur_im = data
      except CvBridgeError, e:
          print e

    def predict(self):
        if self.cur_im:
            im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
            im = np.array(im, dtype=np.float32)
            im[np.where(np.isnan(im))] = 0
            im[im > utils.basket_net_ul] = 0
            im[im < utils.basket_net_ll] = 0
            im = (im - utils.basket_net_mean) / utils.basket_net_std
            # TODO: Cut im to bounds
            im = cv2.resize(im, (utils.basket_im_dims[1], utils.basket_im_dims[0]))
            im = im.broadcast_to(1, utils.basket_im_dims[0], utils.basket_im_dims[1], 3)
            return self.net.predict(im)
        return np.array([np.nan, np.nan, np.nan])
