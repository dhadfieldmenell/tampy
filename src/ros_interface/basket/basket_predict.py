import numpy as np

from keras import backend as K
from keras.models import load_model

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image

import ros_interface.utils as utils


TRAINED_MODEL = 'ros_interface/basket/Feb6TrainedBasket.h5'

class BasketPredict:
    def __init__(self):
        self.bridge = CvBridge()
        self.cur_im = None
        self.last_inter = 0
        # TODO: Add catch in case of stream failure
        self.net = load_model(TRAINED_MODEL)
        self.inter_func = K.function([self.net.layers[0].input], [self.net.layers[3].output, self.net.layers[20].output])
        self.inter_inds = [2, 1]
        self.pub1 = rospy.Publisher("net1", Image, queue_size=1)
        self.pub2 = rospy.Publisher("net2", Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self.callback)
        # self.inter_image_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self.intermediary_callback)

    def callback(self, data):
        if data.header.stamp.secs > self.last_inter + 0.2:
            try:
                self.cur_im = data
            except CvBridgeError, e:
                print e

    def intermediary_callback(self, data):
        if data.header.stamp.secs > self.last_inter + 0.5:
            try:
                im = self.get_im()
                inter_preds = self.inter_func([im])
                self.pub1.publish(self.bridge.cv2_to_imgmsg(inter_preds[0][0,:,:,self.inter_inds[0]], "passthrough"))
                self.pub2.publish(self.bridge.cv2_to_imgmsg(inter_preds[1][0,:,:,self.inter_inds[1]]*10, "passthrough"))
                self.last_inter = data.header.stamp.secs
            except CvBridgeError, e:
                print e

    def get_im(self):
        im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
        im = np.array(im, dtype=np.float32)
        im[np.where(np.isnan(im))] = 0
        im[im > utils.basket_net_ul] = 0
        im[im < utils.basket_net_ll] = 0
        im = (im - utils.basket_net_mean) / utils.basket_net_std
        # TODO: Cut im to bounds
        im = cv2.resize(im, (utils.basket_im_dims[1], utils.basket_im_dims[0]))
        im = im.reshape((1, utils.basket_im_dims[0], utils.basket_im_dims[1]))
        new_im = np.zeros((1, utils.basket_im_dims[0], utils.basket_im_dims[1], 3))
        new_im[:,:,:,0] = im
        new_im[:,:,:,1] = im
        new_im[:,:,:,2] = im

        return new_im

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
            im = im.reshape((1, utils.basket_im_dims[0], utils.basket_im_dims[1]))

            new_im = np.zeros((1, utils.basket_im_dims[0], utils.basket_im_dims[1], 3))
            new_im[0,:,:,0] = im
            new_im[0,:,:,1] = im
            new_im[0,:,:,2] = im
            pred = self.net.predict(new_im)

            # The net's zero reference doesn't align exactly with the ground truth
            # The predictions however are accurate in its frame of reference
            zero_x = utils.basket_net_zero_pos[0]
            zero_y = utils.basket_net_zero_pos[1]
            zero_theta = utils.basket_net_zero_pos[2]
            pred[0, 1] *= -1 # The net flips along the y-axis
            pred[0, 0] += zero_x
            pred[0, 1] += zero_y
            pred[0, 2] *= -1
            pred[0, 2] += np.pi

            return pred.flatten()
        return np.array([np.nan, np.nan, np.nan])
        