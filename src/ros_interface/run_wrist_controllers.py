import core.util_classes.baxter_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.param_setup import ParamSetup
from ros_interface.basket_wrist.net import BasketWristPredictor
from ros_interface.controllers import EEController

import baxter_interface

import rospy
from sensor_msgs.msg import Image

import cv2
import cv_bridge

import numpy as np

import ipdb

class WristControllerTest(object):
    def __init__(self):
        self.predictor = BasketWristPredictor()
        self.subscriber = rospy.Subscriber("/cameras/right_hand_camera/image", Image, self.predict, queue_size=1)
        self.active = False
        self.last_basket_time = -5
        self.controller = EEController()
        self.right = baxter_interface.limb.Limb('right')
        init_ee_pose = [0.75, -0.2, 1.25-const.ROTOR_BASE_HEIGHT]
        init_ee_rot = [0, np.pi/2, 0]
        self.controller.update_targets([0.75, 0.2, 1.0], [0, np.pi/2, 0], init_ee_pose, init_ee_rot)
        self.error_prediction = np.zeros((3,))
        self.baxter = ParamSetup.setup_baxter()

        self.bridge = cv_bridge.CvBridge()

        env = ParamSetup.setup_env()
        self.baxter.openrave_body = OpenRAVEBody(env, 'baxter', self.baxter.geom)
        rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        init_state = rs.state().enabled

        rs.enable()

    def predict(self, msg):
        if msg.header.stamp.secs > self.last_basket_time + 2.5 and self.active:
            image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            image_array = np.array(image, dtype=np.float32)
            with self.predictor.graph.as_default():
                self.error_prediction = self.predictor.predict(image_array[:,:,:3])
            self.last_basket_time = msg.header.stamp.secs

    def initialize(self):
        self.controller.move_to_targets(limbs=['right'])
        self.active = True

    def move_to_prediction(self):
        actual_joint_angles = self.right.joint_angles()
        joint_values = [actual_joint_angles['right_s0'], actual_joint_angles['right_s1'], \
                        actual_joint_angles['right_e0'], actual_joint_angles['right_e1'], \
                        actual_joint_angles['right_w0'], actual_joint_angles['right_w1'], actual_joint_angles['right_w2']]
        self.baxter.openrave_body.set_dof({'rArmPose': joint_values})
        end_effector_pos = self.baxter.openrave_body.env_body.GetLink('right_gripper').GetTransformPose()[-3:]
        error_pred = self.error_prediction.copy()

        target_pos = (end_effector_pos + np.r_[error_pred[0], error_pred[1], 0])
        target_rot = [error_pred[2], np.pi/2, 0]
        print 'Moving to :', target_pos
        self.controller.update_targets([], [], target_pos, target_rot)
        self.controller.move_to_targets(limbs=['right'])


if __name__ == '__main__':
    rospy.init_node('wrist_controller', anonymous=True)
    wc = WristControllerTest()
    wc.initialize()
    ipdb.set_trace()
    wc.move_to_prediction()
    ipdb.set_trace()
