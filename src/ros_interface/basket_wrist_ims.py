import core.util_classes.baxter_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.param_setup import ParamSetup

import baxter_interface

import roslib
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

import sys


def get_joint_positions(limb, pos, i):
    return {limb + "_s0": pos[0][i], limb + "_s1": pos[1][i], \
        limb + "_e0": pos[2][i], limb + "_e1": pos[3][i], \
        limb + "_w0": pos[4][i], limb + "_w1": pos[5][i], \
        limb + "_w2": pos[6][i]}

def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

class image_converter:
    def __init__(self):
        self.right = baxter_interface.limb.Limb('right')
        self.bridge = CvBridge()
        self.right_image_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, self.callback, queue_size=1)
        self.cur_im = None
        try:
            self.old_saved_ims = np.load('basketWristGreenImages2.npy')
            self.old_saved_labs = np.load('basketWristGreenLabels2.npy')
            print len(self.old_saved_ims)
        except IOError:
            self.old_saved_ims = []
            self.old_saved_labs = []
            print 'No old data found.'
        self.saved_ims = []
        self.saved_labs = []
        env = ParamSetup.setup_env()
        self.robot = ParamSetup.setup_baxter()
        self.robot.openrave_body = OpenRAVEBody(env, 'baxter', self.robot.geom)
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        init_state = rs.state().enabled

        def clean_shutdown():
            print("\nExiting...")
            if not init_state:
                print("Disabling robot...")
                rs.disable()
        rospy.on_shutdown(clean_shutdown)

        print("Enabling robot... ")
        rs.enable()
        print("Running. Ctrl-c to quit")

    def callback(self, data):
        try:
            self.cur_im = data
        except CvBridgeError, e:
            print e

    def save_im(self, label):
        depth_image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
        depth_array = np.array(depth_image, dtype=np.float32)
        self.saved_ims.append(depth_array)
        self.saved_labs.append(label)

    def collect_images(self, label):
        wrist_rot = 0
        terminate = 1
        label[0] += .75
        label[1] += .02
        handle_offset = .325
        handle_x = label[0] + np.cos(label[3]-np.pi)*handle_offset
        handle_y = label[1] - np.sin(label[3])*handle_offset
        print 'Initial coordinate: {}'.format(label)
        arm_pose = self.robot.openrave_body.get_ik_from_pose([handle_x, handle_y, label[2]+.1], [wrist_rot, np.pi/2, 0], "right_arm")
        if len(arm_pose):
            current_angles = arm_pose[0]
            joint_angles = get_joint_positions('right', current_angles.reshape((7,1)), 0)
            self.right.move_to_joint_positions(joint_angles)
            rospy.sleep(1)
            joints_are_close = True
            actual_joint_angles = self.right.joint_angles()
            for k in joint_angles.keys():
                if np.abs(joint_angles[k]-actual_joint_angles[k]) > .05:
                    joints_are_close = False
            if joints_are_close:
                print 'Joints are close to target angles.'
            terminate = input("Is this position good?")
        if terminate:
            return

        try:
            for i in range(1):
                height = label[2] + .125 #i*.025+.1
                for j in range(17):
                    x_coord = handle_x + (j - 8)*.02
                    for k in range(17):
                        y_coord = handle_y + (k - 8)*.02
                        print 'New coordinate: ({}, {}, {})'.format(x_coord, y_coord, height)
                        arm_pose = self.robot.openrave_body.get_ik_from_pose([x_coord, y_coord, height], [wrist_rot, np.pi/2, 0], "right_arm")
                        if len(arm_pose):
                            current_angles = closest_arm_pose(arm_pose, current_angles)
                            joint_angles = get_joint_positions('right', current_angles.reshape((7,1)), 0)
                            self.right.move_to_joint_positions(joint_angles, timeout=20.0)
                            rospy.sleep(5)
                            image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
                            image_array = np.array(image, dtype=np.float32)
                            joints_are_close = True
                            actual_joint_angles = self.right.joint_angles()
                            joint_values = [actual_joint_angles['right_s0'], actual_joint_angles['right_s1'], \
                                                        actual_joint_angles['right_e0'], actual_joint_angles['right_e1'], \
                                                        actual_joint_angles['right_w0'], actual_joint_angles['right_w1'], actual_joint_angles['right_w2']]
                            self.robot.openrave_body.set_dof({'rArmPose': joint_values})
                            end_effector_pos = self.robot.openrave_body.env_body.GetLink('right_gripper').GetTransformPose()[-3:]
                            for k in joint_angles.keys():
                                if np.abs(joint_angles[k] - actual_joint_angles[k]) > .05:
                                    joints_are_close = False
                            if joints_are_close:
                                self.saved_ims.append(image_array)
                                self.saved_labs.append(np.r_[np.array([label[0], label[1], label[2], label[3]]), end_effector_pos, joint_values])
                                print 'Saved Image'
                            else:
                                print 'Skipped image'
                terminate = input('Continue?')
                if terminate:
                    return
        except (KeyboardInterrupt):
            import ipdb; ipdb.set_trace()

def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
        # rospy.spin()
        terminate = 0
        while not terminate:
            try:
                x = input("x coordinate of the basket in this image: ")
                y = input("y coordinate of the basket in this image: ")
                theta = input("rotation of the basket in this images: ")
                z = .83 - const.ROTOR_BASE_HEIGHT
                ic.collect_images([x, y, z, theta])
            except:
                print 'Error in input, discarding results.'
            terminate = input("end? ")
        if len(ic.old_saved_ims):
            print 'Saving data'
            np.save('basketWristGreenImages2.npy', np.concatenate([ic.old_saved_ims, np.array(ic.saved_ims)]))
            np.save('basketWristGreenLabels2.npy', np.concatenate([ic.old_saved_labs, np.array(ic.saved_labs)]))
        else:
            print 'Saving data'
            np.save('basketWristGreenImages2.npy', np.array(ic.saved_ims))
            np.save('basketWristGreenLabels2.npy', np.array(ic.saved_labs))
    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()
        print "Shutting down"
    except:
        import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main(sys.argv)
