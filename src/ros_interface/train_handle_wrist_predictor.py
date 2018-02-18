import core.util_classes.baxter_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from pma.robot_ll_solver import RobotLLSolver
from ros_interface.trajectory_controller import TrajectoryController

import baxter_interface

import roslib
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

import sys


y = 0
n = 1

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
        try:
            baxter_interface.CameraController('head_camera').close()
        except AttributeError:
            pass
        baxter_interface.CameraController('left_hand_camera').open()
        self.left = baxter_interface.limb.Limb('left')
        self.left_grip = baxter_interface.gripper.Gripper("left")
        self.left_grip.calibrate()
        self.bridge = CvBridge()
        self.left_image_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image, self.callback, queue_size=1)
        self.cur_im = None
        try:
            self.old_saved_ims = np.load('retrainingImagesHandle.npy')
            self.old_saved_labs = np.load('retrainingLabelsHandle.npy')
            print len(self.old_saved_ims)
            if len(self.old_saved_ims) != len(self.old_saved_labs):
                print 'DATA ERROR, ARRAY LENGTHS DO NOT MATCH\n'
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
        self.cur_im = data

    def save_im(self, label):
        depth_image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
        depth_array = np.array(depth_image, dtype=np.float32)
        self.saved_ims.append(depth_array)
        self.saved_labs.append(label)

    def collect_images(self):
        handle_closed_loc = [0.8, 0.75, 1.55-const.ROTOR_BASE_HEIGHT]
        handle_open_loc = [0.50, 0.80, 1.55-const.ROTOR_BASE_HEIGHT]
        terminate = 1

        raw_input('Close door and hit enter ')
        arm_pose = self.robot.openrave_body.get_ik_from_pose([handle_closed_loc[0], handle_closed_loc[1]-.05, handle_closed_loc[2]], [np.pi/2, 0, 0], "left_arm")
        if len(arm_pose):
            current_angles = arm_pose[0]
            joint_angles = get_joint_positions('left', current_angles.reshape((7,1)), 0)
            self.left_grip.open()
            self.left.move_to_joint_positions(joint_angles)
            rospy.sleep(1)
            joints_are_close = True
            actual_joint_angles = self.left.joint_angles()
            for k in joint_angles.keys():
                if np.abs(joint_angles[k]-actual_joint_angles[k]) > .1:
                    joints_are_close = False
            if joints_are_close:
                print 'Joints are close to target angles.'
            terminate = input("Is this position good? (y/n): ")
        else:
            print "Cannot place grippers over desired targets, please move Handle."
        if terminate:
            return

        try:
            height = handle_closed_loc[2]
            for k in range(5):
                y_coord = handle_closed_loc[1] - 0.05 - k*.01
                for j in range(7):
                    x_coord = handle_closed_loc[0] + (j - 3)*.01
                    print 'New coordinate: ({}, {}, {})'.format(x_coord, y_coord, height)
                    arm_pose = self.robot.openrave_body.get_ik_from_pose([x_coord, y_coord, height], [np.pi/2, 0, 0], "left_arm")
                    if len(arm_pose):
                        current_angles = closest_arm_pose(arm_pose, current_angles)
                        joint_angles = get_joint_positions('left', current_angles.reshape((7,1)), 0)
                        self.left.move_to_joint_positions(joint_angles, timeout=5.0)
                        rospy.sleep(5)
                        image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
                        image_array = np.array(image, dtype=np.float32)
                        joints_are_close = True
                        actual_joint_angles = self.left.joint_angles()
                        joint_values = [actual_joint_angles['left_s0'], actual_joint_angles['left_s1'], \
                                                    actual_joint_angles['left_e0'], actual_joint_angles['left_e1'], \
                                                    actual_joint_angles['left_w0'], actual_joint_angles['left_w1'], actual_joint_angles['left_w2']]
                        self.robot.openrave_body.set_dof({'lArmPose': joint_values})

                        end_effector_pos = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransformPose()[-3:]
                        for l in joint_angles.keys():
                            if np.abs(joint_angles[l]-actual_joint_angles[l]) > .2:
                                joints_are_close = False
                                print '{0} is not close to target'.format(l)
                        if joints_are_close:
                            self.saved_ims.append(image_array)
                            self.saved_labs.append(np.r_[handle_closed_loc, end_effector_pos])
                            print 'Saved Image'
                        else:
                            print 'Skipped image'

            for k in range(5):
                y_coord = handle_closed_loc[1] - 0.15 - k*.03
                for j in range(11):
                    x_coord = handle_closed_loc[0] + (j - k)*.015
                    print 'New coordinate: ({}, {}, {})'.format(x_coord, y_coord, height)
                    arm_pose = self.robot.openrave_body.get_ik_from_pose([x_coord, y_coord, height], [np.pi/2, 0, 0], "left_arm")
                    if len(arm_pose):
                        current_angles = closest_arm_pose(arm_pose, current_angles)
                        joint_angles = get_joint_positions('left', current_angles.reshape((7,1)), 0)
                        self.left.move_to_joint_positions(joint_angles, timeout=5.0)
                        rospy.sleep(5)
                        image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
                        image_array = np.array(image, dtype=np.float32)
                        joints_are_close = True
                        actual_joint_angles = self.left.joint_angles()
                        joint_values = [actual_joint_angles['left_s0'], actual_joint_angles['left_s1'], \
                                                    actual_joint_angles['left_e0'], actual_joint_angles['left_e1'], \
                                                    actual_joint_angles['left_w0'], actual_joint_angles['left_w1'], actual_joint_angles['left_w2']]
                        self.robot.openrave_body.set_dof({'lArmPose': joint_values})

                        end_effector_pos = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransformPose()[-3:]
                        for l in joint_angles.keys():
                            if np.abs(joint_angles[l]-actual_joint_angles[l]) > .2:
                                joints_are_close = False
                                print '{0} is not close to target'.format(l)
                        if joints_are_close:
                            self.saved_ims.append(image_array)
                            self.saved_labs.append(np.r_[handle_closed_loc, end_effector_pos])
                            print 'Saved Image'
                        else:
                            print 'Skipped image'
            terminate = input('Continue with door open? (y/n): ')
            if terminate:
                return
        # except (KeyboardInterrupt):
        #     import ipdb; ipdb.set_trace()


        # raw_input('Open door and hit enter ')
        # arm_pose = self.robot.openrave_body.get_ik_from_pose([handle_open_loc[0], handle_open_loc[1]-.15, handle_open_loc[2]], [np.pi/2, 0, 0], "left_arm")
        # if len(arm_pose):
        #     current_angles = arm_pose[0]
        #     joint_angles = get_joint_positions('left', current_angles.reshape((7,1)), 0)
        #     self.left_grip.open()
        #     self.left.move_to_joint_positions(joint_angles)
        #     rospy.sleep(1)
        #     joints_are_close = True
        #     actual_joint_angles = self.left.joint_angles()
        #     for k in joint_angles.keys():
        #         if np.abs(joint_angles[k]-actual_joint_angles[k]) > .1:
        #             joints_are_close = False
        #     if joints_are_close:
        #         print 'Joints are close to target angles.'
        #     terminate = input("Is this position good? (y/n): ")
        # else:
        #     print "Cannot place grippers over desired targets, please move Handle."
        # if terminate:
        #     return

        # try:
        #     height = handle_open_loc[2]
        #     for k in range(5):
        #         y_coord = handle_open_loc[1] - 0.1 - k*.02
        #         for j in range(11):
        #             x_coord = handle_open_loc[0] + (j - 6)*.015
        #             print 'New coordinate: ({}, {}, {})'.format(x_coord, y_coord, height)
        #             arm_pose = self.robot.openrave_body.get_ik_from_pose([x_coord, y_coord-.1, height], [np.pi/2, 0, 0], "left_arm")
        #             if len(arm_pose):
        #                 current_angles = closest_arm_pose(arm_pose, current_angles)
        #                 joint_angles = get_joint_positions('left', current_angles.reshape((7,1)), 0)
        #                 self.left.move_to_joint_positions(joint_angles, timeout=5.0)
        #                 rospy.sleep(1)
        #                 image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
        #                 image_array = np.array(image, dtype=np.float32)
        #                 joints_are_close = True
        #                 actual_joint_angles = self.left.joint_angles()
        #                 joint_values = [actual_joint_angles['left_s0'], actual_joint_angles['left_s1'], \
        #                                             actual_joint_angles['left_e0'], actual_joint_angles['left_e1'], \
        #                                             actual_joint_angles['left_w0'], actual_joint_angles['left_w1'], actual_joint_angles['left_w2']]
        #                 self.robot.openrave_body.set_dof({'lArmPose': joint_values})

        #                 end_effector_pos = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransformPose()[-3:]
        #                 for l in joint_angles.keys():
        #                     if np.abs(joint_angles[l]-actual_joint_angles[l]) > .1:
        #                         joints_are_close = False
        #                 if joints_are_close:
        #                     self.saved_ims.append(image_array)
        #                     self.saved_labs.append(np.r_[handle_open_loc, end_effector_pos])
        #                     print 'Saved Image'
        #                 else:
        #                     print 'Skipped image'
        except (KeyboardInterrupt):
            import ipdb; ipdb.set_trace()

def main(args):
    rospy.init_node('image_converter', anonymous=True)

    ic = image_converter()
    try:
        terminate = 0
        while not terminate:
            ic.collect_images()
            terminate = input("Run again? (y/n): ")
        ic.left_grip.open()
        if len(ic.old_saved_ims):
            print 'Saving data'
            np.save('retrainingImagesHandle.npy', np.concatenate([ic.old_saved_ims, np.array(ic.saved_ims)]))
            np.save('retrainingLabelsHandle.npy', np.concatenate([ic.old_saved_labs, np.array(ic.saved_labs)]))
        else:
            print 'Saving data'
            np.save('retrainingImagesHandle.npy', np.array(ic.saved_ims))
            np.save('retrainingLabelsHandle.npy', np.array(ic.saved_labs))
    except KeyboardInterrupt:
        print "Shutting down"
        ic.left_grip.open()
        import ipdb; ipdb.set_trace()
    except Exception as e:
        import ipdb; ipdb.set_trace()
        ic.left_grip.open()

if __name__ == '__main__':
    main(sys.argv)
