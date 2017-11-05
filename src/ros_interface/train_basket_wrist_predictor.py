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
        baxter_interface.CameraController('right_hand_camera').open()
        self.left = baxter_interface.limb.Limb('left')
        self.right = baxter_interface.limb.Limb('right')
        self.right.set_joint_position_speed(0.5)
        self.left.set_joint_position_speed(0.5)
        self.left_grip = baxter_interface.gripper.Gripper("left")
        self.left_grip.calibrate()
        self.bridge = CvBridge()
        self.right_image_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, self.callback, queue_size=1)
        self.cur_im = None
        self.solver = RobotLLSolver()
        self.traj_control = TrajectoryController()
        # pd = PlanDeserializer()
        # self.plan = pd.read_from_hdf5('move_plan.hdf5')
        try:
            self.old_saved_ims = np.load('retrainingImagesBasket.npy')
            self.old_saved_labs = np.load('retrainingLabelsBasket.npy')
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
            import ipdb; ipdb.set_trace()
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
        # label = [0.7, -0.2, 1.05-const.ROTOR_BASE_HEIGHT, np.pi/2] # Centered on table  & neutral rotation
        # label = [0.825, -0.175, 1.05-const.ROTOR_BASE_HEIGHT, np.pi/2] # Centered on table  & neutral rotation
        # label = [0.85, -0.125, 1.05-const.ROTOR_BASE_HEIGHT, np.pi/2] # Centered on table  & neutral rotation
        label = [0.7, -0.125, 1.05-const.ROTOR_BASE_HEIGHT, np.pi/2] # Centered on table  & neutral rotation
        # self.plan.params['basket'].pose[:,0] = label[:3]
        # self.plan.params['basket'].rotation[0,:] = label[3]
        # self.plan.params['table'].pose[2,:] -= const.ROTOR_BASE_HEIGHT
        # self.plan.params['init_target'].pose[:,0] = self.plan.params['basket'].pose[:,0]
        # self.plan.params['init_target'].rotation[0,0] = self.plan.params['basket'].rotation[0,0]
        # self.plan.params['end_target'].pose[:,0] = self.plan.params['basket'].pose[:,0]
        # self.plan.params['end_target'].rotation[0,0] = self.plan.params['basket'].rotation[0,0]
        terminate = 1
        handle_offset = .325
        handle_x = label[0] #label[0] + np.cos(label[3]-np.pi)*handle_offset
        handle_y = label[1] - handle_offset #label[1] - np.sin(label[3])*handle_offset
        self.left_grip.open()
        # print "Please center Baxter's left gripper onto the left handle..."
        # raw_input("Type 'c' and hit enter when ready to continue... ")
        # self.left_grip.close()
        arm_pose = self.robot.openrave_body.get_ik_from_pose([handle_x, handle_y, label[2]+.05], [0, np.pi/2, 0], "right_arm")
        left_arm_pose = self.robot.openrave_body.get_ik_from_pose([label[0], label[1] + handle_offset, label[2]], [0, np.pi/2, 0], "left_arm")
        if len(arm_pose) and len(left_arm_pose):
            current_angles = arm_pose[1]
            joint_angles = get_joint_positions('right', current_angles.reshape((7,1)), 0)
            self.right.move_to_joint_positions(joint_angles)
            self.left.move_to_joint_positions(get_joint_positions('left', left_arm_pose[1].reshape((7,1)), 0), timeout=5)
            rospy.sleep(1)
            joints_are_close = True
            actual_joint_angles = self.right.joint_angles()
            for k in joint_angles.keys():
                if np.abs(joint_angles[k]-actual_joint_angles[k]) > .1:
                    joints_are_close = False
            if joints_are_close:
                print 'Joints are close to target angles.'
            terminate = input("Is this position good? (y/n): ")
        else:
            print "Cannot place grippers over desired targets."
            terminate = 1
        if terminate:
            self.left_grip.open()
            return

        terminate = 1
        # self.left_grip.open()
        left_arm_pose = self.robot.openrave_body.get_ik_from_pose([label[0] - np.cos(label[3]-np.pi)*handle_offset, label[1] + np.sin(label[3])*handle_offset, label[2]], [0, np.pi/2, 0], "left_arm")
        if len(left_arm_pose):
            self.left.move_to_joint_positions(get_joint_positions('left', left_arm_pose[0].reshape((7,1)), 0))
            terminate = input("Did the left arm grasp the basket without moving it? (y/n): ")
        if terminate:
            left_arm_pose = self.robot.openrave_body.get_ik_from_pose([label[0] - np.cos(label[3]-np.pi)*handle_offset, label[1] + np.sin(label[3])*handle_offset, label[2]+.15], [0, np.pi/2, 0], "left_arm")
            self.left_grip.open()
            self.left.move_to_joint_positions(get_joint_positions('left', left_arm_pose[0].reshape((7,1)), 0))
            return
        self.left_grip.close()
        left_ref_pose = self.left.joint_angles().copy()

        try:
            height = label[2] + .2
            for j in range(15):
                x_coord = handle_x + -(j - 8)*.01
                for k in range(31):
                    y_coord = handle_y + (k - 16)*.01
                    print 'New coordinate: ({}, {}, {})'.format(x_coord, y_coord, height)
                    angles = np.random.uniform(-np.pi/3, np.pi/3, (5,))
                    angles[0] = 0
                    for angle in angles:
                        arm_pose = self.robot.openrave_body.get_ik_from_pose([x_coord, y_coord, height], [angle, np.pi/2, 0], "right_arm")
                        if len(arm_pose):
                            current_angles = closest_arm_pose(arm_pose, current_angles)
                            joint_angles = get_joint_positions('right', current_angles.reshape((7,1)), 0)
                            if j or k or True: # IN PROGRESS
                                self.right.move_to_joint_positions(joint_angles, timeout=5.0)
                            # else:
                            #     actual_joint_angles = self.right.joint_angles()
                            #     actual_joint_angles = [actual_joint_angles['right_s0'], actual_joint_angles['right_s1'], \
                            #                            actual_joint_angles['right_e0'], actual_joint_angles['right_e1'], \
                            #                            actual_joint_angles['right_w0'], actual_joint_angles['right_w1'], \
                            #                            actual_joint_angles['right_w2']]
                            #     self.plan.params['robot_init_pose'].rArmPose[:,0] = actual_joint_angles
                            #     self.plan.params['baxter'].rArmPose[:,0] = actual_joint_angles
                            #     self.plan.params['robot_end_pose'].rArmPose[:, 0] = current_angles
                            #     self.solver.solve(self.plan)
                            #     self.traj_control.execute_plan(self.plan, limbs=['right'])
                            rospy.sleep(1)
                            image = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
                            image_array = np.array(image, dtype=np.float32)
                            joints_are_close = True
                            actual_joint_angles = self.right.joint_angles()
                            joint_values = [actual_joint_angles['right_s0'], actual_joint_angles['right_s1'], \
                                            actual_joint_angles['right_e0'], actual_joint_angles['right_e1'], \
                                            actual_joint_angles['right_w0'], actual_joint_angles['right_w1'], actual_joint_angles['right_w2']]
                            self.robot.openrave_body.set_dof({'rArmPose': joint_values})

                            left_actual_pos = self.left.joint_angles();
                            for l in left_actual_pos.keys():
                                if abs(left_actual_pos[l] - left_ref_pose[l]) > 1e-2:
                                    self.left_grip.open()
                                    print 'Left arm has moved, possible basket is out of position. Saving & exiting.'
                                    return

                            end_effector_pos = self.robot.openrave_body.env_body.GetLink('right_gripper').GetTransformPose()[-3:]
                            for k in joint_angles.keys():
                                if np.abs(joint_angles[k]-actual_joint_angles[k]) > .25:
                                    joints_are_close = False
                            if joints_are_close:
                                self.saved_ims.append(image_array)
                                self.saved_labs.append(np.r_[label, end_effector_pos, angle])
                                print 'Saved Image'
                            else:
                                print 'Skipped image'
                terminate = input('Up to distance {0}. Continue? (y/n): '.format(x_coord))
                if terminate:
                    return
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
            np.save('retrainingImagesBasket.npy', np.concatenate([ic.old_saved_ims, np.array(ic.saved_ims)]))
            np.save('retrainingLabelsBasket.npy', np.concatenate([ic.old_saved_labs, np.array(ic.saved_labs)]))
        else:
            print 'Saving data'
            np.save('retrainingImagesBasket.npy', np.array(ic.saved_ims))
            np.save('retrainingLabelsBasket.npy', np.array(ic.saved_labs))
    except KeyboardInterrupt:
        print "Shutting down"
        ic.left_grip.open()
        import ipdb; ipdb.set_trace()
    except Exception as e:
        import ipdb; ipdb.set_trace()
        ic.left_grip.open()

if __name__ == '__main__':
    main(sys.argv)
