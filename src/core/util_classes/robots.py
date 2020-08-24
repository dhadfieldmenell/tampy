import core.util_classes.common_constants as const

if const.USE_OPENRAVE:
    from openravepy import IkParameterizationType, databases
else:
    import pybullet as p
    from baxter_gym.util_classes.ik_controller import *

import numpy as np

try:
    import baxter_gym
except:
    pass


class Robot(object):
    """
    Base class of every robot parameter
    """
    def __init__(self, shape):
        self.shape = shape
        self.file_type = 'urdf'

    def setup(self, robot):
        return


class NAMO(Robot):
    def __init__(self):
        self._type = "robot"
        self.file_type = 'mjcf'
        self.radius = 0.3
        self.shape = baxter_gym.__path__[0]+'/robot_info/lidar_namo.xml'
        self.dof_map = {'xpos': 0, 'ypos': 1, 'robot_theta': 2, 'left_grip': 6, 'right_grip': 4}


class PR2(Robot):
    """
    Defines geometry used in the PR2 domain.
    """
    def __init__(self):
        self._type = "pr2"
        pr2_shape = "../models/pr2/pr2.zae"
        # down to 30 links from 45
        self.col_links = set(['base_link', 'torso_lift_link', 'l_shoulder_pan_link',
                              'l_shoulder_lift_link', 'l_upper_arm_roll_link', 'l_upper_arm_link',
                              'l_elbow_flex_link', 'l_forearm_roll_link', 'l_forearm_link',
                              'l_wrist_flex_link', 'l_wrist_roll_link', 'l_gripper_palm_link',
                              'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_link',
                              'l_gripper_r_finger_link', 'l_gripper_r_finger_tip_link', 'r_shoulder_pan_link',
                              'r_shoulder_lift_link', 'r_upper_arm_roll_link', 'r_upper_arm_link',
                              'r_elbow_flex_link', 'r_forearm_roll_link', 'r_forearm_link', 'r_wrist_flex_link',
                              'r_wrist_roll_link', 'r_gripper_palm_link', 'r_gripper_l_finger_link',
                              'r_gripper_l_finger_tip_link', 'r_gripper_r_finger_link',
                              'r_gripper_r_finger_tip_link'])
        self.dof_map = {"backHeight": [12], "lArmPose": list(range(15,22)), "lGripper": [22], "rArmPose": list(range(27,34)), "rGripper":[34]}
        super(PR2, self).__init__(pr2_shape)

    def setup(self, robot):
        """
        Nothing to setup for pr2
        """
        return


class Baxter(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        self._type = "baxter"
        baxter_shape = "../models/baxter/baxter.zae"
        # self.col_links = set(["torso", "pedestal", "head", "sonar_ring", "screen", "collision_head_link_1",
        #                       "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
        #                       "right_upper_elbow", "right_upper_elbow_visual", "right_lower_elbow",
        #                       "right_upper_forearm", "right_upper_forearm_visual", "right_lower_forearm",
        #                       "right_wrist", "right_hand", "right_gripper_base", "right_gripper",
        #                       "right_gripper_l_finger", "right_gripper_r_finger", "right_gripper_l_finger_tip",
        #                       "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
        #                       "left_upper_elbow", "left_upper_elbow_visual", "left_lower_elbow",
        #                       "left_upper_forearm", "left_upper_forearm_visual", "left_lower_forearm",
        #                       "left_wrist", "left_hand", "left_gripper_base", "left_gripper",
        #                       "left_gripper_l_finger", "left_gripper_r_finger", "left_gripper_l_finger_tip",
        #                       "left_gripper_r_finger_tip"])
        self.col_links = set(["torso", "head", "sonar_ring", "screen", "collision_head_link_1",
                              "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
                              "right_upper_elbow", "right_upper_elbow_visual", "right_lower_elbow",
                              "right_upper_forearm", "right_upper_forearm_visual", "right_lower_forearm",
                              "right_wrist", "right_hand", "right_gripper_base", "right_gripper",
                              "right_gripper_l_finger", "right_gripper_r_finger", "right_gripper_l_finger_tip",
                              "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
                              "left_upper_elbow", "left_upper_elbow_visual", "left_lower_elbow",
                              "left_upper_forearm", "left_upper_forearm_visual", "left_lower_forearm",
                              "left_wrist", "left_hand", "left_gripper_base", "left_gripper",
                              "left_gripper_l_finger", "left_gripper_r_finger", "left_gripper_l_finger_tip",
                              "left_gripper_r_finger_tip"])
        if const.USE_OPENRAVE:
            self.dof_map = {"lArmPose": list(range(2,9)), "lGripper": [9], "rArmPose": list(range(10,17)), "rGripper":[17]}
        else:
            self.dof_map = {"lArmPose": list(range(31,39)), "rArmPose": list(range(13,21))}
        super(Baxter, self).__init__(baxter_shape)

    def setup(self, robot):
        """
        Need to setup iksolver for baxter
        """
        if const.USE_OPENRAVE:
            iktype = IkParameterizationType.Transform6D
            ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, IkParameterizationType.Transform6D, True)
            if not ikmodel.load():
                print 'Something went wrong when loading ikmodel'
            #   ikmodel.autogenerate()
            right_manip = robot.GetManipulator('right_arm')
            ikmodel.manip = right_manip
            right_manip.SetIkSolver(ikmodel.iksolver)

            ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, IkParameterizationType.Transform6D, True)
            if not ikmodel.load():
              print 'Something went wrong'
            left_manip = robot.GetManipulator('left_arm')
            ikmodel.manip = left_manip
            left_manip.SetIkSolver(ikmodel.iksolver)
        else:
            self.ik_solver = BaxterIKController(lambda: np.zeros(14))
            self.col_links = set([self.ik_solver.name2id(name) for name in self.col_links])


class HSR(Robot):
    """
    Defines geometry used in the HSR domain.
    """
    def __init__(self):
        self._type = "hsr"
        shape = "../models/hsr/hsrb4s.xml"
        self.col_links = set([u'arm_lift_link', u'arm_flex_link', u'arm_roll_link', 
                              u'wrist_flex_link', u'wrist_roll_link', u'hand_palm_link', 
                              u'hand_l_proximal_link', u'hand_l_spring_proximal_link', 
                              u'hand_l_mimic_distal_link', u'hand_l_distal_link', 
                              u'hand_motor_dummy_link', u'hand_r_proximal_link', 
                              u'hand_r_spring_proximal_link', u'hand_r_mimic_distal_link', 
                              u'hand_r_distal_link', u'base_roll_link', 
                              u'base_l_drive_wheel_link', u'base_l_passive_wheel_z_link', 
                              u'base_r_drive_wheel_link', u'base_r_passive_wheel_z_link', 
                              u'torso_lift_link', u'head_pan_link', u'head_tilt_link'])
        self.dof_map = {'arm': [0, 1, 2, 3, 4], 'gripper': [6]}
        super(HSR, self).__init__(shape)

    def setup(self, robot):
        """
        Need to setup iksolver for baxter
        """
        if const.USE_OPENRAVE:
            iktype = IkParameterizationType.Translation3D
            ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype, True)
            if not ikmodel.load():
                print 'Something went wrong when loading ikmodel'
                ikmodel.autogenerate()
            manip = robot.GetManipulator('arm')
            ikmodel.manip = manip
            manip.SetIkSolver(ikmodel.iksolver)
        else:
            self.ik_solver = HSRIKController(lambda: np.zeros(14))
            self.col_links = set([self.ik_solver.name2id(name) for name in self.col_links])


class Washer(Robot):
    """
        Object stores all the information to for a Washer model
    """

    def __init__(self, mockup = True, topload = False):
        self._type = "washer"
        if mockup:
            self.shape = "../models/items/washer_mock_up/washer_col_2.xml"
            """
            to variate the depth of the mockup washer, simply change the y-cord
            of the washer_bottom body in washer.xml
            """
            if topload:
                self.good_pos = np.array([0.416, 1.509, 0.825])
                self.good_rot = np.array([np.pi, 0, np.pi/2])
            else:
                self.good_pos = np.array([0.505, 1.161, 1.498])
                self.good_rot = np.array([np.pi, 0, 0])
        else:
            self.shape = "../models/items/washer.xml"
            self.up_right_rot = [1.57, 0, 0]
            self.good_pos = np.array([0.5, 0.8, 0])
            self.good_rot = np.array([np.pi/2, 0, 0])
        self.dof_map = {"door": [0]}

        self.col_links = set(["external_1", "external_2", "external_3", "external_4", "back", 
                              "corner_1", "corner_2", "corner_3", "corner_4",
                              "strip_1", "strip_2", "strip_3", "strip_4", "washer_door",
                              "washer_handle", "barrel_1", "barrel_2",
                              "barrel_3", "barrel_4", "barrel_5",
                              "barrel_6", "barrel_7", "barrel_8",
                              "barrel_9", "barrel_10", "barrel_11",
                              "barrel_12", "barrel_13", "barrel_14",
                              "barrel_15", "barrel_16", "barrel_17",
                              "barrel_18", "barrel_19", "barrel_20",
                              "barrel_21", "barrel_22", "barrel_23",
                              "barrel_24", "barrel_25", "barrel_26",
                              "barrel_27", "barrel_28", "barrel_29",
                              "barrel_30", "barrel_back", "base"])

    def setup(self, robot):
        pass
