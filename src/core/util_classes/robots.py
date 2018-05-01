from openravepy import IkParameterizationType, databases
import numpy as np
class Robot(object):
    """
    Base class of every robot parameter
    """
    def __init__(self, shape):
        self.shape = shape

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
        self.dof_map = {"lArmPose": list(range(2,9)), "lGripper": [9], "rArmPose": list(range(10,17)), "rGripper":[17]}
        super(Baxter, self).__init__(baxter_shape)

    def setup(self, robot):
        """
        Need to setup iksolver for baxter
        """
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

class Washer(Robot):
    """
        Object stores all the information to for a Washer model
    """

    def __init__(self, mockup = True, topload = False):
        self._type = "washer"
        if mockup:
            self.shape = "../models/items/washer_mock_up/washer_col.xml"
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
