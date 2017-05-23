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
      pass



class Baxter(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        baxter_shape = "../models/baxter/baxter.zae"
        self.col_links = set(["torso", "pedestal", "head", "sonar_ring", "screen", "collision_head_link_1",
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

      from openravepy import IkParameterizationType, databases
      manip = robot.GetManipulator('right_arm')
      iktype = IkParameterizationType.Transform6D
      ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, IkParameterizationType.Transform6D, True)
      if not ikmodel.load():
          print 'Something went wrong'
        #   ikmodel.autogenerate()

      ikmodel.manip = robot.GetManipulator('right_arm')
      manip.SetIkSolver(ikmodel.iksolver)
