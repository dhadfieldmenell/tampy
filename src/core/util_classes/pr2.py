GEOM_FILE = "../models/pr2/pr2.zae"

class PR2(object):
    """
    Defines geometry used in the PR2 domain.
    """
    def __init__(self):
        self.shape = GEOM_FILE
        # down to 30 links from 45
        self.col_links = set(['base_link', 'torso_lift_link', 'l_shoulder_pan_link', 'l_shoulder_lift_link', 'l_upper_arm_roll_link', 'l_upper_arm_link', 'l_elbow_flex_link', 'l_forearm_roll_link', 'l_forearm_link', 'l_wrist_flex_link', 'l_wrist_roll_link', 'l_gripper_palm_link', 'l_gripper_l_finger_link', 'l_gripper_l_finger_tip_link', 'l_gripper_r_finger_link', 'l_gripper_r_finger_tip_link', 'r_shoulder_pan_link', 'r_shoulder_lift_link', 'r_upper_arm_roll_link', 'r_upper_arm_link', 'r_elbow_flex_link', 'r_forearm_roll_link', 'r_forearm_link', 'r_wrist_flex_link', 'r_wrist_roll_link', 'r_gripper_palm_link', 'r_gripper_l_finger_link', 'r_gripper_l_finger_tip_link', 'r_gripper_r_finger_link', 'r_gripper_r_finger_tip_link'])
