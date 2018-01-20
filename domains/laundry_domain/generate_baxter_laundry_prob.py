from IPython import embed as shell
import itertools
import numpy as np
import random

import ros_interface.utils as utils


NUM_CLOTH = 2
NUM_SYMBOLS = 20

# SEED = 1234
NUM_PROBS = 1
filename = "laundry_probs/baxter_laundry_{0}.prob".format(NUM_CLOTH)
GOAL = "(BaxterRobotAt baxter robot_end_pose), (BaxterWasherAt washer washer_close_pose_0)"


# init Baxter pose
BAXTER_INIT_POSE = [np.pi/4]
BAXTER_END_POSE = [np.pi/4]
R_ARM_INIT = [-np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
L_ARM_INIT = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
INT_GRIPPER = [0.02]
CLOSE_GRIPPER = [0.015]

MONITOR_LEFT = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
MONITOR_RIGHT = [-np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]

# init basket pose
BASKET_NEAR_POS = utils.basket_near_pos
BASKET_FAR_POS = utils.basket_far_pos
BASKET_NEAR_ROT = utils.basket_near_rot
BASKET_FAR_ROT = utils.basket_far_rot

CLOTH_ROT = [0, 0, 0]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_ROT = [0,0,0]

ROBOT_DIST_FROM_TABLE = 0.05

WASHER_CONFIG = [True, True]
# WASHER_INIT_POS = [0.97, 1.0, 0.97-0.375+0.65/2]
# WASHER_INIT_ROT = [np.pi/2,0,0]
WASHER_INIT_POS = [0.85, 1.25, 0.97-0.375+0.65/2]
WASHER_INIT_ROT = [3*np.pi/4,0,0]

WASHER_OPEN_DOOR = [-np.pi/2]
WASHER_CLOSE_DOOR = [0.0]
WASHER_PUSH_DOOR = [-np.pi/6]

REGION1 = [np.pi/4]
REGION2 = [0]
REGION3 = [-np.pi/4]
REGION4 = [-np.pi/2]

cloth_init_poses = np.ones((NUM_CLOTH, 3)) * 0.615
cloth_init_poses = cloth_init_poses.tolist()

def get_baxter_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    return s

def get_robot_pose_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(lArmPose {} undefined), ".format(name)
    s += "(lGripper {} undefined), ".format(name)
    s += "(rArmPose {} undefined), ".format(name)
    s += "(rGripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    return s

def get_undefined_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def get_underfine_washer_pose(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    s += "(door {} undefined), ".format(name)
    return s

def main():
    for iteration in range(NUM_PROBS):
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        s += "Basket (name {}); ".format("basket")

        s += "Robot (name baxter); "
        for i in range(NUM_CLOTH):
            s += "Cloth (name {}); ".format("cloth{0}".format(i))

        for i in range(NUM_SYMBOLS):
            s += "EEPose (name {}); ".format("cg_ee_{0}".format(i))
            s += "EEPose (name {}); ".format("cp_ee_{0}".format(i))
            s += "EEPose (name {}); ".format("bg_ee_left_{0}".format(i))
            s += "EEPose (name {}); ".format("bp_ee_left_{0}".format(i))
            s += "EEPose (name {}); ".format("bg_ee_right_{0}".format(i))
            s += "EEPose (name {}); ".format("bp_ee_right_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("basket_grasp_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("basket_grasp_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("basket_putdown_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("basket_putdown_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("robot_region_1_pose_{0}".format(i))
            s += "RobotPose (name {}); ".format("robot_region_2_pose_{0}".format(i))
            s += "RobotPose (name {}); ".format("robot_region_3_pose_{0}".format(i))
            s += "RobotPose (name {}); ".format("robot_region_4_pose_{0}".format(i))
            # s += "RobotPose (name {}); ".format("robot_region_1_pose_2_{0}".format(i))
            # s += "RobotPose (name {}); ".format("robot_region_2_pose_2_{0}".format(i))
            # s += "RobotPose (name {}); ".format("robot_region_3_pose_2_{0}".format(i))
            # s += "RobotPose (name {}); ".format("robot_region_4_pose_2_{0}".format(i))
            s += "EEPose (name {}); ".format("open_door_ee_approach_{0}".format(i))
            s += "EEPose (name {}); ".format("open_door_ee_retreat_{0}".format(i))
            s += "EEPose (name {}); ".format("close_door_ee_approach_{0}".format(i))
            s += "EEPose (name {}); ".format("close_door_ee_retreat_{0}".format(i))
            s += "RobotPose (name {}); ".format("close_door_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("open_door_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("close_door_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("open_door_end_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_begin_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_end_{0}".format(i))
            s += "WasherPose (name {}); ".format("washer_open_pose_{0}".format(i))
            s += "WasherPose (name {}); ".format("washer_close_pose_{0}".format(i))

        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "Washer (name {}); ".format("washer")
        s += "Obstacle (name {}); ".format("table")
        s += "BasketTarget (name {}); ".format("basket_near_target")
        s += "BasketTarget (name {}); ".format("basket_far_target")
        s += "Rotation (name {}); ".format("region1")
        s += "Rotation (name {}); ".format("region2")
        s += "Rotation (name {}); ".format("region3")
        s += "Rotation (name {}) \n\n".format("region4")

        s += "Init: "
        s += "(geom basket), "
        s += "(pose basket {}), ".format(BASKET_FAR_POS)
        s += "(rotation basket {}), ".format(BASKET_FAR_ROT)

        for i in range(NUM_CLOTH):
            s += "(geom cloth{0}), ".format(i)
            s += "(pose cloth{0} {1}), ".format(i, cloth_init_poses[i])
            s += "(rotation cloth{0} {1}), ".format(i, CLOTH_ROT)

        for i in range(NUM_SYMBOLS):
            s += get_undefined_symbol('cloth_target_end_{0}'.format(i))
            s += get_undefined_symbol('cloth_target_begin_{0}'.format(i))

            s += get_undefined_symbol("cg_ee_{0}".format(i))
            s += get_undefined_symbol("cp_ee_{0}".format(i))
            s += get_undefined_symbol("bg_ee_left_{0}".format(i))
            s += get_undefined_symbol("bp_ee_left_{0}".format(i))
            s += get_undefined_symbol("bg_ee_right_{0}".format(i))
            s += get_undefined_symbol("bp_ee_right_{0}".format(i))

            s += get_undefined_robot_pose_str("cloth_grasp_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_grasp_end_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_putdown_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_putdown_end_{0}".format(i))
            s += get_undefined_robot_pose_str("basket_grasp_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("basket_grasp_end_{0}".format(i))
            s += get_undefined_robot_pose_str("basket_putdown_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("basket_putdown_end_{0}".format(i))
            s += get_undefined_robot_pose_str("robot_region_1_pose_{0}".format(i))
            s += get_undefined_robot_pose_str("robot_region_2_pose_{0}".format(i))
            s += get_undefined_robot_pose_str("robot_region_3_pose_{0}".format(i))
            s += get_undefined_robot_pose_str("robot_region_4_pose_{0}".format(i))
            # s += get_undefined_robot_pose_str("robot_region_1_pose_2_{0}".format(i))
            # s += get_undefined_robot_pose_str("robot_region_2_pose_2_{0}".format(i))
            # s += get_undefined_robot_pose_str("robot_region_3_pose_2_{0}".format(i))
            # s += get_undefined_robot_pose_str("robot_region_4_pose_2_{0}".format(i))
            s += get_undefined_robot_pose_str("open_door_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("open_door_end_{0}".format(i))
            s += get_undefined_robot_pose_str("close_door_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("close_door_end_{0}".format(i))
            s += get_undefined_symbol("open_door_ee_approach_{0}".format(i))
            s += get_undefined_symbol("open_door_ee_retreat_{0}".format(i))
            s += get_undefined_symbol("close_door_ee_approach_{0}".format(i))
            s += get_undefined_symbol("close_door_ee_retreat_{0}".format(i))
            s += "(geom washer_open_pose_{0} {1}), ".format(i, WASHER_CONFIG)
            s += "(value washer_open_pose_{0} {1}), ".format(i, WASHER_INIT_POS)
            s += "(rotation washer_open_pose_{0} {1}), ".format(i, WASHER_INIT_ROT)
            s += "(door washer_open_pose_{0} {1}), ".format(i, WASHER_OPEN_DOOR)

            s += "(geom washer_close_pose_{0} {1}), ".format(i, WASHER_CONFIG)
            s += "(value washer_close_pose_{0} {1}), ".format(i, WASHER_INIT_POS)
            s += "(rotation washer_close_pose_{0} {1}), ".format(i, WASHER_INIT_ROT)
            s += "(door washer_close_pose_{0} {1}), ".format(i, WASHER_CLOSE_DOOR)

        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_end_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_END_POSE)

        s += "(value region1 {}), ".format(REGION1)
        s += "(value region2 {}), ".format(REGION2)
        s += "(value region3 {}), ".format(REGION3)
        s += "(value region4 {}), ".format(REGION4)

        s += "(geom washer {}), ".format(WASHER_CONFIG)
        s += "(pose washer {}), ".format(WASHER_INIT_POS)
        s += "(rotation washer {}), ".format(WASHER_INIT_ROT)
        s += "(door washer {}), ".format(WASHER_OPEN_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}), ".format(TABLE_ROT)

        s += "(geom basket_near_target), "
        s += "(value basket_near_target {}), ".format(BASKET_NEAR_POS)
        s += "(rotation basket_near_target {}), ".format(BASKET_NEAR_ROT)

        s += "(geom basket_far_target), "
        s += "(value basket_far_target {}), ".format(BASKET_FAR_POS)
        s += "(rotation basket_far_target {}); ".format(BASKET_FAR_ROT)


        # s += "(BaxterAt basket basket_init_target), "
        # s += "(BaxterBasketLevel basket), "
        s += "(BaxterRobotAt baxter robot_init_pose) \n\n"
        # s += "(BaxterWasherAt washer washer_init_pose), "
        # s += "(BaxterEEReachableLeftVer baxter basket_grasp_begin bg_ee_left), "
        # s += "(BaxterEEReachableRightVer baxter basket_grasp_begin bg_ee_right), "
        # s += "(BaxterBasketGraspValidPos bg_ee_left bg_ee_right basket_init_target), "
        # s += "(BaxterBasketGraspValidRot bg_ee_left bg_ee_right basket_init_target), "
        # s += "(BaxterBasketGraspValidPos bp_ee_left bp_ee_right end_target), "
        # s += "(BaxterBasketGraspValidRot bp_ee_left bp_ee_right end_target), "
        # s += "(BaxterStationaryBase baxter), "
        # s += "(BaxterIsMP baxter), "
        # s += "(BaxterWithinJointLimit baxter), "
        # s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()