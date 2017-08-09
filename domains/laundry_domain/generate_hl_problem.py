from IPython import embed as shell
import itertools
import numpy as np
import random

# SEED = 1234
NUM_PROBS = 1
filename = "laundry_probs/laundry_hl.prob"
# GOAL = "(BaxterRobotAt baxter robot_end_pose), (BaxterClothAt cloth cloth_target_end_1)"
GOAL = "(BaxterRobotAt baxter robot_end_pose), (BaxterBasketInGripper baxter basket)"


# robot_init_pose
BAXTER_INIT_POSE = [0]
BAXTER_INIT_R_ARM = [0, -0.785, 0, 0, 0, 0, 0]
BAXTER_INIT_L_ARM = [0, -0.785, 0, 0, 0, 0, 0]
BAXTER_OPEN_GRIPPER = [0.02]
# robot_end_pose
BAXTER_END_POSE = [0]
BAXTER_END_R_ARM = [0, -0.785, 0, 0, 0, 0, 0]
BAXTER_END_L_ARM = [0, -0.785, 0, 0, 0, 0, 0]
BAXTER_CLOSE_GRIPPER = [0.015]


MONITOR_L_ARM = [0, -0.785, 0, 0, 0, 0, 0]
MONITOR_R_ARM = [0, -0.785, 0, 0, 0, 0, 0]
MONITOR_POSE = [0]

# init basket pose
BASKET_INIT_POS = [0.65 , -0.283,  0.81]
BASKET_INIT_ROT = [np.pi/2, 0, np.pi/2]

# end basket pose
BASKET_END_POS = [0.65, 0.2, 0.81]
BASKET_END_ROT = [np.pi/2, 0, np.pi/2]

# table pose
ROBOT_DIST_FROM_TABLE = 0.05
TABLE_GEOM = [0.3, 0.6, 0.018]
TABLE_POS = [0.75, 0.02, 0.522]
TABLE_ROT = [0,0,0]

# washer_pose
WASHER_CONFIG = [True, True]
WASHER_INIT_POS = [1.0, 0.84, 0.85]
WASHER_INIT_ROT = [np.pi/2,0,0]
WASHER_OPEN_DOOR = [-np.pi/2]
WASHER_CLOSE_DOOR = [0.0]

# cloth target
CLOTH_INIT_POS_1 = [0.65, 0.401, 0.557]
CLOTH_INIT_ROT_1 = [0,0,0]

CLOTH_END_POS_1 = [ 0.65 , -0.283,  0.601]
CLOTH_END_ROT_1 = [0,0,0]

CLOTH_END_POS_2 = [ 0.65 ,  0.801, 0.557]
CLOTH_END_ROT_2 = [0,0,0]


def get_baxter_str(name, LArm = BAXTER_INIT_L_ARM, RArm = BAXTER_INIT_R_ARM, G = BAXTER_OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    return s

def get_robot_pose_str(name, LArm = BAXTER_INIT_L_ARM, RArm = BAXTER_INIT_R_ARM, G = BAXTER_OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
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

def get_underfine_symbol(name):
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
        s += "BasketTarget (name {}); ".format("basket_init_target")
        s += "BasketTarget (name {}); ".format("basket_end_target")

        s += "Robot (name {}); ".format("baxter")
        s += "EEPose (name {}); ".format("cg_ee_1")
        s += "EEPose (name {}); ".format("cp_ee_1")
        s += "EEPose (name {}); ".format("bg_ee_left")
        s += "EEPose (name {}); ".format("bg_ee_right")
        s += "EEPose (name {}); ".format("bp_ee_left")
        s += "EEPose (name {}); ".format("bp_ee_right")
        s += "EEPose (name {}); ".format("open_door_ee_1")
        s += "EEPose (name {}); ".format("open_door_ee_2")
        s += "EEPose (name {}); ".format("cg_ee_2")
        s += "EEPose (name {}); ".format("cp_ee_2")
        s += "EEPose (name {}); ".format("close_door_ee_1")
        s += "EEPose (name {}); ".format("close_door_ee_2")
        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("cloth_grasp_begin_1")
        s += "RobotPose (name {}); ".format("cloth_grasp_end_1")
        s += "RobotPose (name {}); ".format("cloth_putdown_begin_1")
        s += "RobotPose (name {}); ".format("cloth_putdown_end_1")
        s += "RobotPose (name {}); ".format("basket_grasp_begin")
        s += "RobotPose (name {}); ".format("basket_grasp_end")
        s += "RobotPose (name {}); ".format("basket_putdown_begin")
        s += "RobotPose (name {}); ".format("basket_putdown_end")
        s += "RobotPose (name {}); ".format("open_door_begin")
        s += "RobotPose (name {}); ".format("open_door_end")
        s += "RobotPose (name {}); ".format("cloth_grasp_begin_2")
        s += "RobotPose (name {}); ".format("cloth_grasp_end_2")
        s += "RobotPose (name {}); ".format("cloth_putdown_begin_2")
        s += "RobotPose (name {}); ".format("cloth_putdown_end_2")
        s += "RobotPose (name {}); ".format("close_door_begin")
        s += "RobotPose (name {}); ".format("close_door_end")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "RobotPose (name {}); ".format("monitor_pose")
        s += "Washer (name {}); ".format("washer")
        s += "WasherPose (name {}); ".format("washer_close_pose")
        s += "WasherPose (name {}); ".format("washer_open_pose")
        s += "Obstacle (name {}); ".format("table")
        s += "Cloth (name {}); ".format("cloth")
        s += "ClothTarget (name {}); ".format("cloth_init_target")
        s += "ClothTarget (name {}); ".format("cloth_target_end_1")
        s += "ClothTarget (name {}); ".format("cloth_target_begin_2")
        s += "ClothTarget (name {}) \n\n".format("cloth_target_end_2")

        s += "Init: "
        s += "(geom basket), "
        s += "(pose basket {}), ".format(BASKET_INIT_POS)
        s += "(rotation basket {}), ".format(BASKET_INIT_ROT)

        s += "(geom basket_init_target)"
        s += "(value basket_init_target {}), ".format(BASKET_INIT_POS)
        s += "(rotation basket_init_target {}), ".format(BASKET_INIT_ROT)

        s += "(geom basket_end_target), "
        s += "(value basket_end_target {}), ".format(BASKET_END_POS)
        s += "(rotation basket_end_target {}), ".format(BASKET_END_ROT)

        s += "(geom cloth), "
        s += "(pose cloth {}), ".format(CLOTH_INIT_POS_1)
        s += "(rotation cloth {}), ".format(CLOTH_INIT_ROT_1)

        s += "(value cloth_init_target {}), ".format(CLOTH_INIT_POS_1)
        s += "(rotation cloth_init_target {}), ".format(CLOTH_INIT_ROT_1)

        s += "(value cloth_target_end_1 {}), ".format(CLOTH_END_POS_1)
        s += "(rotation cloth_target_end_1 {}), ".format(CLOTH_END_ROT_1)

        s += get_underfine_symbol("cloth_target_begin_2")

        s += "(value cloth_target_end_2 {}), ".format(CLOTH_END_POS_2)
        s += "(rotation cloth_target_end_2 {}), ".format(CLOTH_END_ROT_2)

        s += get_underfine_symbol("cg_ee_1")
        s += get_underfine_symbol("cp_ee_1")
        s += get_underfine_symbol("bg_ee_left")
        s += get_underfine_symbol("bg_ee_right")
        s += get_underfine_symbol("bp_ee_left")
        s += get_underfine_symbol("bp_ee_right")
        s += get_underfine_symbol("open_door_ee_1")
        s += get_underfine_symbol("open_door_ee_2")
        s += get_underfine_symbol("cg_ee_2")
        s += get_underfine_symbol("cp_ee_2")
        s += get_underfine_symbol("close_door_ee_1")
        s += get_underfine_symbol("close_door_ee_2")

        s += get_baxter_str('baxter', BAXTER_INIT_L_ARM, BAXTER_INIT_R_ARM, BAXTER_OPEN_GRIPPER, BAXTER_INIT_POSE)

        s += get_robot_pose_str('robot_init_pose', BAXTER_INIT_L_ARM, BAXTER_INIT_R_ARM, BAXTER_OPEN_GRIPPER, BAXTER_INIT_POSE)
        s += get_undefined_robot_pose_str("cloth_grasp_begin_1")
        s += get_undefined_robot_pose_str("cloth_grasp_end_1")
        s += get_undefined_robot_pose_str("cloth_putdown_begin_1")
        s += get_undefined_robot_pose_str("cloth_putdown_end_1")
        s += get_undefined_robot_pose_str("basket_grasp_begin")
        s += get_undefined_robot_pose_str("basket_grasp_end")
        s += get_undefined_robot_pose_str("basket_putdown_begin")
        s += get_undefined_robot_pose_str("basket_putdown_end")
        s += get_undefined_robot_pose_str("open_door_begin")
        s += get_undefined_robot_pose_str("open_door_end")
        s += get_undefined_robot_pose_str("cloth_grasp_begin_2")
        s += get_undefined_robot_pose_str("cloth_grasp_end_2")
        s += get_undefined_robot_pose_str("cloth_putdown_begin_2")
        s += get_undefined_robot_pose_str("cloth_putdown_end_2")
        s += get_undefined_robot_pose_str("close_door_begin")
        s += get_undefined_robot_pose_str("close_door_end")
        s += get_robot_pose_str('monitor_pose', MONITOR_L_ARM, MONITOR_R_ARM, BAXTER_OPEN_GRIPPER, MONITOR_POSE)
        s += get_robot_pose_str('robot_end_pose', BAXTER_INIT_L_ARM, BAXTER_INIT_R_ARM, BAXTER_OPEN_GRIPPER, BAXTER_INIT_POSE)

        s += "(geom washer {}), ".format(WASHER_CONFIG)
        s += "(pose washer {}), ".format(WASHER_INIT_POS)
        s += "(rotation washer {}), ".format(WASHER_INIT_ROT)
        s += "(door washer {}), ".format(WASHER_CLOSE_DOOR)

        s += "(geom washer_close_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_close_pose {}), ".format(WASHER_INIT_POS)
        s += "(rotation washer_close_pose {}), ".format(WASHER_INIT_ROT)
        s += "(door washer_close_pose {}), ".format(WASHER_CLOSE_DOOR)

        s += "(geom washer_open_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_open_pose {}), ".format(WASHER_INIT_POS)
        s += "(rotation washer_open_pose {}), ".format(WASHER_INIT_ROT)
        s += "(door washer_open_pose {}), ".format(WASHER_OPEN_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}); ".format(TABLE_ROT)

        # Positional Constraints
        s += "(BaxterRobotAt baxter robot_init_pose), "
        s += "(BaxterClothAt cloth cloth_init_target), "
        s += "(BaxterAt basket basket_init_target), "
        # Stationary Constraints
        s += "(BaxterStationaryBase baxter), "
        s += "(BaxterStationary basket), "
        s += "(BaxterStationaryCloth cloth), "
        s += "(BaxterStationaryWasher washer), "
        s += "(BaxterStationaryWasherDoor washer), "
        s += "(BaxterStationaryW table), "
        # Pose Pairing Constraints
        s += "(BaxterPosePair cloth_grasp_begin_1 cloth_grasp_end_1), "
        s += "(BaxterPosePair cloth_putdown_begin_1 cloth_putdown_end_1), "
        s += "(BaxterPosePair basket_grasp_begin basket_grasp_end), "

        # Cloth Grasp Constraints
        s += "(BaxterEEReachableLeftVer baxter cloth_grasp_begin_1 cg_ee_1), "
        s += "(BaxterClothGraspValid cg_ee_1 cloth_init_target), "

        # Cloth Putdown Constraints
        s += "(BaxterEEReachableLeftVer baxter cloth_putdown_begin_1 cp_ee_1), "
        s += "(BaxterClothGraspValid cp_ee_1 cloth_target_end_1), "

        # Basket Grasp Constraints
        s += "(BaxterEEReachableLeftVer baxter basket_grasp_begin bg_ee_left), "
        s += "(BaxterEEReachableRightVer baxter basket_grasp_begin bg_ee_right), "
        s += "(BaxterBasketGraspLeftPos bg_ee_left basket_init_target), "
        s += "(BaxterBasketGraspLeftRot bg_ee_left basket_init_target), "
        s += "(BaxterBasketGraspRightPos bg_ee_right basket_init_target), "
        s += "(BaxterBasketGraspRightRot bg_ee_right basket_init_target), "


        s += "(BaxterBasketLevel basket), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterOpenGripperLeft baxter), "
        s += "(BaxterOpenGripperRight baxter) "

        s += " \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()
    print "Problem File Generated Successfully "
