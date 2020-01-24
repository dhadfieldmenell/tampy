from IPython import embed as shell
import itertools
import numpy as np
import random

# SEED = 1234
NUM_PROBS = 1
NUM_CLOTHS = 5
NUM_SYMBOLS = NUM_CLOTHS
filename = "laundry_probs/sort{0}.prob".format(NUM_CLOTHS)
GOAL = "(BaxterRobotAt baxter robot_end_pose)"
for i in range(NUM_CLOTHS):
    if i > NUM_CLOTHS / 2:
        GOAL += ", (BaxterClothAt cloth{0} left_target_1)".format(i)
    else:
        GOAL += ", (BaxterClothAt cloth{0} right_target_1)".format(i)


CLOTH_CAPSULES = (6, 3)

# init Baxter pose
BAXTER_INIT_POSE = [0]
R_ARM_INIT = [-np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
L_ARM_INIT = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
INT_GRIPPER = [0.02]
CLOSE_GRIPPER = [0.015]


ROBOT_DIST_FROM_TABLE = 0.05

# TABLE_GEOM = [1.4, 1.4, 0.97/2]
TABLE_GEOM = [0.3, 0.6, 0.97/2]
# TABLE_POS = [0, 0, 0.97/2-0.375]
TABLE_POS = [0.75, 0.0, 0.97/2-0.375]
TABLE_ROT = [0,0,0]


def get_baxter_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    s += "(ee_left_pos {} {}), ".format(name, [0, 0, 0])
    s += "(ee_left_rot {} {}), ".format(name, [0, 0, 0])
    s += "(ee_right_pos {} {}), ".format(name, [0, 0, 0])
    s += "(ee_right_rot {} {}), ".format(name, [0, 0, 0])
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

def get_underfine_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s


def main():
    for iteration in range(NUM_PROBS):
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "

        s += "Robot (name {}); ".format("baxter")
        s += "EEPose (name left_rest_pose); "
        s += "EEPose (name right_rest_pose); "
        s += "EEPose (name left_target_pose); "
        s += "EEPose (name right_target_pose); "

        s += "EEPose (name {}); ".format("cg_ee_left")
        s += "EEPose (name {}); ".format("cp_ee_left")
        s += "EEPose (name {}); ".format("cg_ee_right")
        s += "EEPose (name {}); ".format("cp_ee_right")
        s += "RobotPose (name {}); ".format("cloth_grasp_begin")
        s += "RobotPose (name {}); ".format("cloth_grasp_end")
        s += "RobotPose (name {}); ".format("cloth_putdown_begin")
        s += "RobotPose (name {}); ".format("cloth_putdown_end")
        
        for i in range(NUM_SYMBOLS):
            s += "EEPose (name {}); ".format("cg_ee_left_{0}".format(i))
            s += "EEPose (name {}); ".format("cp_ee_left_{0}".format(i))
            s += "EEPose (name {}); ".format("cg_ee_right_{0}".format(i))
            s += "EEPose (name {}); ".format("cp_ee_right_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_end_{0}".format(i))
        
        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")

        for i in range(NUM_CLOTHS):
            s += "Cloth (name {}); ".format("cloth{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth{0}_init_target".format(i))
            s += "ClothTarget (name {});".format("cloth{0}_end_target".format(i))
            s += "ClothTarget (name {});".format("cloth{0}_free_target".format(i))

        s += "ClothTarget (name left_target_1); "
        s += "ClothTarget (name right_target_1); "
        s += "ClothTarget (name middle_target_1); "


        s += "Obstacle (name {}); \n\n".format("table")

        s += "Init: "
        s += "(value left_rest_pose {0}), ".format([0.6, 0.5, 0.7])
        s += "(rotation left_rest_pose {0}), ".format([0, 1.57, 0])

        s += "(value right_rest_pose {0}), ".format([0.6, -0.5, 0.7])
        s += "(rotation right_rest_pose {0}), ".format([0, 1.57, 0])

        s += "(value left_target_pose {0}), ".format([0.6, 0.5, 0.7])
        s += "(rotation left_target_pose {0}), ".format([0, 1.57, 0])

        s += "(value right_target_pose {0}), ".format([0.6, -0.5, 0.7])
        s += "(rotation right_target_pose {0}), ".format([0, 1.57, 0])

        for i in range(NUM_CLOTHS):
            s += "(geom cloth{0}), ".format(i)
            s += "(pose cloth{0} {1}), ".format(i, [0, 0, 0.615])
            s += "(rotation cloth{0} {1}), ".format(i, [0, 0, 0])

            s += "(value cloth{0}_init_target {1}), ".format(i, [0, 0, 0.615])
            s += "(rotation cloth{0}_init_target {1}), ".format(i, [0, 0, 0])

            s += "(value cloth{0}_end_target {1}), ".format(i, [0, 0, 0.615])
            s += "(rotation cloth{0}_end_target {1}), ".format(i, [0, 0, 0])

            s += get_underfine_symbol("cloth{0}_free_target".format(i))

        s += "(value left_target_1 {0}),".format([0.4, 0.75, 0.625])
        s += "(value right_target_1 {0}),".format([0.4, -0.75, 0.625])
        s += "(value middle_target_1 {0}),".format([0.4, 0., 0.625])
        s += "(rotation left_target_1 {0}),".format([0, 0, 0])
        s += "(rotation right_target_1 {0}),".format([0, 0, 0])
        s += "(rotation middle_target_1 {0}),".format([0, 0., 0])



        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_undefined_robot_pose_str("cloth_grasp_begin")
        s += get_undefined_robot_pose_str("cloth_grasp_end")
        s += get_undefined_robot_pose_str("cloth_putdown_begin")
        s += get_undefined_robot_pose_str("cloth_putdown_end")
        s += get_underfine_symbol("cg_ee_left")
        s += get_underfine_symbol("cp_ee_left")
        s += get_underfine_symbol("cg_ee_right")
        s += get_underfine_symbol("cp_ee_right")
        for i in range(NUM_SYMBOLS):
            s += get_undefined_robot_pose_str("cloth_grasp_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_grasp_end_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_putdown_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_putdown_end_{0}".format(i))
            s += get_underfine_symbol("cg_ee_left_{0}".format(i))
            s += get_underfine_symbol("cp_ee_left_{0}".format(i))
            s += get_underfine_symbol("cg_ee_right_{0}".format(i))
            s += get_underfine_symbol("cp_ee_right_{0}".format(i))
        s += get_undefined_robot_pose_str('robot_end_pose')

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}); ".format(TABLE_ROT)

        s += "(BaxterRobotAt baxter robot_init_pose), "
        # for i in range(NUM_CLOTHS):
        #     s += "(BaxterClothTargetHor cloth{0}_end_target), ".format(i)

        # for i in range(NUM_CLOTHS):
        #     s += "(BaxterClothTargetVer cloth{0}_end_target), ".format(i)

        s += "(BaxterStationaryBase baxter), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()
