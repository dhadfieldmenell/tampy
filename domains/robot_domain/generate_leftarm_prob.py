from IPython import embed as shell
import itertools
import numpy as np
import random


# SEED = 1234
NUM_PROBS = 1
NUM_CLOTH = 1
filename = "probs/left_arm_prob{}.prob".format(NUM_CLOTH)
GOAL = "(RobotAt baxter robot_end_pose)"


# init Baxter pose
BAXTER_INIT_POSE = [0, 0, 0]
BAXTER_END_POSE = [0, 0, 0]
L_ARM_INIT = [0.39620987, -0.97739414, -0.04612781, 1.74220501, 0.03562036, 0.8089644, -0.45207411]
R_ARM_INIT = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
OPEN_GRIPPER = [0.02]
CLOSE_GRIPPER = [0.015]

MONITOR_LEFT = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
MONITOR_RIGHT = [-np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
CLOTH_ROT = [0, 0, 0]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375-0.665]
TABLE_ROT = [0,0,0]

ROBOT_DIST_FROM_TABLE = 0.05
REGION1 = [np.pi/4]
REGION2 = [0]
REGION3 = [-np.pi/4]
REGION4 = [-np.pi/2]

cloth_init_poses = np.ones((NUM_CLOTH, 3)) * 0.615
cloth_init_poses = cloth_init_poses.tolist()

def get_baxter_pose_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(left {} {}), ".format(name, LArm)
    s += "(left_ee_pos {} {}), ".format(name, [0,0,0])
    s += "(left_gripper {} {}), ".format(name, G)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, [0,0,0])
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    return s

def get_baxter_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(left {} {}), ".format(name, LArm)
    s += "(left_ee_pos {} {}), ".format(name, [0,0,0])
    s += "(left_gripper {} {}), ".format(name, G)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, [0,0,0])
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(left {} undefined), ".format(name)
    s += "(left_ee_pos {} undefined), ".format(name)
    s += "(left_gripper {} undefined), ".format(name)
    s += "(right {} undefined), ".format(name)
    s += "(right_ee_pos {} undefined), ".format(name)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    return s

def get_undefined_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def main():
    for iteration in range(NUM_PROBS):
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        s += "Baxter (name baxter); "
        for i in range(NUM_CLOTH):
            s += "Cloth (name {}); ".format("cloth{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth{0}_init_target".format(i))
            s += "ClothTarget (name {}); ".format("cloth{0}_end_target".format(i))

        s += "BaxterPose (name {}); ".format("cloth_grasp_begin".format(i))
        s += "BaxterPose (name {}); ".format("cloth_grasp_end".format(i))
        s += "BaxterPose (name {}); ".format("cloth_putdown_begin".format(i))
        s += "BaxterPose (name {}); ".format("cloth_putdown_end".format(i))
        s += "ClothTarget (name {}); ".format("middle_target_1")
        s += "ClothTarget (name {}); ".format("middle_target_2")
        s += "ClothTarget (name {}); ".format("left_mid_target")
        s += "ClothTarget (name {}); ".format("right_mid_target")

        s += "BaxterPose (name {}); ".format("robot_init_pose")
        s += "BaxterPose (name {}); ".format("robot_end_pose")
        s += "Obstacle (name {}) \n\n".format("table")

        s += "Init: "

        for i in range(NUM_CLOTH):
            s += "(geom cloth{0}), ".format(i)
            s += "(pose cloth{0} {1}), ".format(i, [0, 0, 0])
            s += "(rotation cloth{0} {1}), ".format(i, [0, 0, 0])
            s += "(value cloth{0}_init_target [0, 0, 0]), ".format(i)
            s += "(rotation cloth{0}_init_target [0, 0, 0]), ".format(i)
            s += "(value cloth_target_{0} [0, 0, 0]), ".format(i)
            s += "(rotation cloth_target_{0} [0, 0, 0]), ".format(i)
            s += "(value cloth{0}_end_target [0, 0, 0]), ".format(i)
            s += "(rotation cloth{0}_end_target [0, 0, 0]), ".format(i)

        s += "(value middle_target_1 [0, 0, 0]), "
        s += "(rotation middle_target_1 [0, 0, 0]), "
        s += "(value middle_target_2 [0, 0, 0]), "
        s += "(rotation middle_target_2 [0, 0, 0]), "
        s += "(value left_mid_target [0, 0, 0]), "
        s += "(rotation left_mid_target [0, 0, 0]), "
        s += "(value right_mid_target [0, 0, 0]), "
        s += "(rotation right_mid_target [0, 0, 0]), "

        s += get_undefined_robot_pose_str("cloth_grasp_begin".format(i))
        s += get_undefined_robot_pose_str("cloth_grasp_end".format(i))
        s += get_undefined_robot_pose_str("cloth_putdown_begin".format(i))
        s += get_undefined_robot_pose_str("cloth_putdown_end".format(i))

        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_INIT_POSE)
        s += get_baxter_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_INIT_POSE)
        # s += get_baxter_pose_str('robot_end_pose', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_END_POSE)
        s += get_undefined_robot_pose_str('robot_end_pose')

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}); ".format(TABLE_ROT)

        for n in range(NUM_CLOTH):
            s += "(At cloth{0} cloth{0}_init_target), ".format(n)
        s += "(RobotAt baxter robot_init_pose),"
        s += "(StationaryBase baxter), "
        s += "(IsMP baxter), "
        s += "(WithinJointLimit baxter), "
        s += "(StationaryW table) \n\n"

        s += "Goal: {}\n\n".format(GOAL)

        s += "Invariants: "
        s += "(StationaryBase baxter), "
        s += "(StationaryRightArm baxter), "
        s += "(LeftGripperDownRot baxter), "
        #s += "(LeftEEValid baxter), "
        s += "\n\n"

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()