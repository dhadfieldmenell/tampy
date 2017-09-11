from IPython import embed as shell
import itertools
import numpy as np
import random

NUM_PROBS=1
filename = "laundry_probs/cloth_grasp_policy.prob"
GOAL = "(BaxterRobotAt baxter robot_end_pose)"


# init Baxter pose
BAXTER_INIT_POSE = [0]
R_ARM_INIT = [0, -0.785, 0, 0, 0, 0, 0]
L_ARM_INIT = [ 0.7 , -0.77271635, -1.42337285,  1.94256044,  1.05746083, 0.7127481 , -0.38002847]
INT_GRIPPER = [0.02]
CLOSE_GRIPPER = [0.015]
# init basket pose
BASKET_INIT_POS = [0.75 , 0.3,  0.81]
BASKET_INIT_ROT = [0, 0, np.pi/2]

ROBOT_DIST_FROM_TABLE = 0.05
TABLE_GEOM = [0.3, 0.6, 0.018]
TABLE_POS = [0.75, 0.02, 0.522]
TABLE_ROT = [0,0,0]

# WASHER_POS = [2,2,2]
WASHER_POS = [0.08, 0.781, 0.28]
WASHER_ROT = [np.pi, 0, np.pi/2]
WASHER_DOOR = [0.0]
WASHER_END_DOOR = [-np.pi/2]
WASHER_CONFIG = [True, True]

CLOTH_ROT = [0,0,0]

CLOTH_INIT_POS_1 = [0.65, 0.401, 0.557]
CLOTH_INIT_ROT_1 = [0,0,0]

CLOTH_END_POS_1 = [0.65, -0.283,0.626]
CLOTH_END_ROT_1 = [0,0,0]


"""
Intermediate Poses adding it simplifies the plan
"""
GRASP_POSE = [-np.pi/8]
GRASP_LARMPOSE = [-0.2       , -1.61656414, -0.61176606,  1.93732774, -0.02776806, 1.24185857, -0.40960045]
GRASP_RARMPOSE = [ 0.7       , -0.96198717,  0.03612888,  0.99775438, -0.02067175, 1.5353429 , -0.44772444]
GRASP_GRIPPER = [0.02]

PUTDOWN_POSE = [np.pi/8]
PUTDOWN_LARMPOSE = [-0.8       , -0.87594019,  0.2587353 ,  0.92223949,  2.97696004, -1.54149409, -2.5580562 ]
PUTDOWN_RARMPOSE = [-0.2       , -1.38881187,  1.25178981,  1.81230334, -0.18056559, 1.27622517,  0.70704811]
PUTDOWN_GRIPPER = [0.02]

WASHER_BEGIN_POSE = [np.pi/3]
WASHER_BEGIN_LARMPOSE = [-0.8       , -0.93703369, -0.27464748,  1.09904023, -2.97863535, -1.4287909 ,  2.35686368]
WASHER_BEGIN_RARMPOSE = [-0.2       , -1.38881187,  1.25178981,  1.81230334, -0.18056559, 1.27622517,  0.70704811]

CLOTH_PUTDOWN_BEGIN_1_POSE = [0]
CLOTH_PUTDOWN_BEGIN_1_LARMPOSE = [-1.2, 0.30161054, -2.28704166, 0.95204077, 2.26996069, 1.91600073, -1.12607844]
CLOTH_PUTDOWN_BEGIN_1_RARMPOSE = [0, -0.785, 0, 0, 0, 0, 0]

WASHER_EE_POS = [0.29 ,  0.781,  0.785]
WASHER_EE_ROT = [0, np.pi/2, -np.pi/2]

WASHER_END_EE_POS = [-0.305,  0.781,  1.17 ]
WASHER_END_EE_ROT = [0,  0,  -np.pi/2]

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

def get_random_cloth_init_poses():
    cur_xy = [-.25, -.525]
    cloth_poses = []
    for i in range(10):
        if not (i+1) % 4:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(-0.4, -0.5), np.random.uniform(0.1, 0.15)])
            cur_xy[0] = max(cur_xy[0], -.25)
        else:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(0.1, 0.15), np.random.uniform(-0.025, 0.025)])
        pos = np.array(TABLE_POS) + np.array([cur_xy[0], cur_xy[1], 0.04])
        cloth_poses.append(pos.tolist())
    return cloth_poses

def get_random_cloth_end_poses():
    cur_xy = [-.11, .11]
    cloth_poses = []
    for i in range(10):
        if not (i+1) % 4:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(-0.21, -0.23), np.random.uniform(0.045, 0.055)])
            cur_xy[0] = max(cur_xy[0], -.11)
        else:
            cur_xy = np.array(cur_xy) + np.array([np.random.uniform(0.045, 0.055), np.random.uniform(-0.01, 0.01)])
        pos = np.array(TABLE_POS) + np.array([cur_xy[0], cur_xy[1], 0.04])
        cloth_poses.append(pos.tolist())
    return cloth_poses

cloth_init_poses = get_random_cloth_init_poses()
cloth_end_poses = get_random_cloth_end_poses()

def main():
    for iteration in range(NUM_PROBS):
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        s += "Basket (name {}); ".format("basket")
        s += "BasketTarget (name {}); ".format("init_target")
        s += "BasketTarget (name {}); ".format("end_target")

        s += "Robot (name {}); ".format("baxter")
        for i in range(10):
            s += "EEPose (name {}); ".format("cg_ee_{0}".format(i))
            s += "EEPose (name {}); ".format("cp_ee_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_grasp_end_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_begin_{0}".format(i))
            s += "RobotPose (name {}); ".format("cloth_putdown_end_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_begin_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_end_{0}".format(i))
            s += "Cloth (name {}); ".format("cloth_{0}".format(i))

        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "Washer (name {}); ".format("washer")
        s += "WasherPose (name {}); ".format("washer_init_pose")
        s += "WasherPose (name {}); ".format("washer_end_pose")
        s += "Obstacle (name {})\n\n".format("table")

        s += "Init: "
        s += "(geom basket), "
        s += "(pose basket {}), ".format(BASKET_INIT_POS)
        s += "(rotation basket {}), ".format(BASKET_INIT_ROT)
        s += "(geom init_target)"
        s += "(value init_target {}), ".format(BASKET_INIT_POS)
        s += "(rotation init_target {}), ".format(BASKET_INIT_ROT)

        s += "(geom end_target), "
        s += "(value end_target {}), ".format(BASKET_INIT_POS)
        s += "(rotation end_target {}), ".format(BASKET_INIT_ROT)

        for i in range(10):
            s += "(geom cloth_{0}), ".format(i)
            s += "(pose cloth_{0} {1}), ".format(i, cloth_init_poses[i])
            s += "(rotation cloth_{0} {1}), ".format(i, CLOTH_ROT)

            s += "(value cloth_target_begin_{0} {1}), ".format(i, cloth_init_poses[i])
            s += "(rotation cloth_target_begin_{0} {1}), ".format(i, CLOTH_ROT)

            s += "(value cloth_target_end_{0} {1}), ".format(i, cloth_end_poses[i])
            s += "(rotation cloth_target_end_{0} {1}), ".format(i, CLOTH_ROT)

            s += get_underfine_symbol("cg_ee_{0}".format(i))
            s += get_underfine_symbol("cp_ee_{0}".format(i))

            s += get_undefined_robot_pose_str("cloth_grasp_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_grasp_end_{0}".format(i))

            s += get_undefined_robot_pose_str("cloth_putdown_begin_{0}".format(i))
            s += get_undefined_robot_pose_str("cloth_putdown_end_{0}".format(i))


        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)

        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_end_pose', L_ARM_INIT, R_ARM_INIT, CLOSE_GRIPPER, BAXTER_INIT_POSE)

        s += "(geom washer {}), ".format(WASHER_CONFIG)
        s += "(pose washer {}), ".format(WASHER_POS)
        s += "(rotation washer {}), ".format(WASHER_ROT)
        s += "(door washer {}), ".format(WASHER_DOOR)

        s += "(geom washer_init_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_init_pose {}), ".format(WASHER_POS)
        s += "(rotation washer_init_pose {}), ".format(WASHER_ROT)
        s += "(door washer_init_pose {}), ".format(WASHER_DOOR)

        s += "(geom washer_end_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_end_pose {}), ".format(WASHER_POS)
        s += "(rotation washer_end_pose {}), ".format(WASHER_ROT)
        s += "(door washer_end_pose {}), ".format(WASHER_END_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}); ".format(TABLE_ROT)


        s += "(BaxterAt basket init_target), "
        s += "(BaxterBasketLevel basket), "
        s += "(BaxterRobotAt baxter robot_init_pose), "
        s += "(BaxterWasherAt washer washer_init_pose), "

        s += "(BaxterStationaryBase baxter), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()