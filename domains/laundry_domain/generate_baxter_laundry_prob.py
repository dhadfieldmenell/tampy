from IPython import embed as shell
import itertools
import numpy as np
import random

import ros_interface.utils as utils


NUM_CLOTH = 1
NUM_SYMBOLS = 5

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

# WASHER_SCAN_LARM = [1.7, -0.40261638, 0.05377066, 1.83155908, -1.68825323, 1.60365558, 2.99452377]
WASHER_SCAN_LARM = [1.6, -0.23329773, -0.94132533, 2.44132915, 1.21860071, -0.70738072,  0.57346915]
# WASHER_SCAN_LARM = [1., -0.86217968, -0.55454339, 2.50171728, 0.97046582, -1.27317176, 0.18678969]
WASHER_SCAN_RARM = [-np.pi/4, -0.8436, -0.09, 0.91, 0.043, 1.5, -0.05]

# CLOSE_DOOR_SCAN_LARM = [-0.5, -1.14183058, 2.30465956, 2.18895412, -2.53979033, 0.48512255, 2.2696758]
CLOSE_DOOR_SCAN_RARM = [-np.pi/4, -0.8436, -0.09, 0.91, 0.043, 1.5, -0.05]

CLOSE_DOOR_SCAN_LARM = [1.5, -0.80587341, -2.60831063, 2.44689589, -1.15837669, -1.30896309, 1.3912442]

# OPEN_DOOR_SCAN_LARM = [-1., -1.18189329, 2.71308993, 2.25489801, -1.93466169, 1.04191726, 1.94737484]
OPEN_DOOR_SCAN_RARM = [-np.pi/4, -0.8436, -0.09, 0.91, 0.043, 1.5, -0.05]

OPEN_DOOR_SCAN_LARM = [-0.9, -1.40654414, 2.34422276, 2.1106438, 1.41479777, -1.50970774, -1.14848834]
# OPEN_DOOR_SCAN_LARM = [1.7=,  0.50188187, -2.66683967, 2.37502706, -1.3456904, -1.2051367,  2.8246885]

# LOAD_WASHER_INTERMEDIATE_LARM = [0.8, 0.62943625, -1.44191234,  2.34674592, -1.2537245, -0.35386465, -2.92138512]
# LOAD_WASHER_INTERMEDIATE_LARM = [0.4, -0.706814, -0.92032024, 2.43696462, -1.8888946, 0.92580404, -3.01834089]
LOAD_WASHER_INTERMEDIATE_LARM = [1.2, -1.08151176, -1.77326592, 2.35781058, 1.49005473, -0.82626846, -0.70849067]
# LOAD_WASHER_INTERMEDIATE_LARM = [ 1., -0.27609405, -1.85094945,  2.37489925, -0.46382826,        0.15948452,  1.64496354]
# LOAD_WASHER_INTERMEDIATE_LARM = [-1.3, -1.87505533,  1.26395478,  2.40366208, -1.38750247,        1.3399258 ,  2.29505724]
# LOAD_WASHER_INTERMEDIATE_LARM = [ 1., -1.89881202, -1.24804102,  2.40897135, -1.9547075 ,        1.59862157,  2.25969824]

# PUT_INTO_WASHER_LARM =  [1.3, -0.49416386, -0.73767691,  2.44375669,  1.22131026, -1.01131956,  0.39567399]
# PUT_INTO_WASHER_LARM = [1.2, -0.52079592, -0.76534048, 2.39096019, 1.11967046, -1.01651007, 0.37709747]
PUT_INTO_WASHER_LARM = [1.7, 0.26427428, -1.07809409, 2.41433644, 0.89524332, -0.30717996, 1.20404544]

# IN_WASHER_ADJUST_LARM = [0.3, -0.22460627, -0.2449543, 1.85941311, -2.39364561, 1.44774053, 2.961685]
# IN_WASHER_ADJUST_LARM = [1.1, 0.54529813, -1.0642083, 1.731427, 1.36273194, -0.49012445, 0.4115331]
IN_WASHER_ADJUST_LARM = [1.2, 0.28760191, -0.89001922, 1.97487363, -1.96009447, 0.72513493, -2.69577026]

GRASP_EE_1_LARM = [1., 0.22291691, -0.90607442, 1.94649067, 1.08593605, -0.78341323, 0.4329876]

UNLOAD_WASHER_0_LARM = [0., -0.37067681, 0.00334665, 0.93011956, 1.8754418, -0.36962104, -1.63029407]
UNLOAD_WASHER_1_LARM = [0.3, -0.50636126, -0.33647714, 1.00977222, -0.36120271, 0.47043466, 1.56690279]
UNLOAD_WASHER_2_LARM = [-0.5, -0.69907449, 0.64630227, 1.58033184, -1.73427023, 0.60789478, 1.03357583]
UNLOAD_WASHER_3_LARM = [0., 0.06180852, -0.05918108, 0.21925722, -1.36508716, 0.40547471, 1.55224061]
UNLOAD_WASHER_4_LARM = [0.1, -0.04923454, -0.06040849, 0.37378552, 1.86542172, -0.48461707, -0.98989422]
UNLOAD_WASHER_5_LARM = [-0.2, -0.17400108, 0.23694677, 0.52691205, -1.23359682, 0.37644819, 0.51082788]

## Use PUT_INTO_WASHER_BEGIN; pretty much the same gripper position
# DOOR_SCAN_IR_LARM = [1.6, -0.72647354, 0.185418, 2.02671195, -1.71400472, 1.67279272, 2.84759038]
# DOOR_SCAN_IR_LARM = [1.7, -0.73574863, 0.01571105, 2.02178752, 1.45038389, -1.54767978, -0.28413698]

BASKET_SCAN_LARM = [0.75, -0.75, 0, 0, 0, 0, 0]
BASKET_SCAN_RARM = [-0.75, -0.75, 0, 0, 0, 0, 0]

# init basket pose
BASKET_NEAR_POS = utils.basket_near_pos.tolist()
BASKET_FAR_POS = utils.basket_far_pos.tolist()
BASKET_NEAR_ROT = utils.basket_near_rot.tolist()
BASKET_FAR_ROT = utils.basket_far_rot.tolist()

CLOTH_ROT = [0, 0, 0]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_ROT = [0,0,0]

ROBOT_DIST_FROM_TABLE = 0.05

WASHER_CONFIG = [True, True]
# WASHER_INIT_POS = [0.97, 1.0, 0.97-0.375+0.65/2]
# WASHER_INIT_ROT = [np.pi/2,0,0]
# WASHER_INIT_POS = [0.2, 1.39, 0.97-0.375+0.65/2]
WASHER_INIT_POS = [0.19, 1.37, 0.97-0.375+0.65/2+0.015]
WASHER_INIT_ROT = [5*np.pi/6,0,0]
# Center of barrel is at (0.1, 1.12)

WASHER_OPEN_DOOR = [-np.pi/2]
WASHER_CLOSE_DOOR = [0.0]
WASHER_PUSH_DOOR = [-np.pi/6]

# REGION1 = [np.pi/4]
# REGION2 = [0]
# REGION3 = [-np.pi/4]
# REGION4 = [-np.pi/2]

REGION1 = utils.regions[0]
REGION2 = utils.regions[1]
REGION3 = utils.regions[2]
REGION4 = utils.regions[3]

# # EEPOSE_PUT_INTO_WASHER_POS_1 = [0.05, 1.0, 0.75]
# EEPOSE_PUT_INTO_WASHER_POS_1 = [0.02, 1.14, 0.73]
# EEPOSE_PUT_INTO_WASHER_ROT_1 = [np.pi/3, np.pi/14, 0]
# EEPOSE_PUT_INTO_WASHER_POS_1 = [0.05, 1.0, 0.69]
# EEPOSE_PUT_INTO_WASHER_ROT_1 = [np.pi/3, 0, 0]
EEPOSE_PUT_INTO_WASHER_POS_1 = [0.08, 1, 0.78]
EEPOSE_PUT_INTO_WASHER_ROT_1 = [np.pi/3, 0, -np.pi/8]

# EEPOSE_PUT_INTO_WASHER_POS_2 = [0.12, 1.2, 0.85]
# EEPOSE_PUT_INTO_WASHER_POS_2 = [0.11, 1.15, 0.85]
EEPOSE_PUT_INTO_WASHER_POS_2 = [-0.03, 0.92, 0.9]
EEPOSE_PUT_INTO_WASHER_ROT_2 = [np.pi/3, 0, 0]

# EEPOSE_PUT_INTO_WASHER_POS_3 = [0.15, 1.3, 0.8]
EEPOSE_PUT_INTO_WASHER_POS_3 = [0.11, 1.25, 0.85]
EEPOSE_PUT_INTO_WASHER_ROT_3 = [np.pi/3, np.pi/20, 0]

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
        s += "RobotPose (name {}); ".format("washer_scan_pose")
        s += "RobotPose (name {}); ".format("close_door_scan_pose")
        s += "RobotPose (name {}); ".format("open_door_scan_pose")
        s += "RobotPose (name {}); ".format("basket_scan_pose_1")
        s += "RobotPose (name {}); ".format("basket_scan_pose_2")
        s += "RobotPose (name {}); ".format("basket_scan_pose_3")
        s += "RobotPose (name {}); ".format("basket_scan_pose_4")
        s += "RobotPose (name {}); ".format("load_washer_intermediate_pose")
        s += "RobotPose (name {}); ".format("put_into_washer_begin")
        s += "RobotPose (name {}); ".format("in_washer_adjust")
        s += "RobotPose (name {}); ".format("grab_ee_1")
        s += "RobotPose (name {}); ".format("unload_washer_0")
        s += "RobotPose (name {}); ".format("unload_washer_1")
        s += "RobotPose (name {}); ".format("unload_washer_2")
        s += "RobotPose (name {}); ".format("unload_washer_3")
        s += "RobotPose (name {}); ".format("unload_washer_4")
        s += "RobotPose (name {}); ".format("unload_washer_5")
        s += "Washer (name {}); ".format("washer")
        s += "Obstacle (name {}); ".format("table")
        s += "BasketTarget (name {}); ".format("basket_near_target")
        s += "BasketTarget (name {}); ".format("basket_far_target")
        s += "EEPose (name {}); ".format("put_into_washer_ee_1")
        s += "EEPose (name {}); ".format("put_into_washer_ee_2")
        s += "EEPose (name {}); ".format("put_into_washer_ee_3")
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
            # s += get_undefined_symbol('cloth_target_begin_{0}'.format(i))
            s += "(value cloth_target_begin_{0} {1})".format(i, cloth_init_poses[i % NUM_CLOTH])
            s += "(rotation cloth_target_begin_{0} {1})".format(i, [0, 0, 0])

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

        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('robot_end_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_END_POSE)
        s += get_robot_pose_str('washer_scan_pose', WASHER_SCAN_LARM, BASKET_SCAN_RARM, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('close_door_scan_pose', CLOSE_DOOR_SCAN_LARM, CLOSE_DOOR_SCAN_RARM, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('open_door_scan_pose', OPEN_DOOR_SCAN_LARM, OPEN_DOOR_SCAN_RARM, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('basket_scan_pose_1', BASKET_SCAN_LARM, BASKET_SCAN_RARM, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('basket_scan_pose_2', BASKET_SCAN_LARM, BASKET_SCAN_RARM, INT_GRIPPER, REGION2)
        s += get_robot_pose_str('basket_scan_pose_3', BASKET_SCAN_LARM, BASKET_SCAN_RARM, INT_GRIPPER, REGION3)
        s += get_robot_pose_str('basket_scan_pose_4', BASKET_SCAN_LARM, BASKET_SCAN_RARM, INT_GRIPPER, REGION4)
        s += get_robot_pose_str('load_washer_intermediate_pose', LOAD_WASHER_INTERMEDIATE_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('put_into_washer_begin', PUT_INTO_WASHER_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('in_washer_adjust', IN_WASHER_ADJUST_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('grab_ee_1', GRASP_EE_1_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_0', UNLOAD_WASHER_0_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_1', UNLOAD_WASHER_1_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_2', UNLOAD_WASHER_2_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_3', UNLOAD_WASHER_3_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_4', UNLOAD_WASHER_4_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_5', UNLOAD_WASHER_5_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)

        s += "(value region1 {}), ".format(REGION1)
        s += "(value region2 {}), ".format(REGION2)
        s += "(value region3 {}), ".format(REGION3)
        s += "(value region4 {}), ".format(REGION4)

        s += "(value put_into_washer_ee_1 {}), ".format(EEPOSE_PUT_INTO_WASHER_POS_1)
        s += "(rotation put_into_washer_ee_1 {}), ".format(EEPOSE_PUT_INTO_WASHER_ROT_1)
        s += "(value put_into_washer_ee_2 {}), ".format(EEPOSE_PUT_INTO_WASHER_POS_2)
        s += "(rotation put_into_washer_ee_2 {}), ".format(EEPOSE_PUT_INTO_WASHER_ROT_2)
        s += "(value put_into_washer_ee_3 {}), ".format(EEPOSE_PUT_INTO_WASHER_POS_3)
        s += "(rotation put_into_washer_ee_3 {}), ".format(EEPOSE_PUT_INTO_WASHER_ROT_3)

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
