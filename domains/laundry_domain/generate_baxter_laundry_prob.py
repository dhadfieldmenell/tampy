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
# WASHER_SCAN_LARM = [1.6, -0.23329773, -0.94132533, 2.44132915, 1.21860071, -0.70738072,  0.57346915]
# WASHER_SCAN_LARM = [1., -0.86217968, -0.55454339, 2.50171728, 0.97046582, -1.27317176, 0.18678969]
# WASHER_SCAN_LARM = [1.4, -1.01943417, -1.78970125, 1.86139833, 0.62509778, 0.65282738, -0.11619921]
WASHER_SCAN_LARM = [1.6, -1.08089616, -1.77184074, 2.17039854, 0.55206468, 0.19450778, -0.02135907]
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
# IN_WASHER_ADJUST_LARM = [1.2, 0.28760191, -0.89001922, 1.97487363, -1.96009447, 0.72513493, -2.69577026]
# IN_WASHER_ADJUST_LARM = [0.5, -0.76896147, -0.58273532, 2.05385498, 0.3773333, -0.53610864, -0.16385522]
# IN_WASHER_ADJUST_LARM = [0.5, -0.79825096, -0.58529658, 2.06204512, 0.38053872, -0.52297494, -0.1854936]
# IN_WASHER_ADJUST_LARM = [0.8, -0.40938122, -0.98349649, 1.93509787, -0.05496322, -0.42669661, 0.65419766]
# IN_WASHER_ADJUST_LARM = [0.6, -0.61635273, -0.71124468, 1.95806439, 0.29155893, -0.47623078, -0.01273099]
# IN_WASHER_ADJUST_LARM = [0.5, -0.62540308, -0.38016443, 1.92245318, 0.73700839, -0.74747471, 0.02996213]
# IN_WASHER_ADJUST_LARM = [0.5, -0.65235978, -0.38188952, 1.93549983, 0.73729593, -0.73988333, 0.01599394]
# IN_WASHER_ADJUST_LARM = [0.4, -0.62529473, -0.34263913, 1.84066668, 0.70013524, -0.70966563, -0.02596892]
# IN_WASHER_ADJUST_LARM = [0.3, -0.72533812, -0.32199372, 1.8276743, 0.66207906, -0.50719327, -0.11202211]
# IN_WASHER_ADJUST_LARM = [0.3, -0.65213943, -0.47141835, 2.01152129, 0.3227363, -0.79305333, -0.1028706]
IN_WASHER_ADJUST_LARM = [0.3, -0.79212235, -0.45905875, 2.06904321, 0.2875251, -0.61402656, -0.15201012]

# IN_WASHER_ADJUST_2_LARM = [0.4, -0.10145733, -1.15841966, 0.82868032, 0.71200458, 0.13920051, 0.5178818]
# IN_WASHER_ADJUST_2_LARM = [0.3, -0.4011153 , -0.64698334, 0.90038623, 1.55130337, -0.26267718, -0.86788791]
# IN_WASHER_ADJUST_2_LARM = [0.1, -0.282647, -0.43853686, 0.61053858, 1.25564311, -0.38928987, -0.77948963]
# IN_WASHER_ADJUST_2_LARM = [0.4, -0.66825283, -0.62261453, 1.45222368, 0.69664904, -0.5970904, -0.09223986]
IN_WASHER_ADJUST_2_LARM = [0.3, -0.48460244, -0.64299246, 1.17806108, 0.7626744 , -0.49882909, -0.10116166]

LOAD_BASKET_FAR_LARM = [-0.4, -1.03370072, -0.50974778, 0.40054644, -1.34636192, 1.6954402, 0.89965769]
LOAD_BASKET_NEAR_LARM = [0., -0.96128652, -1.96981899, 1.13576933, -0.25629043, 0.83559803, 0.74095247]
LOAD_BASKET_INTER_2_LARM = [-0.4, -1.03370072, -0.50974778, 0.40054644, -1.34636192, 1.6954402, 0.89965769]

START_GRASP_2_LARM = [0.5, -0.70216574, -1.89032033, 1.20493774, 1.35330378, 0.50691187, -0.35623453]

ROTATE_WITH_CLOTH_LARM = [-1., -1.30863276, 0.43647191, 0.98983662, -0.11649001, 1.91105911, 0.16966814]

GRASP_EE_1_LARM = [1., 0.22291691, -0.90607442, 1.94649067, 1.08593605, -0.78341323, 0.4329876]

# UNLOAD_WASHER_0_LARM = [0.4, -0.49244978, -0.54997052, 1.67150043, 0.51340403, -0.54670088, 0.22624599]
UNLOAD_WASHER_0_LARM = [0.3, -0.3552711, -0.61910971, 1.32043013, 0.51139179, -0.33070529, 0.23705041]
# UNLOAD_WASHER_0_LARM = [0.4, -0.50516885, -0.55034516, 1.67796919, 0.51074349, -0.54304382, 0.22157778]
UNLOAD_WASHER_1_LARM = [0.3, -0.50636126, -0.33647714, 1.00977222, -0.36120271, 0.47043466, 1.56690279]
UNLOAD_WASHER_2_LARM = [-0.5, -0.69907449, 0.64630227, 1.58033184, -1.73427023, 0.60789478, 1.03357583]
# UNLOAD_WASHER_3_LARM = [0.2, 0.00430631, -0.91698346, 0.39204878, 0.04554458, 0.36574977, 1.01936829]
# UNLOAD_WASHER_3_LARM = [0., 0.05451098, -0.80474549, 0.2519506, -0.09869922, 0.29024519, 0.48322255]
# UNLOAD_WASHER_3_LARM = [0.1, -0.26380738, -0.41376808, 0.81277683, -1.24936207, 0.18221274, 1.26918756]

# UNLOAD_WASHER_3_LARM = [0.2, -0.25126872, -0.47213494, 0.89539967, 1.47760549, -0.23967536, -0.84200852]
# UNLOAD_WASHER_3_LARM = [0.4, -0.19232621, -0.83302898,  1.02334477,  1.37054977, -0.09780143, -0.39119112]
# UNLOAD_WASHER_3_LARM = [0.3, -0.16528917, -0.8997124, 0.87071525, 0.76285755, 0.15585718, 0.23112184]
# UNLOAD_WASHER_3_LARM = [-0.1, 0.06280933, 0.11816894, 0.12925587, -1.10577863, 0.39185929, 1.08838455]
# UNLOAD_WASHER_3_LARM = [0.3, 0.0549754, -1.28082124, 0.67874805, 1.1711163, 0.30692752, 0.232318]
# UNLOAD_WASHER_3_LARM = [0.2, -0.14288316, -0.87448019, 0.74195305, 0.6839947, 0.20026969, 0.27126269]
# UNLOAD_WASHER_3_LARM = [ 0.3       ,  0.01516635, -1.23125549,  0.69629862,  1.12102826, 0.30277768,  0.21948926]
# UNLOAD_WASHER_3_LARM = [0.3, -0.17482291, -0.8992441, 0.92653075, 1.05108976, 0.12819128, -0.06616722]
UNLOAD_WASHER_3_LARM = [0., -0.17728146, -0.3153726 , 0.69641116, 1.52249477, -0.2103464, -1.08109426]
UNLOAD_WASHER_4_LARM = [0.1, -0.04923454, -0.06040849, 0.37378552, 1.86542172, -0.48461707, -0.98989422]
UNLOAD_WASHER_5_LARM = [-0.2, -0.17400108, 0.23694677, 0.52691205, -1.23359682, 0.37644819, 0.51082788]

# INTERMEDIATE_UNLOAD_LARM = [1.1, -0.86459062, -0.97227641, 2.23663424, 0.69570417, -0.47837221, -0.33542266]
# INTERMEDIATE_UNLOAD_LARM = [1.1, -0.5470185 , -0.93117317,  2.11125834,  0.91997842, -0.82086358,  0.27235029]
INTERMEDIATE_UNLOAD_LARM = [0.6, -1.23076612, -0.40187972, 2.53417753, 0.68715756, -1.26100506, -0.47337906]

## Use PUT_INTO_WASHER_BEGIN; pretty much the same gripper position
# DOOR_SCAN_IR_LARM = [1.6, -0.72647354, 0.185418, 2.02671195, -1.71400472, 1.67279272, 2.84759038]
# DOOR_SCAN_IR_LARM = [1.7, -0.73574863, 0.01571105, 2.02178752, 1.45038389, -1.54767978, -0.28413698]

BASKET_SCAN_LARM = [0.75, -0.75, 0, 0, 0, 0, 0]
BASKET_SCAN_RARM = [-0.75, -0.75, 0, 0, 0, 0, 0]

LARM_FORWARD = [0, -0.75, 0, 0, 0, 0, 0]
RARM_FORWARD = [0, -0.75, 0, 0, 0, 0, 0]

LARM_BACKWARD = [1.5, -0.75, 0, 0, 0, 0, 0]


CLOTH_FOLD_LARM_AFTER_GRASP = [-0.3, -1.14799978, -1.76643182, 1.13115012, 0.47054775, 2.04914857, -1.06937525] # EE at 0.5, 0, 1.2
CLOTH_FOLD_LARM_AFTER_DRAG = [0.5, -0.47861236, -0.7299432, 1.15500323, 0.71010347, 1.13834947, 0.51377547] # EE at 0.5, 0.8, 0.65
CLOTH_FOLD_LARM_SCAN_FOR_CORNER_1 = [-0.1, -0.93750567, -0.78511595, 0.80543256, 0.44764154, 1.83030012, 0.16473142] # EE at 0.75, 0.5, 1.05
CLOTH_FOLD_LARM_AFTER_GRASP_2 = [-1.4, -0.7146655, -1.22840151, 0.59829491, 0.11014072, 0.54789971, 0.69913018] # EE at 0.5, -0.7, 1.45
# CLOTH_FOLD_LARM_AFTER_DRAG_2 = [-0.1, -0.94072904, -0.58511655, 2.02695359, 0.56882439, 0.64868041, -1.81315771] # EE at 0.6, 0.4, 0.655
CLOTH_FOLD_LARM_AFTER_DRAG_2 = [-0.4, -1.16950788, -0.0342681, 2.3684357, 0.03679739, 0.37230904, -1.25122245]
# CLOTH_FOLD_RARM_SCAN_FOR_CORNER_2 = [0.8, -0.8133342, 0.08481631, 0.42255944, -0.06302153, 1.96266681, 0.0871061 ] # EE at 0.85, -0.2, 1.05
CLOTH_FOLD_RARM_SCAN_FOR_CORNER_2 = [0.8, -0.70980126, -0.20536722, 0.51791927, 0.16287424, 1.72182649, -0.0878267]
CLOTH_FOLD_LARM_SCAN_TWO_CORNER = [-0.6, -1.43361283, -0.81721771, 1.45860241, 0.09990902, 1.5887184, -0.66021639] # EE at 0.6, 0, 1.05

CLOTH_FOLD_HOLDING_BOTH_CORNERS_LARM = [-0.7, -1.06296608, 0.22662184, 1.00532877, -0.10973573, 1.63895279, 0.24181173]
CLOTH_FOLD_HOLDING_BOTH_CORNERS_RARM = [0.7, -1.01026434, -0.078992, 0.90689717, 0.04219482, 1.67547625, -0.18283847]
CLOTH_FOLD_HOLDING_BOTH_CORNERS_AFTER_DRAG_LARM = [0.3, -0.93675309, -0.88253178, 2.3922476 , -2.10919189, -0.56206989, 2.46083622]
CLOTH_FOLD_HOLDING_BOTH_CORNERS_AFTER_DRAG_RARM = [0.2, -1.12904411, 0.432573, 2.35899417, -0.45948146, 0.41601928, 0.20013397]
CLOTH_FOLD_SCAN_BOTH_CORNERS_LARM = [-1.1, -0.92635705, -0.03270419, 0.76229904, 0.01991296, 1.7350792, -0.3723978]
CLOTH_FOLD_LARM_FINAL_FOLD_BEGIN = [0.1, -0.60691025, -0.86389044, 1.01430532, -0.86561746, 1.70785855, 1.72458161]
CLOTH_FOLD_RARM_FINAL_FOLD_BEGIN = [0.2, -0.71224912, 0.40765909, 0.96818051, 1.20129272, 1.84515186, -1.78226479]
CLOTH_FOLD_LARM_FINAL_FOLD_MID = [0.7, -0.32352744, -0.92111569, 1.97524021, -0.34446657, 1.28416016, 2.31872965]
CLOTH_FOLD_RARM_FINAL_FOLD_MID = [-0.8, 0.17580006, 1.26630857, 1.90759271, -0.19344644, 1.23466217, -1.80687513]
CLOTH_FOLD_FINAL_FOLD_LARM_END = [0., -0.03439064, -0.80929995, 0.92621978, -0.09353719, 1.30169803, 2.06723639]
CLOTH_FOLD_FINAL_FOLD_RARM_END = [0.1, 0.02054086, 0.87543503, 0.82736412, 0.02483959, 1.33911524, -2.02606305]

# init basket pose
BASKET_NEAR_POS = utils.basket_near_pos.tolist()
BASKET_FAR_POS = utils.basket_far_pos.tolist()
BASKET_NEAR_ROT = utils.basket_near_rot.tolist()
BASKET_FAR_ROT = utils.basket_far_rot.tolist()

CLOTH_ROT = [0, 0, 0]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
# TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_ROT = [0,0,0]

WALL_GEOM = [0.03, 0.25, 0.5]
WALL_POS = [-0.6, 0.75, 0.5]
WALL_ROT = [0,0,0]

ROBOT_DIST_FROM_TABLE = 0.05

WASHER_CONFIG = [True, True]
# WASHER_INIT_POS = [0.97, 1.0, 0.97-0.375+0.65/2]
# WASHER_INIT_ROT = [np.pi/2,0,0]
# WASHER_INIT_POS = [0.2, 1.39, 0.97-0.375+0.65/2]
# WASHER_INIT_POS = [0.19, 1.37, 0.97-0.375+0.65/2+0.015]
# WASHER_INIT_POS = [0.2, 1.37, 0.97-0.375+0.65/2+0.05] # true height
WASHER_INIT_POS = [0.29, 1.37, 0.97-0.375+0.65/2+0.03]
# WASHER_INIT_POS = [0.27, 1.37, 0.97-0.375+0.65/2+0.05] # true height
# WASHER_INIT_POS = [0.24, 1.4, 0.97-0.375+0.65/2+0.015]
# WASHER_INIT_ROT = [5*np.pi/6,0,0]
WASHER_INIT_ROT = [3*np.pi/4, 0, 0]
# Center of barrel is at (0.1, 1.12)

WASHER_OPEN_DOOR = [-np.pi/2]
WASHER_CLOSE_DOOR = [0.0]
WASHER_PUSH_DOOR = [-np.pi/4]

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
EEPOSE_PUT_INTO_WASHER_POS_2 = [-0.03, 0.92, 0.95]
EEPOSE_PUT_INTO_WASHER_ROT_2 = [np.pi/3, 0, 0]

# EEPOSE_PUT_INTO_WASHER_POS_3 = [0.15, 1.3, 0.8]
EEPOSE_PUT_INTO_WASHER_POS_3 = [0.11, 1.25, 0.9]
EEPOSE_PUT_INTO_WASHER_ROT_3 = [np.pi/3, np.pi/20, 0]

cloth_init_poses = np.ones((NUM_CLOTH, 3)) * [-1, 1, 0.625]
cloth_init_poses = cloth_init_poses.tolist()

CLOTH_1_LENGTH = 0.35
CLOTH_2_LENGTH = 0.1

CLOTH_FOLD_AIR_TARGET_1_POSE = [0.8, 0, 0.95]
CLOTH_FOLD_AIR_TARGET_1_ROT = [0, 0, -np.pi/2]

CLOTH_FOLD_TABLE_TARGET_1_POSE = [0.55, 0, 0.65]
CLOTH_FOLD_TABLE_TARGET_1_ROT = [0, 0, -np.pi/2]

CLOTH_FOLD_AIR_TARGET_2_POSE = [1.1, 0.02, 1.5]
CLOTH_FOLD_AIR_TARGET_2_ROT = [0, 0, -np.pi/2]

CLOTH_FOLD_TABLE_TARGET_2_POSE = [0.6, 0.02, 1.05]
CLOTH_FOLD_TABLE_TARGET_2_ROT = [0, 0, -np.pi/2]

CLOTH_FOLD_TABLE_TARGET_3_POSE = [0.9, 0.02, 0.65]
CLOTH_FOLD_TABLE_TARGET_3_ROT = [0, 0, -np.pi/2]

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
            s += "WasherPose (name {}); ".format("washer_push_pose_{0}".format(i))

        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "RobotPose (name {}); ".format("washer_scan_pose")
        s += "RobotPose (name {}); ".format("close_door_scan_pose")
        s += "RobotPose (name {}); ".format("open_door_scan_pose")
        s += "RobotPose (name {}); ".format("basket_scan_pose_1")
        s += "RobotPose (name {}); ".format("basket_scan_pose_2")
        s += "RobotPose (name {}); ".format("basket_scan_pose_3")
        s += "RobotPose (name {}); ".format("basket_scan_pose_4")
        s += "RobotPose (name {}); ".format("arms_forward_1")
        s += "RobotPose (name {}); ".format("arms_forward_2")
        s += "RobotPose (name {}); ".format("arms_forward_3")
        s += "RobotPose (name {}); ".format("arms_forward_4")
        s += "RobotPose (name {}); ".format("arm_back_1")
        s += "RobotPose (name {}); ".format("arm_back_2")
        s += "RobotPose (name {}); ".format("arm_back_3")
        s += "RobotPose (name {}); ".format("arm_back_4")
        s += "RobotPose (name {}); ".format("load_washer_intermediate_pose")
        s += "RobotPose (name {}); ".format("put_into_washer_begin")
        s += "RobotPose (name {}); ".format("in_washer_adjust")
        s += "RobotPose (name {}); ".format("in_washer_adjust_2")
        s += "RobotPose (name {}); ".format("grab_ee_1")
        s += "RobotPose (name {}); ".format("load_basket_far")
        s += "RobotPose (name {}); ".format("load_basket_near")
        s += "RobotPose (name {}); ".format("load_basket_inter_1")
        s += "RobotPose (name {}); ".format("load_basket_inter_2")
        s += "RobotPose (name {}); ".format("load_basket_inter_4")
        s += "RobotPose (name {}); ".format("unload_washer_0")
        s += "RobotPose (name {}); ".format("unload_washer_1")
        s += "RobotPose (name {}); ".format("unload_washer_2")
        s += "RobotPose (name {}); ".format("unload_washer_3")
        s += "RobotPose (name {}); ".format("unload_washer_4")
        s += "RobotPose (name {}); ".format("unload_washer_5")
        s += "RobotPose (name {}); ".format("intermediate_unload")
        s += "RobotPose (name {}); ".format("start_grasp_2")
        s += "RobotPose (name {}); ".format("rotate_with_cloth_2")
        s += "RobotPose (name {}); ".format("fold_after_grasp")
        s += "RobotPose (name {}); ".format("fold_after_drag")
        s += "RobotPose (name {}); ".format("fold_scan_corner_1")
        s += "RobotPose (name {}); ".format("fold_after_grasp_2")
        s += "RobotPose (name {}); ".format("fold_after_drag_2")
        s += "RobotPose (name {}); ".format("fold_scan_corner_2")
        s += "RobotPose (name {}); ".format("fold_scan_two_corner")
        s += "Washer (name {}); ".format("washer")
        s += "Obstacle (name {}); ".format("table")
        s += "Obstacle (name {}); ".format("wall")
        s += "BasketTarget (name {}); ".format("basket_near_target")
        s += "BasketTarget (name {}); ".format("basket_far_target")
        s += "EEPose (name {}); ".format("put_into_washer_ee_1")
        s += "EEPose (name {}); ".format("put_into_washer_ee_2")
        s += "EEPose (name {}); ".format("put_into_washer_ee_3")
        s += "Rotation (name {}); ".format("region1")
        s += "Rotation (name {}); ".format("region2")
        s += "Rotation (name {}); ".format("region3")
        s += "Rotation (name {}); ".format("region4")
        s += "ClothTarget (name {}); ".format("cloth_fold_air_target_1")
        s += "ClothTarget (name {}); ".format("cloth_fold_table_target_1")
        s += "ClothTarget (name {}); ".format("cloth_fold_air_target_2")
        s += "ClothTarget (name {}); ".format("cloth_fold_table_target_2")
        s += "ClothTarget (name {}); ".format("cloth_fold_table_target_3")
        s += "Can (name {}); ".format("cloth_long_edge")
        s += "Can (name {}); \n\n".format("cloth_short_edge")

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
            s += "(rotation cloth_target_begin_{0} {1})".format(i, [0.0, 0.0, 0.0])

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

            s += "(geom washer_push_pose_{0} {1}), ".format(i, WASHER_CONFIG)
            s += "(value washer_push_pose_{0} {1}), ".format(i, WASHER_INIT_POS)
            s += "(rotation washer_push_pose_{0} {1}), ".format(i, WASHER_INIT_ROT)
            s += "(door washer_push_pose_{0} {1}), ".format(i, WASHER_PUSH_DOOR)

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
        s += get_robot_pose_str('arms_forward_1', LARM_FORWARD, RARM_FORWARD, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('arms_forward_2', LARM_FORWARD, RARM_FORWARD, INT_GRIPPER, REGION2)
        s += get_robot_pose_str('arms_forward_3', LARM_FORWARD, RARM_FORWARD, INT_GRIPPER, REGION3)
        s += get_robot_pose_str('arms_forward_4', LARM_FORWARD, RARM_FORWARD, INT_GRIPPER, REGION4)
        s += get_robot_pose_str('arm_back_1', LARM_BACKWARD, BASKET_SCAN_RARM, INT_GRIPPER, REGION1)
        s += get_robot_pose_str('arm_back_2', LARM_BACKWARD, BASKET_SCAN_RARM, INT_GRIPPER, REGION2)
        s += get_robot_pose_str('arm_back_3', LARM_BACKWARD, BASKET_SCAN_RARM, INT_GRIPPER, REGION3)
        s += get_robot_pose_str('arm_back_4', LARM_BACKWARD, BASKET_SCAN_RARM, INT_GRIPPER, REGION4)
        s += get_robot_pose_str('load_washer_intermediate_pose', LOAD_WASHER_INTERMEDIATE_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('put_into_washer_begin', PUT_INTO_WASHER_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('in_washer_adjust', IN_WASHER_ADJUST_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('in_washer_adjust_2', IN_WASHER_ADJUST_2_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('grab_ee_1', GRASP_EE_1_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('load_basket_far', LOAD_BASKET_FAR_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION3)
        s += get_robot_pose_str('load_basket_near', LOAD_BASKET_NEAR_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('load_basket_inter_1', LOAD_BASKET_NEAR_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('load_basket_inter_2', LOAD_BASKET_FAR_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('load_basket_inter_4', LOAD_BASKET_FAR_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION4)
        s += get_robot_pose_str('unload_washer_0', UNLOAD_WASHER_0_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_1', UNLOAD_WASHER_1_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_2', UNLOAD_WASHER_2_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_3', UNLOAD_WASHER_3_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_4', UNLOAD_WASHER_4_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('unload_washer_5', UNLOAD_WASHER_5_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('intermediate_unload', INTERMEDIATE_UNLOAD_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION1)
        s += get_robot_pose_str('start_grasp_2', START_GRASP_2_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('rotate_with_cloth_2', ROTATE_WITH_CLOTH_LARM, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_after_grasp', CLOTH_FOLD_LARM_AFTER_GRASP, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_after_drag', CLOTH_FOLD_LARM_AFTER_DRAG, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_scan_corner_1', CLOTH_FOLD_LARM_SCAN_FOR_CORNER_1, BASKET_SCAN_RARM, INT_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_after_grasp_2', CLOTH_FOLD_LARM_AFTER_GRASP_2, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_after_drag_2', CLOTH_FOLD_LARM_AFTER_DRAG_2, BASKET_SCAN_RARM, CLOSE_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_scan_corner_2', CLOTH_FOLD_LARM_AFTER_DRAG_2, CLOTH_FOLD_RARM_SCAN_FOR_CORNER_2, INT_GRIPPER, REGION2)
        s += get_robot_pose_str('fold_scan_two_corner', CLOTH_FOLD_LARM_SCAN_TWO_CORNER, BASKET_SCAN_RARM, INT_GRIPPER, REGION2)

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

        s += "(value cloth_fold_table_target_1 {}), ".format(CLOTH_FOLD_TABLE_TARGET_1_POSE)
        s += "(rotation cloth_fold_table_target_1 {}), ".format(CLOTH_FOLD_TABLE_TARGET_1_ROT)

        s += "(value cloth_fold_air_target_1 {}), ".format(CLOTH_FOLD_AIR_TARGET_1_POSE)
        s += "(rotation cloth_fold_air_target_1 {}), ".format(CLOTH_FOLD_AIR_TARGET_1_ROT)

        s += "(value cloth_fold_table_target_2 {}), ".format(CLOTH_FOLD_TABLE_TARGET_2_POSE)
        s += "(rotation cloth_fold_table_target_2 {}), ".format(CLOTH_FOLD_TABLE_TARGET_2_ROT)

        s += "(value cloth_fold_air_target_2 {}), ".format(CLOTH_FOLD_AIR_TARGET_2_POSE)
        s += "(rotation cloth_fold_air_target_2 {}), ".format(CLOTH_FOLD_AIR_TARGET_2_ROT)

        s += "(value cloth_fold_table_target_3 {}), ".format(CLOTH_FOLD_TABLE_TARGET_3_POSE)
        s += "(rotation cloth_fold_table_target_3 {}), ".format(CLOTH_FOLD_TABLE_TARGET_3_ROT)

        s += "(geom washer {}), ".format(WASHER_CONFIG)
        s += "(pose washer {}), ".format(WASHER_INIT_POS)
        s += "(rotation washer {}), ".format(WASHER_INIT_ROT)
        s += "(door washer {}), ".format(WASHER_OPEN_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}), ".format(TABLE_ROT)

        s += "(geom cloth_long_edge {} {}), ".format(0.01, CLOTH_1_LENGTH)
        s += "(pose cloth_long_edge {}), ".format([1.5,0,0])
        s += "(rotation cloth_long_edge {}), ".format([0,0,np.pi/2])

        s += "(geom cloth_short_edge {} {}), ".format(0.01, CLOTH_2_LENGTH)
        s += "(pose cloth_short_edge {}), ".format([1.6,0,0])
        s += "(rotation cloth_short_edge {}), ".format([0,0,np.pi/2])

        s += "(geom wall {}), ".format(WALL_GEOM)
        s += "(pose wall {}), ".format(WALL_POS)
        s += "(rotation wall {}), ".format(WALL_ROT)

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
