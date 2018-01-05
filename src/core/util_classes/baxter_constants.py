import numpy as np
"""
This file contains constants specifically in baxter domain.
This file is refferenced in:
    baxter_predicates
    baxter_sampling
    test_baxter_predicates
"""

"""
Following Constants are used in baxter_predicates
"""
# Baxter dimension constant
BASE_DIM = 1
JOINT_DIM = 16
ROBOT_ATTR_DIM = 17
TWOARMDIM = 16
JOINT_FAST_VELOCITY = 0.1
JOINT_SLOW_VELOCITY = 0.01

# Baxter Movement Constraints
BASE_MOVE = 1
JOINT_MOVE_FACTOR = 15
ROT_LB = -np.pi
ROT_UB = np.pi

# EEReachable Constants
APPROACH_DIST = 0.02
RETREAT_DIST = 0.02
EEREACHABLE_STEPS = 5

# Collision Constants
DIST_SAFE = 1e-3
RCOLLIDES_DSAFE = 1e-3
COLLIDES_DSAFE = 1e-3

# Plan Coefficient

EEREACHABLE_COEFF = 1e-3
EEREACHABLE_ROT_COEFF = 5e-1
IN_GRIPPER_COEFF = 7.5e1
IN_GRIPPER_ROT_COEFF = 2e1
WASHER_IN_GRIPPER_ROT_COEFF = 1e-2

OBSTRUCTS_COEFF = 7.5e-1
COLLIDE_COEFF = 1e2
RCOLLIDE_COEFF = 1e-1
MAX_CONTACT_DISTANCE = .1

GRASP_VALID_COEFF = 7.5e1
EEGRASP_VALID_COEFF = 1e2
# Gripper Value
GRIPPER_OPEN_VALUE = 0.02
GRIPPER_CLOSE_VALUE = 0.015
BASKET_OFFSET = 0.317

# How far to go into the washer
WASHER_DEPTH_OFFSET = -0.1

# Added height from rotor base
ROTOR_BASE_HEIGHT = 0.12

# Height difference from table top to bottom of basket
BASKET_BASE_DELTA = 0.035

"""
Following constants are used in baxter_sampling
"""

EE_ANGLE_SAMPLE_SIZE = 8
NUM_RESAMPLES = 10
MAX_ITERATION_STEP = 200
BIAS_RADIUS = 0.1
ROT_BIAS = np.pi/8
RESAMPLE_FACTOR = [0.005,0.005,0.25]
RESAMPLE_DIR = [1,1,0.5]
RESAMPLE_ROT = [np.pi/2, 0, 0]
RESAMPLE_FACTOR_LR = [0.1, 0.1, 0.05]
"""
Following constants are for testing purposes
"""

TOL = 1e-4
TEST_GRAD = True

'''
Following constants are for computing torque controllers
'''
time_delta = 0.005

'''
Following constants are for general use
'''
ATTR_MAP = {
    "Robot": (("lArmPose", np.array(range(7), dtype=np.int)),
             ("lGripper", np.array([0], dtype=np.int)),
             ("rArmPose", np.array(range(7), dtype=np.int)),
             ("rGripper", np.array([0], dtype=np.int)),
             ("pose", np.array([0], dtype=np.int)),
             ("time", np.array([0], dtype=np.int))),
    "RobotPose": (("lArmPose", np.array(range(7), dtype=np.int)),
                 ("lGripper", np.array([0], dtype=np.int)),
                 ("rArmPose", np.array(range(7), dtype=np.int)),
                 ("rGripper", np.array([0], dtype=np.int)),
                 ("value", np.array([0], dtype=np.int))),
    "RotDegree": (("value", np.array([0], dtype=np.int))),
    "Can": (("pose", np.array([0,1,2], dtype=np.int)),
           ("rotation", np.array([0,1,2], dtype=np.int))),
    "EEPose": (("value", np.array([0,1,2], dtype=np.int)),
              ("rotation", np.array([0,1,2], dtype=np.int))),
    "Target": (("value", np.array([0,1,2], dtype=np.int)),
              ("rotation", np.array([0,1,2], dtype=np.int))),
    "Table": (("pose", np.array([0,1,2], dtype=np.int)),
             ("rotation", np.array([0,1,2], dtype=np.int))),
    "Obstacle": (("pose", np.array([0,1,2], dtype=np.int)),
                ("rotation", np.array([0,1,2], dtype=np.int))),
    "Basket": (("pose", np.array([0,1,2], dtype=np.int)),
              ("rotation", np.array([0,1,2], dtype=np.int))),
    "BasketTarget": (("value", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
    "Washer": (("pose", np.array([0,1,2], dtype=np.int)),
              ("rotation", np.array([0,1,2], dtype=np.int)),
              ("door", np.array([0], dtype=np.int))),
    "WasherPose": (("value", np.array([0,1,2], dtype=np.int)),
                  ("rotation", np.array([0,1,2], dtype=np.int)),
                  ("door", np.array([0], dtype=np.int))),
    "Cloth": (("pose", np.array([0,1,2], dtype=np.int)),
             ("rotation", np.array([0,1,2], dtype=np.int))),
    "ClothTarget": (("value", np.array([0,1,2], dtype=np.int)),
             ("rotation", np.array([0,1,2], dtype=np.int))),
    "EEVel": (("value", np.array([0], dtype=np.int)))
}