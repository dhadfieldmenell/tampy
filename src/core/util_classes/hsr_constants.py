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
BASE_DIM = 3
JOINT_DIM = 6
ROBOT_ATTR_DIM = 8
JOINT_FAST_VELOCITY = 0.1
JOINT_SLOW_VELOCITY = 0.01

# Baxter Movement Constraints
BASE_MOVE = 0.3
JOINT_MOVE_FACTOR = 15
ROT_LB = -np.pi
ROT_UB = np.pi
GRIPPER_OPEN = 0.75
GRIPPER_CLOSE = 0.5
GRIPPER_Y_OFFSET = 0.078
GRIPPER_OFFSET_ANGLE = 0.163 # Angle of line segment from robot x, y to hand x, y
GRIPPER_OFFSET_DISP = 0.480 # Distance from robot x, y to hand x, y
HAND_DIST = -0.04 # Distance from where the hand link registers to the center of the hand
COLLISION_DOF_INDICES = [0, 1, 2, 3, 4]

# EEReachable Constants
APPROACH_DIST = 0.02
RETREAT_DIST = 0.02
EEREACHABLE_STEPS = 5

# Collision Constants
DIST_SAFE = 1e-3
RCOLLIDES_DSAFE = 1e-3
COLLIDES_DSAFE = 1e-3

# Plan Coefficient

EEREACHABLE_COEFF = 5e-3
EEREACHABLE_ROT_COEFF = 5e-3
IN_GRIPPER_COEFF = 1e-1
IN_GRIPPER_ROT_COEFF = 1e-1
WASHER_IN_GRIPPER_ROT_COEFF = 1e-2
GRIPPER_AT_COEFF = 1e-2
GRIPPER_AT_ROT_COEFF = 1e-2

OBSTRUCTS_COEFF = 1e0
COLLIDE_COEFF = 4e1
RCOLLIDE_COEFF = 1e-1
MAX_CONTACT_DISTANCE = .1

GRASP_VALID_COEFF = 7.5e1
EEGRASP_VALID_COEFF = 1e2
# BASKET_OFFSET = 0.317
# BASKET_OFFSET = 0.325
BASKET_OFFSET = 0.33
BASKET_GRIP_OFFSET = 0.03
BASKET_SHALLOW_GRIP_OFFSET = 0.05

# How far to go into the washer
WASHER_DEPTH_OFFSET = 0

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
RESAMPLE_OPENED_DOOR_ROT = [np.pi/2, 0, 0]
RESAMPLE_CLOSED_DOOR_ROT = [5*np.pi/6, 0, 0]
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
joints = []

PRODUCTION = False
