import software_constants as const
import numpy as np

"""
This file contains non-robot specific constants.
This file is refferenced in:
    robot_predicates
    common_predicates
"""

USE_OPENRAVE = const.USE_OPENRAVE

"""
Constants used in robot_predicates
"""
# Needed
POSE_TOL = 1e-4
COLLISION_TOL = 1e-3
MAX_CONTACT_DISTANCE = .1
# BASKET_OFFSET = 0.317
# BASKET_OFFSET = 0.325
BASKET_OFFSET = 0.33
BASKET_NARROW_OFFSET = 0.215
BASKET_GRIP_OFFSET = 0
GRASP_DIST = 0.24
EE_STEP = 0.08

# EEReachable Constants
APPROACH_DIST = 0.025
RETREAT_DIST = 0.03
EEREACHABLE_STEPS = 6 # 4

# Collision Constants
DIST_SAFE = 1e-3
RCOLLIDES_DSAFE = 1e-3
COLLIDES_DSAFE = 1e-3

# Plan Coefficient

EEREACHABLE_COEFF = 5e-2
EEREACHABLE_ROT_COEFF = 5e-2
IN_GRIPPER_COEFF = 1e-1
IN_GRIPPER_ROT_COEFF = 1e0#2e0
WASHER_IN_GRIPPER_ROT_COEFF = 1e-2
GRIPPER_AT_COEFF = 1e-2
GRIPPER_AT_ROT_COEFF = 1e-2
EE_VALID_COEFF = 1e-5

OBSTRUCTS_COEFF = 1e0
COLLIDE_COEFF = 4e1
RCOLLIDE_COEFF = 1e-1
MAX_CONTACT_DISTANCE = .1

NEAR_GRIP_COEFF = 1e-2
NEAR_GRIP_ROT_COEFF = 4e-3
NEAR_APPROACH_COEFF = 1e-2
NEAR_APPROACH_ROT_COEFF = 4e-3
EEATXY_COEFF = 1e-2


GRASP_VALID_COEFF = 7.5e1
EEGRASP_VALID_COEFF = 1e2

# Gripper Value
GRIPPER_OPEN_VALUE = 0.02
GRIPPER_CLOSE_VALUE = 0.0
# BASKET_OFFSET = 0.317
# BASKET_OFFSET = 0.325
BASKET_OFFSET = 0.33
BASKET_GRIP_OFFSET = 0.03
BASKET_SHALLOW_GRIP_OFFSET = 0.05


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
Following are for relative positions on complex objects
"""
#DRAWER_HANDLE_POS = [0., -0.36, 0.01]
DRAWER_HANDLE_POS = [0., -0.33, -0.03]
IN_DRAWER_POS = [0., -0.4, 0.01]
DRAWER_HANDLE_ORN = [0., 0., 1.57]
IN_DRAWER_ORN = [0., 0., 0.]

SHELF_HANDLE_POS = [-0.3, -0.07, 1.005]
IN_SHELF_POS = [0.27, 0.15, 0.93]
SHELF_HANDLE_ORN = [1.57, 1.57, 0.]
IN_SHELF_ORN = [1.57, 1.57, 0.]


"""
Following constants are for testing purposes
"""
# Tolerance
TOL = 1e-3
# Predicate Gradient Test Option
TEST_GRAD = False

ATTRMAP = {"Rotation": [("value", np.array([0], dtype=np.int))],
            "Distance": [("value", np.array([0], dtype=np.int))],
            "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
            "Edge": (("pose", np.array([0,1,2], dtype=np.int)),
                     ("rotation", np.array([0,1,2], dtype=np.int)),
                     ("length", np.array([0], dtype=np.int))),
            "EEPose": (("value", np.array([0,1,2], dtype=np.int)),
                       ("rotation", np.array([0,1,2], dtype=np.int))),
            "Target": (("value", np.array([0,1,2], dtype=np.int)),
                       ("rotation", np.array([0,1,2], dtype=np.int))),
            "Table": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Obstacle": (("pose", np.array([0,1,2], dtype=np.int)),
                   ("rotation", np.array([0,1,2], dtype=np.int))),
            "Door": (("pose", np.array([0,1,2], dtype=np.int)),
                    ("hinge", np.array([0], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
            "Basket": (("pose", np.array([0,1,2], dtype=np.int)),
                       ("rotation", np.array([0,1,2], dtype=np.int))),
            "BasketTarget": (("value", np.array([0,1,2], dtype=np.int)),
                       ("rotation", np.array([0,1,2], dtype=np.int))),
            "Cloth": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Handle": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Sphere": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Box": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "ClothTarget": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "CanTarget": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "BoxTarget": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
            "Fabric": (("gripleft", np.array([0,1,2], dtype=np.int)),
                       ("gripright", np.array([0,1,2], dtype=np.int))),
            "Region": [("value", np.array([0,1], dtype=np.int))]}


