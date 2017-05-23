"""
This file contains all pr2 specific constants
This file is refferenced in:
    pr2_predicates.py
    pr2_sampling.py
"""

"""
Follwing constants are used in pr2_predicates
"""
# Dimensional Constants, must be integer
BASE_DIM = 3
JOINT_DIM = 17
ROBOT_ATTR_DIM = 20
# Movement Constraints Constants
BASE_MOVE = 1
JOINT_MOVE_FACTOR = 10
TWOARMDIM = 16
# InGripper Constants
GRIPPER_OPEN_VALUE = 0.528
GRIPPER_CLOSE_VALUE = 0.5
# EEReachable Constants
APPROACH_DIST = 0.05
RETREAT_DIST = 0.075
EEREACHABLE_STEPS = 3
# Collision Constants
DIST_SAFE = 1e-2
COLLISION_TOL = 1e-3
RCOLLIDES_DSAFE = 5e-3
#Plan Coefficient
IN_GRIPPER_COEFF = 1e0
EEREACHABLE_COEFF = 1e0
EEREACHABLE_OPT_COEFF = 1.3e3
EEREACHABLE_ROT_OPT_COEFF = 3e2
INGRIPPER_OPT_COEFF = 3e2
RCOLLIDES_OPT_COEFF = 1e2
OBSTRUCTS_OPT_COEFF = 1e1
GRASP_VALID_COEFF = 1e1

TABLE_SAMPLING_RADIUS = 2.0
OBJ_RING_SAMPLING_RADIUS = .6

"""
Following constants are used in pr2 sampling
"""

DEFAULT_DIST = 0.6
NUM_BASE_RESAMPLES = 10
EE_ANGLE_SAMPLE_SIZE = 5

"""
Following constants are for testing purposes
"""
TOL = 1e-4
TEST_GRAD = False
