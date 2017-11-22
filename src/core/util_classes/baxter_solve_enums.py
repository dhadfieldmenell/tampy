'''
A collection of enums to return upon completion of a call to solve a low-level
trajectory for a Baxter.
'''
SOLVER_SUCCESS = 0
STATE_INCONSISTENT = 1
LINEAR_CONSTRAINTS_UNSATISIFABLE = 2
EARLY_TERMINATION = 3
EXCEEDED_RESAMPLE_LIMIT = 4
POSE_PREDICTOR_FAILED = 5
BAXTER_STOPPED = 6 # Applies only to the physical robot
