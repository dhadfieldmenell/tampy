'''
A collection of enums to return upon completion of a call to solve a low-level
trajectory for a Baxter.
'''
SOLVER_SUCCESS = 0
STATE_INCONSISTENT = 1
LINEAR_CONSTRAINTS_UNSATISIFABLE = 2
EXCEEDED RESAMPLE_LIMIT = 3
POSE_PREDICTOR_FAILED = 4
BAXTER_STOPPED = 5 # Applies only to the physical robot, 
