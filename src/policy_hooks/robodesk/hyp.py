NUM_OBJS = 1
NUM_TARGS = 1

import numpy as np

from pma.robot_solver import RobotSolver
import policy_hooks.robodesk.desk_prob as prob
from policy_hooks.robodesk.robot_agent import RobotAgent
import policy_hooks.utils.policy_solver_utils as utils

def refresh_config(no=NUM_OBJS, nt=NUM_TARGS):
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    opts = prob.get_prim_choices()
    discr_opts = [opt for opt in opts if not np.isscalar(opts[opt])]
    cont_opts = [opt for opt in opts if np.isscalar(opts[opt])]

    config = {
        'opt_wt': 5e2,
        'solver_type': 'adam', #'rmsprop',
        'base_weight_dir': 'panda_',
        'task_map_file': prob.mapping_file,
        'prob': prob,
        'get_vector': prob.get_vector,
        'num_objs': no,
        'num_targs': nt,
        'agent_type': RobotAgent,
        'mp_solver_type': RobotSolver,
        'domain': 'panda',
        'share_buffer': True,
        'robot_name': 'panda',
        'ctrl_mode': 'joint_angle',
        'visual_cameras': [0],

        'state_include': [utils.STATE_ENUM],
        'obs_include': [utils.TASK_ENUM,
                        #utils.END_POSE_ENUM,
                        #utils.END_ROT_ENUM,
                        utils.RIGHT_ENUM,
                        #utils.RIGHT_VEL_ENUM,
                        utils.RIGHT_EE_POS_ENUM,
                        utils.RIGHT_GRIPPER_ENUM,
                        utils.GRIP_CMD_ENUM,
                        utils.OBJ_ENUM,
                        utils.TARG_ENUM,
                        utils.DOOR_ENUM,
                        ],
        'prim_obs_include': [
                             utils.ONEHOT_GOAL_ENUM,
                             utils.RIGHT_EE_POS_ENUM,
                             #utils.RIGHT_EE_ROT_ENUM,
                             utils.RIGHT_ENUM,
                             #utils.RIGHT_VEL_ENUM,
                             utils.RIGHT_GRIPPER_ENUM,
                             utils.GRIP_CMD_ENUM,
                             ],
        'prim_out_include': discr_opts,
        'cont_obs_include': [opt for opt in discr_opts],
        'sensor_dims': {
                utils.OBJ_POSE_ENUM: 3,
                utils.TARG_POSE_ENUM: 3,
                utils.EE_ENUM: 3,
                utils.RIGHT_EE_POS_ENUM: 3,
                utils.RIGHT_EE_ROT_ENUM: 3,
                utils.END_POSE_ENUM: 3,
                utils.ABS_POSE_ENUM: 3,
                utils.END_ROT_ENUM: 3,
                utils.TRUE_POSE_ENUM: 3,
                utils.TRUE_ROT_ENUM: 3,
                utils.GRIPPER_ENUM: 1,
                utils.GOAL_ENUM: 3*no,
                utils.ONEHOT_GOAL_ENUM: 12 + len(prob.GOAL_OPTIONS),
                utils.INGRASP_ENUM: no,
                utils.TRUETASK_ENUM: 2,
                utils.TRUEOBJ_ENUM: no,
                utils.ATGOAL_ENUM: no,
                utils.FACTOREDTASK_ENUM: len(list(prob.get_prim_choices().keys())),
                utils.RIGHT_ENUM: 7,
                utils.RIGHT_VEL_ENUM: 7,
                utils.RIGHT_GRIPPER_ENUM: 2,
                utils.GRIP_CMD_ENUM: 2,
                utils.QPOS_ENUM: 38,
            },
        'time_limit': 2**15,
        'curric_thresh': -1,
        'n_thresh': -1,
        'expand_process': False,
        'num_filters': [32, 32, 16],
        'filter_sizes': [7, 5, 3],
        'prim_filters': [16,16,16], # [16, 32],
        'prim_filter_sizes': [7,5,5], # [7, 5],
        'cont_filters': [32, 16],
        'cont_filter_sizes': [7, 5],
    }

    for o in range(no):
        config['sensor_dims'][utils.OBJ_DELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ENUMS[o]] = 3
    return config

config = refresh_config()

