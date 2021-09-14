

NUM_OBJS = 1
NUM_TARGS = 1

from datetime import datetime
import os
import os.path

import numpy as np

from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd

from policy_hooks.algorithm_impgps import AlgorithmIMPGPS
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.traj_opt_pi2 import TrajOptPI2
from core.util_classes.namo_grip_predicates import ATTRMAP
import policy_hooks.robodesk.desk_prob as prob
prob.NUM_OBJS = NUM_OBJS
prob.NUM_TARGS = NUM_TARGS
from policy_hooks.policy_mp_prior_gmm import PolicyMPPriorGMM
from policy_hooks.policy_prior_gmm import PolicyPriorGMM

from policy_hooks.robodesk.robot_agent import RobotAgent
from pma.robot_solver import RobotSolver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

NUM_CONDS = 1 # Per rollout server
NUM_PRETRAIN_STEPS = 20
NUM_PRETRAIN_TRAJ_OPT_STEPS = 1
NUM_TRAJ_OPT_STEPS = 1
N_SAMPLES = 10
N_TRAJ_CENTERS = 1
HL_TIMEOUT = 600
OPT_WT_MULT = 5e2
N_ROLLOUT_SERVERS = 34 # 58
N_ALG_SERVERS = 0
N_OPTIMIZERS = 0
N_DIRS = 16
N_GRASPS = 4
TIME_LIMIT = 14400


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': NUM_CONDS,
}

algorithm = {
    'type': AlgorithmIMPGPS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'add',
    'sample_on_policy': True,
    'iterations': 1e3, #5e4,
    'max_ent_traj': 0.0,
    'fit_dynamics': False,
    'stochastic_conditions': True,
    'policy_inf_coeff': 1e2,
    'policy_out_coeff': 1e1,
    'kl_step': 1.,
    'min_step_mult': 0.05,
    'max_step_mult': 5.0,
    'sample_ts_prob': 1.0,
    'opt_wt': OPT_WT_MULT,
    'fail_value': 50,
    'use_centroids': True,
    'n_traj_centers': N_TRAJ_CENTERS,
    'num_samples': N_SAMPLES,
    'mp_opt': True,
    'her': False,
    'rollout_opt': False,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.01,
    'pos_gains': 0.00,
}

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 1.,
    'covariance_damping': 0.00,
    'min_temperature': 1e-3,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 50,
}

algorithm['mp_policy_prior'] = {
    'type': PolicyMPPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 50,
}

def refresh_config(no=NUM_OBJS, nt=NUM_TARGS):
    prob.GOAL_OPTIONS = [
                #'(InSlideDoor upright_block shelf)',
                '(InSlideDoor upright_block drawer)',
                #'(InSlideDoor flat_block shelf)',
                #'(InSlideDoor flat_block drawer)',
                #'(InSlideDoor ball shelf)',
                #'(InSlideDoor ball drawer)',
                ]
    #prob.GOAL_OPTIONS = ['(Lifted ball panda)']
    prob.INVARIANT_GOALS = ['(SlideDoorClose shelf_handle shelf)', '(SlideDoorClose drawer_handle drawer)']
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    prob.N_GRASPS = N_GRASPS
    prob.FIX_TARGETS = True
    opts = prob.get_prim_choices()
    discr_opts = [opt for opt in opts if not np.isscalar(opts[opt])]
    cont_opts = [opt for opt in opts if np.isscalar(opts[opt])]

    prob.n_aux = 0
    config = {
        'iterations': algorithm['iterations'],
        'common': common,
        'algorithm': algorithm,
        'num_samples': algorithm['num_samples'],
        'num_conds': NUM_CONDS,
        'opt_wt': algorithm['opt_wt'],
        'solver_type': 'adam', #'rmsprop',
        'base_weight_dir': 'panda_',
        'policy_out_coeff': algorithm['policy_out_coeff'],
        'policy_inf_coeff': algorithm['policy_inf_coeff'],
        'max_sample_queue': 5e2,
        'max_opt_sample_queue': 10,
        'task_map_file': prob.mapping_file,
        'prob': prob,
        'get_vector': prob.get_vector,
        'num_objs': no,
        'num_targs': nt,
        'attr_map': ATTRMAP,
        'agent_type': RobotAgent,
        'mp_solver_type': RobotSolver,
        'll_solver_type': RobotSolver,
        'domain': 'panda',
        'share_buffer': True,
        'split_nets': False,
        'robot_name': 'panda',
        'ctrl_mode': 'joint_angle',
        'visual_cameras': [0],

        'state_include': [utils.STATE_ENUM],
        'obs_include': [utils.TASK_ENUM,
                        #utils.END_POSE_ENUM,
                        #utils.END_ROT_ENUM,
                        utils.RIGHT_ENUM,
                        utils.RIGHT_VEL_ENUM,
                        utils.RIGHT_EE_POS_ENUM,
                        utils.RIGHT_EE_ROT_ENUM,
                        utils.RIGHT_GRIPPER_ENUM,
                        utils.GRIP_CMD_ENUM,
                        utils.OBJ_ENUM,
                        utils.TARG_ENUM,
                        utils.DOOR_ENUM,
                        ],
        'prim_obs_include': [
                             utils.ONEHOT_GOAL_ENUM,
                             utils.RIGHT_EE_POS_ENUM,
                             utils.RIGHT_EE_ROT_ENUM,
                             #utils.RIGHT_ENUM,
                             #utils.RIGHT_VEL_ENUM,
                             #utils.RIGHT_GRIPPER_ENUM,
                             #utils.GRIP_CMD_ENUM,
                             ],
        'val_obs_include': [utils.ONEHOT_GOAL_ENUM,
                            ],
        'prim_out_include': discr_opts,
        'cont_obs_include': [opt for opt in discr_opts],
        'sensor_dims': {
                utils.OBJ_POSE_ENUM: 3,
                utils.TARG_POSE_ENUM: 3,
                utils.LIDAR_ENUM: N_DIRS,
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
                utils.ONEHOT_GOAL_ENUM: 12 + len(prob.GOAL_OPTIONS)+len(prob.INVARIANT_GOALS),
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
                # utils.INIT_OBJ_POSE_ENUM: 2,
            },
        'time_limit': TIME_LIMIT,
        'curric_thresh': -1,
        'n_thresh': -1,
        'expand_process': False,
        'her': False,
        'prim_filters': [32, 32],
        'prim_filter_sizes': [5, 5],
        'cont_filters': [32, 16],
        'cont_filter_sizes': [7, 5],
        'num_filters': [32, 32],
        'filter_sizes': [5, 5],
        'compound_goals': True,
        'max_goals': 4,
    }

    for o in range(len(opts[utils.OBJ_ENUM])):
        config['sensor_dims'][utils.OBJ_DELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ROTDELTA_ENUMS[o]] = 3
        config['sensor_dims'][utils.OBJ_ENUMS[o]] = 3
        config['sensor_dims'][utils.TARG_ENUMS[o]] = 3
        #config['prim_obs_include'].append(utils.OBJ_DELTA_ENUMS[o])
        #config['prim_obs_include'].append(utils.TARG_ENUMS[o])
        #config['prim_obs_include'].append(utils.OBJ_ROTDELTA_ENUMS[o])
        # config['prim_obs_include'].append(utils.TARG_ROTDELTA_ENUMS[o])
    return config

config = refresh_config()

