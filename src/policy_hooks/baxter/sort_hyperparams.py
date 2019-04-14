from __future__ import division

from datetime import datetime
import os
import os.path

import numpy as np

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
# from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
# from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.gui.config import generate_experiment_info

from core.util_classes.baxter_predicates import ATTRMAP
from pma.robot_ll_solver import RobotLLSolver
from policy_hooks.algorithm_impgps import AlgorithmIMPGPS
from policy_hooks.algorithm_tamp_gps import AlgorithmTAMPGPS
from policy_hooks.baxter.baxter_mjc_sorting_agent import BaxterMJCSortingAgent
from policy_hooks.baxter.sort_prob import *
import policy_hooks.baxter.sort_prob as prob
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.policy_prior_gmm import PolicyPriorGMM
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.traj_opt_pi2 import TrajOptPI2
from policy_hooks.policy_mp_prior_gmm import PolicyMPPriorGMM
from policy_hooks.baxter.sorting_motion_plan_server import SortingMotionPlanServer
from policy_hooks.baxter.baxter_policy_solver import BaxterPolicySolver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

N_OPTIMIZERS = 12
N_ROLLOUT_SERVERS = 24

NUM_CONDS = 10
NUM_OBJS = prob.NUM_CLOTHS
NUM_PRETRAIN_STEPS = 100
NUM_TRAJ_OPT_STEPS = 1
NUM_PRETRAIN_TRAJ_OPT_STEPS = 2
HL_TIMEOUT = 100
CLOTH_W = 5 
CLOTH_L = 3
IM_W = 64
IM_H = 64

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
    'iterations': 10,
    'max_ent_traj': 0.0,
    'fit_dynamics': False,
    'stochastic_conditions': True,
    'policy_inf_coeff': 1e1,
    'policy_out_coeff': 1e1,
    'kl_step': 1e-4,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'sample_ts_prob': 1.0,
    'opt_wt': 1e1,
    'fail_value': 5,
    'n_traj_centers': 1,
    'use_centroids': True
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.0004,
    'pos_gains': 0.0,
}

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2e-1,
    'covariance_damping': 1.0,
    'min_temperature': 0.001,
}

# algorithm['policy_prior'] = {
#     'type': PolicyPrior,
# }

# algorithm = {
#     'type': AlgorithmMDGPS,
#     'conditions': common['conditions'],
#     'iterations': 10,
#     'kl_step': 0.1,
#     'min_step_mult': 0.5,
#     'max_step_mult': 3.0,
#     'policy_sample_mode': 'replace',
# }

# algorithm['init_traj_distr'] = {
#     'type': init_pd,
#     'pos_gains':  1e-5,
# }

# algorithm['init_traj_distr'] = {
#     'type': init_lqr,
#     'init_var': 0.001,
#     'stiffness': 10.0,
#     'stiffness_vel': 0.5,
#     'final_weight': 5.0,
# }

# algorithm = {
#     'type': AlgorithmTrajOptPILQR,
#     'conditions': common['conditions'],
#     'iterations': 20,
#     'step_rule': 'res_percent',
#     'step_rule_res_ratio_dec': 0.2,
#     'step_rule_res_ratio_inc': 0.05,
#     'kl_step': np.linspace(0.6, 0.2, 100),
# }

# algorithm['dynamics'] = {
#     'type': DynamicsLRPrior,
#     'regularization': 1e-6,
#     'prior': {
#         'type': DynamicsPriorGMM,
#         'max_clusters': 20,
#         'min_samples_per_cluster': 60,
#         'max_samples': 30,
#     },
# }

# algorithm['traj_opt'] = {
#     'type': TrajOptPILQR,
# }

# algorithm['traj_opt'] = {
#     'type': TrajOptLQRPython,
# }

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 100,
}

algorithm['mp_policy_prior'] = {
    'type': PolicyMPPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 100,
}

config = {
    'gui_on': False,
    'iterations': algorithm['iterations'],
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'algorithm': algorithm,
    'num_samples': 15,
    'num_distilled_samples': 5,
    'num_conds': NUM_CONDS,
    'mode': 'add',
    'stochastic_conditions': algorithm['stochastic_conditions'],
    'policy_coeff': 1e0,
    'sample_on_policy': True,
    'hist_len': 3,
    'take_optimal_sample': True,
    'num_rollouts': 2,
    'max_tree_depth': 25,
    'branching_factor': 4,
    'opt_wt': algorithm['opt_wt'],
    'fail_value': algorithm['fail_value'],

    'train_iterations': 100000,
    'weight_decay': 0.0001,
    'batch_size': 1000,
    'n_layers': 2,
    'dim_hidden': [100, 100],
    'n_traj_centers': algorithm['n_traj_centers'],
    'traj_opt_steps': NUM_TRAJ_OPT_STEPS,
    'pretrain_steps': NUM_PRETRAIN_STEPS,
    'pretrain_traj_opt_steps': NUM_PRETRAIN_TRAJ_OPT_STEPS,
    'on_policy': True,

    # New for multiprocess, transfer to sequential version as well.

    'n_optimizers': N_OPTIMIZERS,
    'n_rollout_servers': N_ROLLOUT_SERVERS,
    'base_weight_dir': 'baxter_sort',
    'policy_out_coeff': algorithm['policy_out_coeff'],
    'policy_inf_coeff': algorithm['policy_inf_coeff'],
    'max_sample_queue': 2e2,
    'max_opt_sample_queue': 2e2,
    'task_map_file': 'policy_hooks/baxter/sort_task_mapping',
    'prob': prob,
    'get_vector': get_vector,
    'robot_name': 'baxter',
    'obj_type': 'cloth',
    'num_objs': NUM_OBJS,
    'attr_map': ATTRMAP,
    'agent_type': BaxterMJCSortingAgent,
    'opt_server_type': SortingMotionPlanServer,
    'solver_type': BaxterPolicySolver,
    'update_size': 1e3,
    'use_local': True,
    'n_dirs': 16,
    'domain': 'baxter',
    'perturb_steps': 3,
    'mcts_early_stop_prob': 0.5,
    'hl_timeout': HL_TIMEOUT,
    'multi_policy': False,
    'image_width': IM_W,
    'image_height': IM_H,
    'image_channels': 3,
    'lr': 1e-3,
    'opt_prob': 0.9,

    'cloth_width': CLOTH_W,
    'cloth_length': CLOTH_L,
    'cloth_spacing': 0.1,
    'cloth_radius': 0.01,

    'state_include': [utils.LEFT_EE_POS_ENUM,
                      utils.RIGHT_EE_POS_ENUM,
                      utils.STATE_ENUM],
    'obs_include': [utils.LEFT_EE_POS_ENUM,
                    utils.RIGHT_EE_POS_ENUM,
                    utils.TASK_ENUM,
                    utils.OBJ_POSE_ENUM,
                    utils.TARG_POSE_ENUM,],
                    # utils.LEFT_IMAGE_ENUM,
                    # utils.RIGHT_IMAGE_ENUM,
                    #utils.OVERHEAD_IMAGE_ENUM],
    'prim_obs_include': [utils.LEFT_EE_POS_ENUM, utils.RIGHT_EE_POS_ENUM, utils.STATE_ENUM], # [utils.OVERHEAD_IMAGE_ENUM],
    'val_obs_include': [utils.TASK_ENUM,
                        # utils.OVERHEAD_IMAGE_ENUM,
                        utils.OBJ_POSE_ENUM,
                        utils.TARG_POSE_ENUM,
                        utils.STATE_ENUM],
    'prim_out_include': [utils.OBJ_ENUM, utils.TARG_ENUM],

    'sensor_dims': {
            utils.LEFT_TARG_POSE_ENUM: 3,
            utils.RIGHT_TARG_POSE_ENUM: 3,
            utils.OBJ_POSE_ENUM: 3,
            utils.TARG_POSE_ENUM: 3,
            utils.RIGHT_EE_POS_ENUM: 3,
            utils.RIGHT_EE_QUAT_ENUM: 4,
            utils.LEFT_EE_POS_ENUM: 3,
            utils.LEFT_EE_QUAT_ENUM: 4,
            utils.LEFT_IMAGE_ENUM: IM_W*IM_H*3,
            utils.RIGHT_IMAGE_ENUM: IM_W*IM_H*3,
            utils.OVERHEAD_IMAGE_ENUM: IM_W*IM_H*3,
        }
}
