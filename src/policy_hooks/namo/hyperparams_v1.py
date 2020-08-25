

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
# from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.gui.config import generate_experiment_info

# from policy_hooks.algorithm_pigps import AlgorithmPIGPS
from policy_hooks.algorithm_impgps import AlgorithmIMPGPS
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.traj_opt_pi2 import TrajOptPI2
from core.util_classes.namo_predicates import ATTRMAP
from pma.namo_solver import NAMOSolver
from policy_hooks.namo.namo_agent import NAMOSortingAgent
from policy_hooks.namo.namo_policy_solver import NAMOPolicySolver
import policy_hooks.namo.sorting_prob_2 as prob
from policy_hooks.namo.namo_motion_plan_server import NAMOMotionPlanServer
from policy_hooks.policy_mp_prior_gmm import PolicyMPPriorGMM
from policy_hooks.policy_prior_gmm import PolicyPriorGMM

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

NUM_OBJS = prob.NUM_OBJS
NUM_CONDS = 5 # Per rollout server
NUM_PRETRAIN_STEPS = 20
NUM_PRETRAIN_TRAJ_OPT_STEPS = 1
NUM_TRAJ_OPT_STEPS = 1
N_SAMPLES = 15
N_TRAJ_CENTERS = 1
HL_TIMEOUT = 600
OPT_WT_MULT = 2e1
N_ROLLOUT_SERVERS = 10
N_OPTIMIZERS = 8
N_DIRS = 16


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
    'policy_inf_coeff': 1e4, #1e3,
    'policy_out_coeff': 1e2,
    'kl_step': 1e-3,
    'min_step_mult': 0.5,
    'max_step_mult': 5.0,
    'sample_ts_prob': 1.0,
    'opt_wt': OPT_WT_MULT,
    'fail_value': 50,
    'use_centroids': True,
    'n_traj_centers': N_TRAJ_CENTERS,
    'num_samples': N_SAMPLES,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.01,
    'pos_gains': 0.00,
}

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 1e0,
    'covariance_damping': 0.00,
    'min_temperature': 1e-3,
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
    'max_samples': 50,
}

algorithm['mp_policy_prior'] = {
    'type': PolicyMPPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 50,
}

cost_wp_mult = np.ones((3 + 2 * NUM_OBJS))
cost_wp_mult[2] = 2.

config = {
    'gui_on': False,
    'iterations': algorithm['iterations'],
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'algorithm': algorithm,
    'num_samples': algorithm['num_samples'],
    'num_distilled_samples': 0,
    'num_conds': NUM_CONDS,
    'mode': 'position',
    'stochastic_conditions': algorithm['stochastic_conditions'],
    'policy_coeff': 1e0,
    'sample_on_policy': True,
    'hist_len': 3,
    'take_optimal_sample': True,
    'num_rollouts': 10,
    'max_tree_depth': 3*NUM_OBJS,
    'branching_factor': 4,
    'opt_wt': algorithm['opt_wt'],
    'fail_value': algorithm['fail_value'],
    'lr': 2e-4,
    'solver_type': 'adam',
    'cost_wp_mult': cost_wp_mult,

    'train_iterations': 10000,
    'weight_decay': 2.5e-5,
    'batch_size': 100,
    'n_layers': 2,
    'dim_hidden': [128, 128],
    'n_traj_centers': algorithm['n_traj_centers'],
    'traj_opt_steps': NUM_TRAJ_OPT_STEPS,
    'pretrain_steps': NUM_PRETRAIN_STEPS,
    'pretrain_traj_opt_steps': NUM_PRETRAIN_TRAJ_OPT_STEPS,
    'on_policy': True,

    # New for multiprocess, transfer to sequential version as well.

    'n_optimizers': N_OPTIMIZERS,
    'n_rollout_servers': N_ROLLOUT_SERVERS,
    'base_weight_dir': 'namo_',
    'policy_out_coeff': algorithm['policy_out_coeff'],
    'policy_inf_coeff': algorithm['policy_inf_coeff'],
    'max_sample_queue': 5e2,
    'max_opt_sample_queue': 5e2,
    'hl_plan_for_state': prob.hl_plan_for_state,
    'task_map_file': 'policy_hooks/namo/sorting_task_mapping_2',
    'prob': prob,
    'get_vector': prob.get_vector,
    'robot_name': 'pr2',
    'obj_type': 'can',
    'num_objs': NUM_OBJS,
    'attr_map': ATTRMAP,
    'agent_type': NAMOSortingAgent,
    'opt_server_type': NAMOMotionPlanServer,
    'solver_type': NAMOPolicySolver,
    'update_size': 500,
    'use_local': True,
    'n_dirs': N_DIRS,
    'domain': 'namo',
    'perturb_steps': 3,
    'mcts_early_stop_prob': 0.5,
    'hl_timeout': HL_TIMEOUT,
    'multi_polciy': False,
    'image_width': 107,
    'image_height': 80,
    'image_channels': 3,

    'state_include': [utils.STATE_ENUM],
    'obs_include': [utils.LIDAR_ENUM,
                    utils.EE_ENUM,
                    utils.TASK_ENUM,
                    utils.OBJ_POSE_ENUM,
                    utils.TARG_POSE_ENUM],
    'prim_obs_include': [utils.STATE_ENUM,
                         utils.TARGETS_ENUM],
    'val_obs_include': [utils.STATE_ENUM,
                        utils.TARGETS_ENUM,
                        utils.TASK_ENUM,
                        utils.OBJ_ENUM,
                        utils.TARG_ENUM],
    'prim_out_include': [utils.TASK_ENUM, utils.OBJ_ENUM, utils.TARG_ENUM],
    'sensor_dims': {
            utils.OBJ_POSE_ENUM: 2,
            utils.TARG_POSE_ENUM: 2,
            utils.LIDAR_ENUM: N_DIRS,
            utils.EE_ENUM: 2,
        },
    'visual': False,
}
