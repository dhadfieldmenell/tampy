

from datetime import datetime
import os
import os.path

import numpy as np

# from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
# from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
# from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
# from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
# from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
# from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
# from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
# from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
# from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.gui.config import generate_experiment_info

# from policy_hooks.algorithm_pigps import AlgorithmPIGPS
# from policy_hooks.algorithm_tamp_gps import AlgorithmTAMPGPS
from policy_hooks.algorithm_optgps import AlgorithmOPTGPS
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.policy_prior_gmm import PolicyPriorGMM
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.traj_opt_pi2 import TrajOptPI2

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

NUM_CONDS = 15


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
    # 'type': AlgorithmTAMPGPS,
    'type': AlgorithmOPTGPS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'add',
    'sample_on_policy': True,
    'iterations': 20,
    'max_ent_traj': 0.0,
    'fit_dynamics': False,
    'stochastic_conditions': True,
    'policy_inf_coeff': 1e1,
    'policy_out_coeff': 1e1,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 5.0,
    'sample_ts_prob': 1.0,
    'opt_wt': 1e1,
    'fail_value': 50,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.0025,
    'pos_gains': 0.0,
}

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 1e-1,
    'covariance_damping': 0.01,
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
    'max_samples': 20,
}

config = {
    'gui_on': False,
    'iterations': algorithm['iterations'],
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'algorithm': algorithm,
    'num_samples': 5,
    'num_distilled_samples': 1,
    'num_conds': NUM_CONDS,
    'mode': 'position',
    'stochastic_conditions': algorithm['stochastic_conditions'],
    'policy_coeff': 1e0,
    'sample_on_policy': True,
    'hist_len': 3,
    'take_optimal_sample': True,
    'num_rollouts': 2,
    'max_tree_depth': 15,
    'branching_factor': 4,
    'opt_wt': algorithm['opt_wt'],
    'fail_value': algorithm['fail_value'],

    'train_iterations': 100000,
    'weight_decay': 0.00005,
    'batch_size': 1000,
    'n_layers': 3,
    'dim_hidden': [40, 40, 40],
    'traj_opt_steps': 1,
}
