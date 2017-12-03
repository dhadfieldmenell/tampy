from __future__ import division

from datetime import datetime
import os
import os.path

import numpy as np

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.gui.config import generate_experiment_info

from policy_hooks.algorithm_tamp_gps import AlgorithmTAMPGPS
import policy_hooks.policy_solver_utils as utils

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

NUM_CONDS = 10

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': NUM_CONDS,
}

# algorithm = {
#     'type': AlgorithmTAMPGPS,
#     # 'type': AlgorithmPIGPS,
#     'conditions': common['conditions'],
#     'policy_sample_mode': 'replace',
#     'sample_on_policy': False,
#     'iterations': 20,
#     'max_ent_traj': 0.0,
#     'fit_dynamics': False,
#     'stochastic_conditions': True,
#     'policy_transfer_coeff': 1e-1,
#     'policy_scale_factor': 1,
# }

# algorithm['init_traj_distr'] = {
#     'type': init_pd,
#     'init_var': 0.0004,
#     'pos_gains': 0.0,
# }

# algorithm['traj_opt'] = {
#     'type': TrajOptPI2,
#     'kl_threshold': 1.0,
#     'covariance_damping': 5.0,
#     'min_temperature': 0.0001,
# }

# algorithm['policy_prior'] = {
#     'type': PolicyPrior,
# }

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 10,
    'kl_step': 2.5,
    'min_step_mult': 0.5,
    # 'max_ent_traj': 0.005,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
}

# algorithm['init_traj_distr'] = {
#     'type': init_pd,
#     'pos_gains':  1e-5,
# }

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 10.0,
    'stiffness': 100.0,
    'stiffness_vel': 0.5,
    'final_weight': 1.0,
}

# algorithm = {
#     'type': AlgorithmTrajOptPILQR,
#     'conditions': common['conditions'],
#     'iterations': 20,
#     'step_rule': 'res_percent',
#     'step_rule_res_ratio_dec': 0.2,
#     'step_rule_res_ratio_inc': 0.05,
#     'kl_step': np.linspace(0.6, 0.2, 100),
# }

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

# algorithm['traj_opt'] = {
#     'type': TrajOptPILQR,
# }

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'cons_per_step': False
}

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
    'num_samples': 40,
    'num_conds': NUM_CONDS,
    'mode': 'position',
    'stochastic_conditions': False,
    'policy_coeff': 1e1
}
