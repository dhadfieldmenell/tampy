""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior

import policy_hooks.policy_solver_utils as utils

common = {
    'conditions': 20,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 12,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'pos_gains':  1e-5,
}

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

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'iterations': 4000,
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
}
