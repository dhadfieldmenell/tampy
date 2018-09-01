import os

import numpy as np

import gurobipy as grb

from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.prob import Prob
from sco.solver import Solver
from sco.variable import Variable

# from gps.gps_main import GPSMain
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
# from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *

from core.util_classes.namo_predicates import ATTRMAP
from pma.namo_solver import NAMOSolver
from policy_hooks.multi_task_main import GPSMain
from policy_hooks.namo.vector_include import *
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.namo.namo_agent import NAMOSortingAgent
import policy_hooks.namo.namo_hyperparams as namo_hyperparams
from policy_hooks.namo.namo_policy_predicates import NAMOPolicyPredicate
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.namo.sorting_prob import *
from policy_hooks.task_net import tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policY_solver import get_base_solver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

N_RESAMPLES = 5
MAX_PRIORITY = 3
DEBUG=False


# Dynamically determine the original MP solver to put the policy code on top of
BASE_CLASS = get_base_solver(RobotLLSolver)

class BaxterPolicySolver(BASE_CLASS):
    # TODO: Add hooks for online policy learning
    def train_policy(self, num_cans, hyperparams=None):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must have the same state dimension and action diemensions as the others, and equivalent parameters in both (e..g same # of 
        cans, same table dimensions, etc.)
        '''

        is_first_run = not self.config
        if is_first_run:
            self.config = baxter_hyperparams.config if not hyperparams else hyperparams

        if hyperparams and self.config:
            self.config.update(hyperparams)

        conditions = self.config['num_conds']
        self.task_list = tuple(get_tasks('policy_hooks/baxter/sorting_task_mapping').keys())
        self.task_durations = get_task_durations('policy_hooks/baxter/sorting_task_mapping')
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)

        plans = {}
        task_breaks = []
        goal_states = []
        targets = []

        env = None
        openrave_bodies = {}
        for task in self.task_list:
            for c in range(num_cans):
                plans[task, 'cloth{0}'.format(c)] = get_plan_for_task(task, ['cloth{0}'.format(c)], num_cans, env, openrave_bodies)
                if env is None:
                    env = plans[task, 'cloth{0}'.format(c)].env
                    for param in plans[task, 'cloth{0}'.format(c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body

        state_vector_include, action_vector_include, target_vector_include = get_vector(num_cans)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], 'baxter', ATTRMAP, state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], ATTRMAP, target_vector_include)

        for i in range(conditions):
            targets.append(get_end_targets(num_cans))
        
        x0 = get_random_initial_state_vec(num_cans, targets, self.dX, self.state_inds, conditions)
        obj_list = ['cloth{0}'.format(c) for c in range(num_cans)]

        for plan in plans.values():
            plan.state_inds = self.state_inds
            plan.action_inds = self.action_inds
            plan.dX = self.dX
            plan.dU = self.dU
            plan.symbolic_bound = self.symbolic_bound
            plan.target_dim = self.target_dim
            plan.target_inds = self.target_inds

        sensor_dims = {
            utils.STATE_ENUM: self.symbolic_bound,
            utils.ACTION_ENUM: self.dU,
            utils.TRAJ_HIST_ENUM: self.dU*self.config['hist_len'],
            utils.TASK_ENUM: len(self.task_list),
            utils.TARGETS_ENUM: self.target_dim,
            utils.OBJ_ENUM: num_cans,
            utils.TARG_ENUM: len(targets[0].keys()),
        }

        self.config['plan_f'] = lambda task, targets: plans[task, targets[0].name] 
        self.config['goal_f'] = goal_f
        self.config['cost_f'] = cost_f
        self.config['target_f'] = get_next_target
        self.config['encode_f'] = sorting_state_encode

        self.config['task_durations'] = self.task_durations

        self.policy_traj_coeff = self.config['algorithm']['policy_traj_coeff']
        self.policy_out_coeff = self.config['algorithm']['policy_out_coeff']
        if is_first_run:
            self.config['agent'] = {
                'type': NAMOSortingAgent,
                'x0': x0,
                'targets': targets,
                'task_list': self.task_list,
                'plans': plans,
                'task_breaks': task_breaks,
                'task_encoding': task_encoding,
                'task_durations': self.task_durations,
                'state_inds': self.state_inds,
                'action_inds': self.action_inds,
                'target_inds': self.target_inds,
                'dU': self.dU,
                'dX': self.symbolic_bound,
                'symbolic_bound': self.symbolic_bound,
                'target_dim': self.target_dim,
                'get_plan': get_plan,
                'sensor_dims': sensor_dims,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [utils.STATE_ENUM,
                                utils.TARGETS_ENUM,
                                utils.TASK_ENUM,
                                utils.OBJ_ENUM,
                                utils.TARG_ENUM,
                                utils.TRAJ_HIST_ENUM],
                'prim_obs_include': [utils.STATE_ENUM,
                                     utils.TARGETS_ENUM],
                'conditions': self.config['num_conds'],
                'solver': self,
                'num_cans': num_cans,
                'obj_list': obj_list,
                'stochastic_conditions': self.config['stochastic_conditions'],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'hist_len': self.config['hist_len'],
                'T': 1,
                'viewer': None,
                'model': None,
                'get_hl_plan': hl_plan_for_state,
                'env': env,
                'openrave_bodies': openrave_bodies,
            }

        else:
            # TODO: Fill in this case
            self.config['agent']['conditions'] += self.config['num_conds']
            self.config['agent']['x0'].extend(x0s)

        self.config['algorithm']['dObj'] = sensor_dims[utils.OBJ_ENUM]
        self.config['algorithm']['dTarg'] = sensor_dims[utils.TARG_ENUM]

        # action_cost_wp = np.ones((self.config['agent']['T'], self.dU), dtype='float64')
        state_cost_wp = np.ones((self.symbolic_bound), dtype='float64')
        traj_cost = {
                        'type': StateTrajCost,
                        'data_types': {
                            utils.STATE_ENUM: {
                                'wp': state_cost_wp,
                                'target_state': np.zeros((1, self.symbolic_bound)),
                                'wp_final_multiplier': 1.0,
                            }
                        },
                        'ramp_option': RAMP_CONSTANT
                    }
        action_cost = {
                        'type': ActionTrajCost,
                        'data_types': {
                            utils.ACTION_ENUM: {
                                'wp': np.ones((1, self.dU), dtype='float64'),
                                'target_state': np.zeros((1, self.dU)),
                            }
                        },
                        'ramp_option': RAMP_CONSTANT
                     }

        constr_cost = {
                        'type': TrajConstrCost,
                      }

        # self.config['algorithm']['cost'] = {
        #                                         'type': CostSum,
        #                                         'costs': [traj_cost, action_cost],
        #                                         'weights': [1.0, 1.0],
        #                                    }

        self.config['algorithm']['cost'] = constr_cost

        self.config['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dt'] = 1.0

        self.config['algorithm']['policy_opt'] = {
            'type': MultiHeadPolicyOptTf,
            'network_params': {
                'obs_include': self.config['agent']['obs_include'],
                'prim_obs_include': self.config['agent']['prim_obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': self.config['n_layers'],
                'num_filters': [5,10],
                'dim_hidden': self.config['dim_hidden'],
            },
            'distilled_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 3,
                'num_filters': [5,10],
                'dim_hidden': [100, 100, 100]
            },
            'primitive_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 2,
                'num_filters': [5,10],
                'dim_hidden': [40, 40],
                'output_boundaries': [len(self.task_list),
                                      len(obj_list),
                                      len(targets[0].keys())],
                'output_order': ['task', 'obj', 'targ'],
            },
            'value_network_params': {
                'obs_include': self.config['agent']['obs_include'],
                # 'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 1,
                'num_filters': [5,10],
                'dim_hidden': [50]
            },
            'lr': self.config['lr'],
            'network_model': tf_network,
            'distilled_network_model': tf_network,
            'primitive_network_model': tf_classification_network,
            'value_network_model': tf_network,
            'iterations': self.config['train_iterations'],
            'batch_size': self.config['batch_size'],
            'weight_decay': self.config['weight_decay'],
            'weights_file_prefix': EXP_DIR + 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list,
            'gpu_fraction': 0.2,
        }

        alg_map = {}
        for task in self.task_list:
            self.config['algorithm']['T'] = self.task_durations[task]
            alg_map[task] = self.config['algorithm']

        self.config['algorithm'] = alg_map

        gps = GPSMain(self.config)
        self.set_gps(gps)
        self.gps.run()
        env.Destroy()


def copy_dict(d):
    new_d = {}
    for key in d:
        if type(d[key]) is d:
            new_d[key] = copy_dict(d[key])
        else:
            new_d[key] = d[key]
    return new_d

if __name__ == '__main__':
    for lr in [1e-3]:
        for init_var in [0.001]:
            for covard in [2]:
                for wt in [1e2]:
                    for klt in [1e-2]:
                        for kl in [1e-1]:
                            for iters in [100000]:
                                for dh in [[50, 50]]:
                                    for hl in [3]:
                                        config = copy_dict(baxter_hyperparams.config)
                                        config['lr'] = lr
                                        config['dim_hidden'] = dh
                                        config['n_layers'] = len(dh)
                                        config['train_iterations'] = iters
                                        config['algorithm']['init_traj_distr']['init_var'] = init_var
                                        config['algorithm']['traj_opt']['covariance_damping'] = covard
                                        config['opt_wt'] = wt
                                        config['algorithm']['opt_wt'] = wt
                                        config['algorithm']['traj_opt']['kl_threshold'] = klt
                                        config['algorithm']['kl_step'] = kl
                                        config['hist_len'] = hl
                                        PS = BaxterPolicySolver()
                                        PS.train_policy(2, config)
