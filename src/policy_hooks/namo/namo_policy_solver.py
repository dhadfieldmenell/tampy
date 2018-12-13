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
# from policy_hooks.namo.multi_task_main import GPSMain
from policy_hooks.namo.vector_include import *
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.namo.namo_agent import NAMOSortingAgent
# import policy_hooks.namo.namo_hyperparams as namo_hyperparams
# import policy_hooks.namo.namo_optgps_hyperparams as namo_hyperparams
from policy_hooks.namo.namo_policy_predicates import NAMOPolicyPredicate
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.namo.sorting_prob_2 import *
from policy_hooks.task_net import tf_binary_network, tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

# N_RESAMPLES = 5
# MAX_PRIORITY = 3
# DEBUG=False

BASE_CLASS = get_base_solver(NAMOSolver)

class NAMOPolicySolver(BASE_CLASS):
    def _fill_sample(self, (i, j, k), start_t, end_t, plan):
        T = end_t - start_t + 1
        self.agent.T = T
        sample = self.fill_sample((i, j, k), start_t, end_t)
        sample = Sample(self.agent)
        state = np.zeros((self.symbolic_bound, T))
        act = np.zeros((self.dU, T))
        for p_name, a_name in self.state_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            state[self.state_inds[p_name, a_name], :] = getattr(p, a_name)[:, start_t:end_t+1]
        for p_name, a_name in self.action_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            x1 = getattr(p, a_name)[:, start_t:end_t]
            x2 = getattr(p, a_name)[:, start_t+1:end_t+1]
            act[self.action_inds[p_name, a_name], :-1] = x2 - x1
        hist_len = self.agent.hist_len
        target_vec = np.zeros(self.agent.target_vecs[0].shape)
        for p_name, a_name in self.target_inds:
            param = plan.params[p_name]
            target_vec[self.target_inds[p_name, a_name]] = getattr(param, a_name).flatten()
        for t in range(start_t, end_t+1):
            sample.set(STATE_ENUM, state[:, t-start_t], t-start_t)
            task_vec = np.zeros((len(self.agent.task_list)))
            task_vec[i] = 1
            obj_vec = np.zeros((len(self.agent.obj_list)))
            obj_vec[j] = 1
            targ_vec = np.zeros((len(self.agent.targ_list)))
            targ_vec[k] = 1
            traj_hist = np.zeros((hist_len, self.dU))
            for sub_t in range(t-hist_len, t):
                if sub_t < start_t:
                    continue
                traj_hist[sub_t-t+hist_len, :] = act[:, sub_t-start_t]
            sample.set(TASK_ENUM, task_vec, t-start_t)
            sample.set(OBJ_ENUM, obj_vec, t-start_t)
            sample.set(TARG_ENUM, targ_vec, t-start_t)
            ee_pose = state[:, t-start_t][plan.state_inds[self.robot_name, 'pose']]
            obj_pose = state[:, t-start_t][plan.state_inds[self.agent.obj_list[j], 'pose']] - ee_pose
            targ_pose = plan.params[self.agent.targ_list[k]].value[:,0] - ee_pose
            sample.set(EE_ENUM, ee_pose, t-start_t)
            sample.set(OBJ_POSE_ENUM, obj_pose, t-start_t)
            sample.set(TARG_POSE_ENUM, targ_pose, t-start_t)
            sample.set(TRAJ_HIST_ENUM, traj_hist.flatten(), t-start_t)
            sample.set(TARGETS_ENUM, target_vec, t-start_t)
            sample.set(ACTION_ENUM, act[:, t-start_t], t-start_t)
            if LIDAR_ENUM in self.agent._hyperparams['obs_include']:
                lidar = self.agent.dist_obs(plan, t)
                sample.set(LIDAR_ENUM, lidar.flatten(), t-start_t)
        return sample


    def train_policy(self, num_cans, hyperparams=None):
#         '''
#         Integrates the GPS code base with the TAMPy codebase to create a robust
#         system for combining motion planning with policy learning

#         Each plan must have the same state dimension and action diemensions as the others, and equivalent parameters in both (e..g same # of 
#         cans, same table dimensions, etc.)
#         '''
        pass

#         is_first_run = not self.config
#         if is_first_run:
#             self.config = namo_hyperparams.config if not hyperparams else hyperparams

#         if hyperparams and self.config:
#             self.config.update(hyperparams)

#         conditions = self.config['num_conds']
#         self.task_list = tuple(get_tasks('policy_hooks/namo/sorting_task_mapping_2').keys())
#         self.task_durations = get_task_durations('policy_hooks/namo/sorting_task_mapping_2')
#         self.config['task_list'] = self.task_list
#         task_encoding = get_task_encoding(self.task_list)

#         plans = {}
#         task_breaks = []
#         goal_states = []
#         targets = []

#         env = None
#         openrave_bodies = {}
#         for task in self.task_list:
#             for c in range(num_cans):
#                 plans[task, 'can{0}'.format(c)] = get_plan_for_task(task, ['can{0}'.format(c), 'can{0}_end_target'.format(c)], num_cans, env, openrave_bodies)
#                 if env is None:
#                     env = plans[task, 'can{0}'.format(c)].env
#                     for param in plans[task, 'can{0}'.format(c)].params.values():
#                         if not param.is_symbol():
#                             openrave_bodies[param.name] = param.openrave_body

#         state_vector_include, action_vector_include, target_vector_include = get_vector(num_cans)

#         self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], 'pr2', ATTRMAP, state_vector_include, action_vector_include)

#         self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], ATTRMAP, target_vector_include)

#         for i in range(conditions):
#             targets.append(get_end_targets(num_cans))
        
#         x0 = get_random_initial_state_vec(num_cans, targets, self.dX, self.state_inds, conditions)
#         obj_list = ['can{0}'.format(c) for c in range(num_cans)]

#         for plan in plans.values():
#             plan.state_inds = self.state_inds
#             plan.action_inds = self.action_inds
#             plan.dX = self.dX
#             plan.dU = self.dU
#             plan.symbolic_bound = self.symbolic_bound
#             plan.target_dim = self.target_dim
#             plan.target_inds = self.target_inds

#         sensor_dims = {
#             utils.STATE_ENUM: self.symbolic_bound,
#             utils.ACTION_ENUM: self.dU,
#             utils.TRAJ_HIST_ENUM: self.dU*self.config['hist_len'],
#             utils.TASK_ENUM: len(self.task_list),
#             utils.TARGETS_ENUM: self.target_dim,
#             utils.OBJ_ENUM: num_cans,
#             utils.TARG_ENUM: len(targets[0].keys()),
#             utils.OBJ_POSE_ENUM: 2,
#             utils.TARG_POSE_ENUM: 2,
#         }

#         self.config['plan_f'] = lambda task, targets: plans[task, targets[0].name] 
#         self.config['goal_f'] = goal_f
#         self.config['cost_f'] = cost_f
#         self.config['target_f'] = get_next_target
#         self.config['encode_f'] = sorting_state_encode
#         # self.config['weight_file'] = 'tf_saved/2018-09-12 23:43:45.748906_namo_5.ckpt'

#         self.config['task_durations'] = self.task_durations

#         self.policy_inf_coeff = self.config['algorithm']['policy_inf_coeff']
#         self.policy_out_coeff = self.config['algorithm']['policy_out_coeff']
#         if is_first_run:
#             self.config['agent'] = {
#                 'type': NAMOSortingAgent,
#                 'x0': x0,
#                 'targets': targets,
#                 'task_list': self.task_list,
#                 'plans': plans,
#                 'task_breaks': task_breaks,
#                 'task_encoding': task_encoding,
#                 'task_durations': self.task_durations,
#                 'state_inds': self.state_inds,
#                 'action_inds': self.action_inds,
#                 'target_inds': self.target_inds,
#                 'dU': self.dU,
#                 'dX': self.symbolic_bound,
#                 'symbolic_bound': self.symbolic_bound,
#                 'target_dim': self.target_dim,
#                 'get_plan': get_plan,
#                 'sensor_dims': sensor_dims,
#                 'state_include': [utils.STATE_ENUM],
#                 'obs_include': [utils.STATE_ENUM,
#                                 utils.TARGETS_ENUM,
#                                 utils.TASK_ENUM,
#                                 utils.OBJ_POSE_ENUM,
#                                 utils.TARG_POSE_ENUM],
#                 'prim_obs_include': [utils.STATE_ENUM,
#                                      utils.TARGETS_ENUM],
#                 'conditions': self.config['num_conds'],
#                 'solver': self,
#                 'num_cans': num_cans,
#                 'obj_list': obj_list,
#                 'stochastic_conditions': self.config['stochastic_conditions'],
#                 'image_width': utils.IM_W,
#                 'image_height': utils.IM_H,
#                 'image_channels': utils.IM_C,
#                 'hist_len': self.config['hist_len'],
#                 'T': 1,
#                 'viewer': None,
#                 'model': None,
#                 'get_hl_plan': hl_plan_for_state,
#                 'env': env,
#                 'openrave_bodies': openrave_bodies,
#             }

#         else:
#             # TODO: Fill in this case
#             self.config['agent']['conditions'] += self.config['num_conds']
#             self.config['agent']['x0'].extend(x0s)

#         self.config['algorithm']['dObj'] = sensor_dims[utils.OBJ_ENUM]
#         self.config['algorithm']['dTarg'] = sensor_dims[utils.TARG_ENUM]

#         # action_cost_wp = np.ones((self.config['agent']['T'], self.dU), dtype='float64')
#         state_cost_wp = np.ones((self.symbolic_bound), dtype='float64')
#         traj_cost = {
#                         'type': StateTrajCost,
#                         'data_types': {
#                             utils.STATE_ENUM: {
#                                 'wp': state_cost_wp,
#                                 'target_state': np.zeros((1, self.symbolic_bound)),
#                                 'wp_final_multiplier': 1.0,
#                             }
#                         },
#                         'ramp_option': RAMP_CONSTANT
#                     }
#         action_cost = {
#                         'type': ActionTrajCost,
#                         'data_types': {
#                             utils.ACTION_ENUM: {
#                                 'wp': np.ones((1, self.dU), dtype='float64'),
#                                 'target_state': np.zeros((1, self.dU)),
#                             }
#                         },
#                         'ramp_option': RAMP_CONSTANT
#                      }

#         # constr_cost = {
#         #                 'type': TrajConstrCost,
#         #               }

#         self.config['algorithm']['cost'] = {
#                                                 'type': CostSum,
#                                                 'costs': [traj_cost, action_cost],
#                                                 'weights': [1.0, 1.0],
#                                            }

#         # self.config['algorithm']['cost'] = constr_cost

#         self.config['dQ'] = self.dU
#         self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
#         self.config['algorithm']['init_traj_distr']['dt'] = 1.0

#         self.config['algorithm']['policy_opt'] = {
#             'type': MultiHeadPolicyOptTf,
#             'network_params': {
#                 'obs_include': self.config['agent']['obs_include'],
#                 'prim_obs_include': self.config['agent']['prim_obs_include'],
#                 # 'obs_vector_data': [utils.STATE_ENUM],
#                 'obs_image_data': [],
#                 'image_width': utils.IM_W,
#                 'image_height': utils.IM_H,
#                 'image_channels': utils.IM_C,
#                 'sensor_dims': sensor_dims,
#                 'n_layers': self.config['n_layers'],
#                 'num_filters': [5,10],
#                 'dim_hidden': self.config['dim_hidden'],
#             },
#             'distilled_network_params': {
#                 'obs_include': self.config['agent']['obs_include'],
#                 # 'obs_vector_data': [utils.STATE_ENUM],
#                 'obs_image_data': [],
#                 'image_width': utils.IM_W,
#                 'image_height': utils.IM_H,
#                 'image_channels': utils.IM_C,
#                 'sensor_dims': sensor_dims,
#                 'n_layers': 3,
#                 'num_filters': [5,10],
#                 'dim_hidden': [100, 100, 100]
#             },
#             'primitive_network_params': {
#                 'obs_include': self.config['agent']['obs_include'],
#                 # 'obs_vector_data': [utils.STATE_ENUM],
#                 'obs_image_data': [],
#                 'image_width': utils.IM_W,
#                 'image_height': utils.IM_H,
#                 'image_channels': utils.IM_C,
#                 'sensor_dims': sensor_dims,
#                 'n_layers': 2,
#                 'num_filters': [5,10],
#                 'dim_hidden': [40, 40],
#                 'output_boundaries': [len(self.task_list),
#                                       len(obj_list),
#                                       len(targets[0].keys())],
#                 'output_order': ['task', 'obj', 'targ'],
#             },
#             'value_network_params': {
#                 'obs_include': self.config['agent']['obs_include'],
#                 # 'obs_vector_data': [utils.STATE_ENUM],
#                 'obs_image_data': [],
#                 'image_width': utils.IM_W,
#                 'image_height': utils.IM_H,
#                 'image_channels': utils.IM_C,
#                 'sensor_dims': sensor_dims,
#                 'n_layers': 1,
#                 'num_filters': [5,10],
#                 'dim_hidden': [40]
#             },
#             'lr': self.config['lr'],
#             'network_model': tf_network,
#             'distilled_network_model': tf_network,
#             'primitive_network_model': tf_classification_network,
#             'value_network_model': tf_binary_network,
#             'iterations': self.config['train_iterations'],
#             'batch_size': self.config['batch_size'],
#             'weight_decay': self.config['weight_decay'],
#             'weights_file_prefix': EXP_DIR + 'policy',
#             'image_width': utils.IM_W,
#             'image_height': utils.IM_H,
#             'image_channels': utils.IM_C,
#             'task_list': self.task_list,
#             'gpu_fraction': 0.2,
#         }

#         alg_map = {}
#         for task in self.task_list:
#             self.config['algorithm']['T'] = self.task_durations[task]
#             alg_map[task] = self.config['algorithm']

#         self.config['algorithm'] = alg_map

#         gps = GPSMain(self.config)
#         self.set_gps(gps)
#         self.gps.run()
#         env.Destroy()


def copy_dict(d):
    new_d = {}
    for key in d:
        if type(d[key]) is d:
            new_d[key] = copy_dict(d[key])
        else:
            new_d[key] = d[key]
    return new_d

# if __name__ == '__main__':
#     for lr in [1e-4]:
#         for covard in [0]:
#             for wt in [1e0]:
#                 for klt in [1e0]:
#                     for kl in [1e-3]:
#                         for iters in [100000]:
#                             for dh in [[50, 50]]:
#                                 config = copy_dict(namo_hyperparams.config)
#                                 config['lr'] = lr
#                                 config['dim_hidden'] = dh
#                                 config['n_layers'] = len(dh)
#                                 config['train_iterations'] = iters
#                                 config['algorithm']['traj_opt']['covariance_damping'] = covard
#                                 config['opt_wt'] = wt
#                                 config['algorithm']['opt_wt'] = wt
#                                 config['algorithm']['traj_opt']['kl_threshold'] = klt
#                                 config['algorithm']['kl_step'] = kl
#                                 PS = NAMOPolicySolver()
#                                 PS.train_policy(5, config)
