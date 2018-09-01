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

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

N_RESAMPLES = 5
MAX_PRIORITY = 3
DEBUG=False

class NAMOPolicySolver(NAMOSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        super(NAMOPolicySolver, self).__init__(early_converge, transfer_norm)
        self.config = None
        self.gps = None
        self.transfer_coeff = 1e0
        self.weak_transfer_coeff = 1e-2
        self.rs_coeff = 2e2
        self.trajopt_coeff = 1e0
        self.policy_pred = None
        self.transfer_always = False
        self.policy_fs = {}

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
            self.config = namo_hyperparams.config if not hyperparams else hyperparams

        if hyperparams and self.config:
            self.config.update(hyperparams)

        conditions = self.config['num_conds']
        self.task_list = tuple(get_tasks('policy_hooks/namo/sorting_task_mapping').keys())
        self.task_durations = get_task_durations('policy_hooks/namo/sorting_task_mapping')
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
                plans[task, 'can{0}'.format(c)] = get_plan_for_task(task, ['can{0}'.format(c)], num_cans, env, openrave_bodies)
                if env is None:
                    env = plans[task, 'can{0}'.format(c)].env
                    for param in plans[task, 'can{0}'.format(c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body

        state_vector_include, action_vector_include, target_vector_include = get_vector(num_cans)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = utils.get_state_action_inds(plans.values()[0], 'pr2', ATTRMAP, state_vector_include, action_vector_include)

        self.target_dim, self.target_inds = utils.get_target_inds(plans.values()[0], ATTRMAP, target_vector_include)

        for i in range(conditions):
            targets.append(get_end_targets(num_cans))
        
        x0 = get_random_initial_state_vec(num_cans, targets, self.dX, self.state_inds, conditions)
        obj_list = ['can{0}'.format(c) for c in range(num_cans)]

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
        for p in self.agent.plans:
            task_ind = self.task_list.index(p[0])
            obj_ind = self.agent.obj_list.index(p[1])
            for targ_ind in range(len(targets[0])):
                self._add_policy_constraints_to_plan(self.agent.plans[p], task_ind, obj_ind, targ_ind)
        self.gps.run()
        env.Destroy()


    def _backtrack_solve(self, plan, callback=None, anum=0, verbose=False, amax=None, n_resamples=5, traj_mean=[], task=None):
        if amax is None:
            amax = len(plan.actions) - 1

        if anum > amax:
            return True

        a = plan.actions[anum]
        # print "backtracking Solve on {}".format(a.name)
        active_ts = a.active_timesteps
        inits = {}
        rs_param = self.get_resample_param(a)

        base_t = active_ts[0]
        if len(traj_mean):
            self.transfer_always = True
            for t in range(1, len(traj_mean)-1):
                for param_name, attr in plan.action_inds:
                    param = plan.params[param_name]
                    getattr(param, attr)[:, base_t+t] = traj_mean[t, plan.action_inds[param_name, attr]]
        else:
            self.transfer_always = False

        def recursive_solve():
            ## don't optimize over any params that are already set
            old_params_free = {}
            for p in plan.params.itervalues():
                if p.is_symbol():
                    if p not in a.params: continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in old_params_free[p].keys():
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, active_ts[1]].copy()
                        p._free_attrs[attr][:, active_ts[1]] = 0
            self.child_solver = self.__class__()
            success = self.child_solver._backtrack_solve(plan, callback=callback, anum=anum+1, verbose=verbose, amax=amax, traj_mean=traj_mean, task=task)

            # reset free_attrs
            for p in plan.params.itervalues():
                if p.is_symbol():
                    if p not in a.params: continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, active_ts[1]] = old_params_free[p][attr]
            return success

        # if there is no parameter to resample or some part of rs_param is fixed, then go ahead optimize over this action
        if rs_param is None or sum([not np.all(rs_param._free_attrs[attr]) for attr in rs_param._free_attrs.keys() ]):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = self.__class__()
            self.child_solver.dX = self.dX
            self.child_solver.dU = self.dU
            self.child_solver.symbolic_bound = self.symbolic_bound
            self.child_solver.state_inds = self.state_inds
            self.child_solver.action_inds = self.action_inds
            self.child_solver.agent = self.agent
            self.child_solver.policy_traj_coeff = self.policy_traj_coeff
            success = self.child_solver.solve(plan, callback=callback_a, n_resamples=n_resamples,
                                              active_ts=active_ts, verbose=verbose, force_init=True, 
                                              traj_mean=traj_mean, task=task)
            if not success:
                ## if planning fails we're done
                return False
            ## no other options, so just return here
            return recursive_solve()

        ## so that this won't be optimized over
        rs_free = rs_param._free_attrs
        rs_param._free_attrs = {}
        for attr in rs_free.keys():
            rs_param._free_attrs[attr] = np.zeros(rs_free[attr].shape)

        """
        sampler_begin
        """
        robot_poses = self.obj_pose_suggester(plan, anum, resample_size=3)

        """
        sampler end
        """

        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        for rp in robot_poses:
            for attr, val in rp.iteritems():
                setattr(rs_param, attr, val)

            success = False
            self.child_solver = self.__class__()
            success = self.child_solver.solve(plan, callback=callback_a, n_resamples=n_resamples,
                                              active_ts = active_ts, verbose=verbose,
                                              force_init=True, traj_mean=traj_mean, task=task)
            if success:
                if recursive_solve():
                    break
                else:
                    success = False

        rs_param._free_attrs = rs_free
        return success


    def solve(self, plan, callback=None, n_resamples=5, active_ts=None,
              verbose=False, force_init=False, traj_mean=[], task=None):
        success = False
        if callback is not None:
            viewer = callback()
        if force_init or not plan.initialized:
            self._solve_opt_prob(plan, priority=-2, callback=callback,
                active_ts=active_ts, verbose=verbose)
            # self._solve_opt_prob(plan, priority=-1, callback=callback,
            #     active_ts=active_ts, verbose=verbose)
            plan.initialized=True

        if success or len(plan.get_failed_preds(active_ts=active_ts)) == 0:
            return True

        for priority in self.solve_priorities:
            for attempt in range(n_resamples):
                ## refinement loop
                success = self._solve_opt_prob(plan, priority=priority,
                                callback=callback, active_ts=active_ts, verbose=verbose, 
                                traj_mean=traj_mean, task=task)

                try:
                    if DEBUG: plan.check_cnt_violation(active_ts=active_ts, priority=priority, tol=1e-3)
                except:
                    print "error in predicate checking"
                if success:
                    break

                self._solve_opt_prob(plan, priority=priority, callback=callback, active_ts=active_ts, 
                        verbose=verbose, resample=True, traj_mean=traj_mean, task=task)

                # if len(plan.get_failed_preds(active_ts=active_ts, tol=1e-3)) > 9:
                #     break

                # print "resample attempt: {}".format(attempt)

                try:
                    if DEBUG: plan.check_cnt_violation(active_ts = active_ts, priority = priority, tol = 1e-3)
                except:
                    print "error in predicate checking"

                assert not (success and not len(plan.get_failed_preds(active_ts = active_ts, priority = priority, tol = 1e-3)) == 0)

            if not success:
                return False

        return success


    def optimize_against_global(self, plan, a_start=0, a_end=-1, cond=0):
        a_num = a_start
        if a_end == -1:
            a_end = len(plan.actions) - 1
        success = True
        if a_start == a_end:
            raise Exception("This method requires at least two actions.")

        all_active_ts = (plan.actions[self.gps.agent.init_plan_states[cond][1][0]].active_timesteps[0],
                         plan.actions[self.gps.agent.init_plan_states[cond][1][1]].active_timesteps[1])

        T = all_active_ts[1] - all_active_ts[0] + 1

        # pol_sample = self.gps.agent.global_policy_samples[cond]
        # global_traj_mean = np.zeros((plan.dU, T))
        # for t in range(all_active_ts[0], all_active_ts[1]):
        #     global_traj_mean[:, t-all_active_ts[0]] = pol_sample.get_U((t-all_active_ts[0])*utils.POLICY_STEPS_PER_SECOND)
        # global_traj_mean[:, T-1] = pol_sample.get_U((T-1)*utils.POLICY_STEPS_PER_SECOND-1)
        global_traj_mean = []

        while a_num < a_end:
            # print "Constraining actions {0} and {1} against the global policy.\n".format(a_num, a_num+1)
            act_1 = plan.actions[a_num]
            act_2 = plan.actions[a_num+1]
            active_ts = (act_1.active_timesteps[0], act_2.active_timesteps[1])
            
            # save free_attrs
            # old_params_free = {}
            # for p in plan.params.itervalues():
            #     if p.is_symbol():
            #         if p in act_1.params or p in act_2.params: continue
            #         old_params_free[p] = p._free_attrs
            #         p._free_attrs = {}
            #         for attr in old_params_free[p].keys():
            #             p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
            #     else:
            #         p_attrs = {}
            #         old_params_free[p] = p_attrs
            #         for attr in p._free_attrs:
            #             p_attrs[attr] = [p._free_attrs[attr][:, :(active_ts[0])].copy(), p._free_attrs[attr][:, (active_ts[1])+1:].copy()]
            #             p._free_attrs[attr][:, (active_ts[1])+1:] = 0
            #             p._free_attrs[attr][:, :(active_ts[0])] = 0
            
            success = success and self._optimize_against_global(plan, (active_ts[0], active_ts[1]), n_resamples=1, global_traj_mean=global_traj_mean)
            
            # reset free_attrs
            # for p in plan.params.itervalues():
            #     if p.is_symbol():
            #         if p in act_1.params or p in act_2.params: continue
            #         p._free_attrs = old_params_free[p]
            #     else:
            #         for attr in p._free_attrs:
            #             p._free_attrs[attr][:, :(active_ts[0])] = old_params_free[p][attr][0]
            #             p._free_attrs[attr][:, (active_ts[1])+1:] = old_params_free[p][attr][1]

            # if not success:
            #     return success
            # print 'Actions: {} and {}'.format(plan.actions[a_num].name, plan.actions[a_num+1].name)
            a_num += 1

        return success

    def _optimize_against_global(self, plan, active_ts, n_resamples=1, global_traj_mean=[]):
        priority = 3
        for attempt in range(n_resamples):
            # refinement loop
            success = self._solve_policy_opt_prob(plan, global_traj_mean, active_ts=active_ts, resample=False)
            if success:
                break

            success = self._solve_policy_opt_prob(plan, global_traj_mean, active_ts=active_ts, resample=True)
        return success


    def _traj_policy_opt(self, plan, traj_mean, start_t, end_t):
        transfer_objs = []
        for param_name, attr_name in self.action_inds:
            param = plan.params[param_name]
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            T = end_t - start_t + 1
            attr_val = traj_mean[start_t:end_t+1, self.action_inds[(param_name, attr_name)]].T
            K = attr_type.dim

            KT = K*T
            v = -1 * np.ones((KT - K, 1))
            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
            # [:,0] allows numpy to see v and d as one-dimensional so
            # that numpy will create a diagonal matrix with v and d as a diagonal
            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
            # P = np.eye(KT)
            Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
            cur_val = attr_val.reshape((KT, 1), order='F')
            A = -2 * cur_val.T.dot(Q)
            b = cur_val.T.dot(Q.dot(cur_val))
            policy_traj_coeff = self.policy_traj_coeff / T

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(2*policy_traj_coeff*Q,
                                 policy_traj_coeff*A, policy_traj_coeff*b)
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)
        return transfer_objs


    def _policy_output_opt(self, plan, (i, j, k), start_t, end_t):
        transfer_objs = []
        if not hasattr(self, 'agent'):
            return []

        T = end_t-start_t+1
        pol_f = self.policy_fs[(i, j, k)]
        pol_out = np.zeros((T, self.dU))
        self.agent.T = T
        sample = Sample(self.agent)
        state = np.zeros((T, self.symbolic_bound))
        act = np.zeros((T, self.dU))
        for p_name, a_name in self.state_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            state[:, self.state_inds[p_name, attr_name]] = getattr(p, a_name)[:, start_t:end_t+1]
        for p_name, a_name in self.action_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            act[:, self.action_inds[p_name, attr_name]] = getattr(p, a_name)[:, start_t:end_t+1]
        hist_len = self.agent.hist_len
        target_vec = np.zeros(self.agent.target_vecs[0].shape)
        for p_name, a_name in self.target_inds:
            param = plan.params[p_name]
            target_vec[self.target_inds[p_name, a_name]] = getattr(param, a_name).flatten()
        for t in range(start_t, end_t+1):
            sample.set(STATE_ENUM, state[t], t-start_t)
            task_vec = np.zeros((len(self.agent.task_list)))
            task_vec[i] = 1
            obj_vec = np.zeros((len(self.agent.obj_list)))
            obj_vec[j] = 1
            targ_vec = np.zeros((len(self.agent.targ_list)))
            targ_vec[k] = 1
            traj_hist = np.zeros((hist_len, self.dU))
            for sub_t in range(t-hist_len, t):
                if sub_t < 0:
                    continue
                traj_hist[sub_t-t-hist_len, :] = act[sub_t]
            sample.set(TASK_ENUM, task_vec, t-start_t)
            sample.set(OBJ_ENUM, obj_vec, t-start_t)
            sample.set(TARG_ENUM, targ_vec, t-start_t)
            sample.set(TRAJ_HIST_ENUM, traj_hist.flatten(), t-start_t)
            sample.set(TARGETS_ENUM, target_vec, t-start_t)
            sample.set(ACTION_ENUM, act[t-start_t], t-start_t)
            pol_out[t-start_t] = pol_f(sample)

        for param_name, attr_name in self.action_inds:
            param = plan.params[param_name]
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            T = end_t - start_t + 1
            attr_val = pol_out[:, self.action_inds[(param_name, attr_name)]].T
            K = attr_type.dim

            KT = K*T
            v = -1 * np.ones((KT - K, 1))
            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
            # [:,0] allows numpy to see v and d as one-dimensional so
            # that numpy will create a diagonal matrix with v and d as a diagonal
            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
            # P = np.eye(KT)
            Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
            cur_val = attr_val.reshape((KT, 1), order='F')
            A = -2 * cur_val.T.dot(Q)
            b = cur_val.T.dot(Q.dot(cur_val))
            policy_out_coeff = self.policy_out_coeff / T

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(2*policy_out_coeff*Q,
                                 policy_out_coeff*A, policy_out_coeff*b)
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)
        return transfer_objs


    def _policy_func(self, sample, task):
        if self.gps.rollout_policies[task].scale is None:
            return sample.get_U(t=0)
        obs = sample.get_obs(t=0)
        return self.gps.rollout_policies[task].act(obs).flatten()


    def _add_policy_constraints_to_plan(self, plan, task_ind, obj_ind, targ_ind):
        for action in plan.actions:
            pred = NAMOPolicyPredicate('PolicyPred', plan, self._policy_func, self.config['policy_coeff'], self.agent, task_ind, obj_ind, targ_ind)
            start = action.active_timesteps[0] + self.agent.hist_len
            end = action.active_timesteps[1] - 1
            pred_dict = {'hl_info': 'policy', 'pred': pred, 'negated': False, 'active_timesteps': (start, end)}
            action.preds.append(pred_dict)


    def set_gps(self, gps):
        self.gps = gps
        self.agent = gps.agent
        self._store_policy_fs()


    def _store_policy_fs(self):
        if hasattr(self, 'agent'):
            for i in range(len(self.agent.task_list)):
                for j in range(len(self.agent.obj_list)):
                    for k in range(len(self.agent.targ_list)):
                        pol_f = lambda s: self._policy_func(s, self.agent.task_list[i])
                        self.policy_fs[(i, j, k)] = pol_f


    def _solve_opt_prob(self, plan, active_ts, priority, traj_mean=[], task=None, resample=False, callback=None, verbose=False):
        self.plan = plan
        plan.save_free_attrs()
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model)
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        initial_trust_region_size = self.initial_trust_region_size
        tol=1e-3

        if resample:
            obj_bexprs = []
            # if len(traj_mean):
            #     obj_bexprs.extend(self._traj_policy_opt(plan, traj_mean, active_ts[0], active_ts[1]))
            failed_preds = plan.get_failed_preds(active_ts = active_ts, priority=priority, tol = tol)
            rs_obj = self._resample(plan, failed_preds, sample_all = True)
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
            self._add_all_timesteps_of_actions(plan, priority=3,
                add_nonlin=True, active_ts=active_ts)
            # self._add_policy_preds(plan, active_ts)
            obj_bexprs.extend(rs_obj)
            if task is not None and task in self.policy_fs:
                obj_bexprs.extend(self._policy_output_opt(plan, task, active_ts[0], active_ts[1]))
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e3
        else:
            self._bexpr_to_pred = {}
            obj_bexprs = []
            if self.transfer_always:
                obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm, coeff=self.weak_transfer_coeff))
            # if priority >= 0 and len(traj_mean):
            #     obj_bexprs.extend(self._traj_policy_opt(plan, traj_mean, active_ts[0], active_ts[1]))
            # self.transfer_coeff *= 1e1
            # obj_bexprs = self._get_transfer_obj(plan, self.transfer_norm)
            # self.transfer_coeff *= 1e-1
            # self._add_obj_bexprs(obj_bexprs)
            # self._add_all_timesteps_of_actions(plan, priority=3, add_nonlin=True,
            #                                    active_ts=active_ts)
            # self._add_policy_preds(plan, active_ts)

            if priority == -2:
                """
                Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(plan,
                    priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose, add_nonlin=False)
                tol = 1e-3
                initial_trust_region_size = 1e3
            elif priority == -1:
                """
                Solve the optimization problem while enforcing every constraints.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(plan,
                    priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose,
                    add_nonlin=True)
                tol = 1e-3
            elif priority >= 0:
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                if task is not None and task in self.policy_fs:
                    obj_bexprs.extend(self._policy_output_opt(plan, task, active_ts[0], active_ts[1]))
                self._add_obj_bexprs(obj_bexprs)
                self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True,
                                                   active_ts=active_ts, verbose=verbose)
                tol=1e-3

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size

        solv.initial_penalty_coeff = self.init_penalty_coeff

        solv.max_merit_coeff_increases = self.max_merit_coeff_increases

        success = solv.solve(self._prob, method='penalty_sqp', tol=tol)
        self._update_ll_params()

        if resample:
            if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
                reward = 0
                if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
                    reward = len(plan.actions)
                else:
                    failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
                    for i in range(len(plan.actions)):
                        if failed_t > plan.actions[i].active_timesteps[1]:
                            reward += 1
                plan.sampling_trace[-1]['reward'] = reward
        ##Restore free_attrs values
        plan.restore_free_attrs()

        self.reset_variable()
        return success


    def _add_policy_preds(self, plan):
        for action in plan.actions:
            for pred_dict in action.preds:
                if pred_dict['hl_info'] == 'policy':
                    self._add_policy_pred_dict(pred_dict, effective_timesteps)


    def _add_policy_pred_dict(self, pred_dict, effective_timesteps):
        """
            This function creates constraints for the predicate and added to
            Prob class in sco.
        """
        start, end = pred_dict['active_timesteps']
        active_range = range(start, end+1)
        negated = pred_dict['negated']
        pred = pred_dict['pred']

        expr = pred.get_expr(negated)

        if expr is not None:
            for t in active_range:
                if t < effective_timesteps[0] or t > effective_timesteps[1]: continue
                var = self._spawn_sco_var_for_policy_pred(pred, t)
                bexpr = BoundExpr(expr, var)

                self._bexpr_to_pred[bexpr] = (negated, pred, t)
                groups = ['all']
                if self.early_converge:
                    ## this will check for convergence per parameter
                    ## this is good if e.g., a single trajectory quickly
                    ## gets stuck
                    groups.extend([param.name for param in pred.params])
                self._prob.add_cnt_expr(bexpr, groups)


    def _spawn_sco_var_for_policy_pred(self, pred, t):
        x = np.empty(pred.dU*(self.agent.hist_len+2), dtype=object)
        v = np.empty(pred.dU*(self.agent.hist_len+2))
        for i in range(t-pred.dU*self.agent.hist_len, t+2):
            for param in pred.params:
                for attr in const.ATTR_MAP[param._type]:
                    if (param.name, attr[0]) in pred.action_inds:
                        ll_p = self._param_to_ll[param]
                        x[t*pred.dU:(t+1)*pred.dU][pred.action_inds[(param.name, attr[0])]] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                        v[t*pred.dU:(t+1)*pred.dU][pred.action_inds[(param.name, attr[0])]] = getattr(param, attr[0])[attr[1], t]

        x = x.reshape((-1,1))
        v = v.reshape((-1,1))

        return Variable(x, v)


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
                                        config = copy_dict(namo_hyperparams.config)
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
                                        PS = NAMOPolicySolver()
                                        PS.train_policy(2, config)
