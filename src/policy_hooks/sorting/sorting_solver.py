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

import core.util_classes.baxter_constants as const
from  pma.robot_ll_solver import RobotLLSolver
from policy_hooks.multi_task_main import GPSMain
# from policy_hooks.cloth_exp_vec_include import *
from policy_hooks.cloth_exp_2_vec_include import *
from policy_hooks.cloth_world_policy_utils import *
from policy_hooks.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.multi_task_tamp_agent import LaundryWorldEEAgent
# from policy_hooks.multi_task_tamp_ee_agent import LaundryWorldEEAgent
from policy_hooks.pose_suggester import obj_pose_suggester
import policy_hooks.policy_hyperparams as baxter_hyperparams
from policy_hooks.policy_predicates import BaxterPolicyPredicate, BaxterPolicyEEPredicate
import policy_hooks.policy_solver_utils as utils
from policy_hooks.sorting_prob import *
from policy_hooks.tamp_agent import LaundryWorldClothAgent
from policy_hooks.tamp_cost import TAMPCost
from policy_hooks.cost_state import CostState
from policy_hooks.tamp_action_cost import CostAction
from policy_hooks.task_net import tf_classification_network


BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

N_RESAMPLES = 5

class BaxterPolicySolver(RobotLLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)
        self.config = None
        self.gps = None
        self.transfer_coeff = 1e0
        self.rs_coeff = 2e2
        self.trajopt_coeff = 1e0
        self.policy_pred = None

    # TODO: Add hooks for online policy learning
    def train_policy(self, num_cloths, hyperparams=None):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must have the same state dimension and action diemensions as the others, and equivalent parameters in both (e..g same # of
        cloths, same table dimensions, etc.)
        '''

        is_first_run = not self.config
        if is_first_run:
            self.config = baxter_hyperparams.config if not hyperparams else hyperparams

        if hyperparams and self.config:
            self.config.update(hyperparams)

        self.task_list = list(get_tasks('policy_hooks/sorting_task_mapping_2').keys())
        self.task_durations = get_task_durations('policy_hooks/sorting_task_mapping_2')
        self.config['task_list'] = self.task_list
        task_encoding = get_task_encoding(self.task_list)

        plans = []
        task_breaks = []
        color_maps = []
        goal_states = []
        for m in range(self.config['num_conds']):
            plan, task_break, color_map, goal_state = get_plan(num_cloths)
            plans.append(plan)
            task_breaks.append(task_break)
            color_maps.append(color_map)
            goal_states.append(goal_state)

        self.dX, self.state_inds, self.dU, self.action_inds, self.symbolic_bound = \
        utils.get_state_action_inds(plans[0], state_vector_include, action_vector_include)

        x0 = []
        for plan in plans:
            x0.append(np.zeros((self.symbolic_bound,)))
            utils.fill_vector([p for p in list(plan.params.values()) if not p.is_symbol()], self.state_inds, x0[-1], 0)


        sensor_dims = {
            utils.STATE_ENUM: self.symbolic_bound,
            utils.ACTION_ENUM: self.dU,
            utils.OBS_ENUM: utils.IM_H*utils.IM_W*utils.IM_C,
            # utils.EE_ENUM: 6,
            utils.TRAJ_HIST_ENUM: self.dU*self.config['hist_len'],
            utils.COLORS_ENUM: num_cloths,
            utils.TASK_ENUM: len(self.task_list)
        }

        self.policy_transfer_coeff = self.config['algorithm']['policy_transfer_coeff']
        if is_first_run:
            self.config['agent'] = {
                'type': LaundryWorldEEAgent,
                'x0': x0,
                'plans': plans,
                'task_breaks': task_breaks,
                'color_maps': color_maps,
                'task_encoding': task_encoding,
                'task_durations': self.task_durations,
                'state_inds': self.state_inds,
                'action_inds': self.action_inds,
                'dU': self.dU,
                'dX': self.symbolic_bound,
                'symbolic_bound': self.symbolic_bound,
                'get_plan': get_plan,
                'sensor_dims': sensor_dims,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [utils.STATE_ENUM, utils.TRAJ_HIST_ENUM],
                'conditions': self.config['num_conds'],
                'solver': self,
                'num_cloths': num_cloths,
                'stochastic_conditions': self.config['stochastic_conditions'],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'hist_len': self.config['hist_len'],
                'T': -1,
                'viewer': None,
                'model': None
            }
            self.config['algorithm']['cost'] = []

        else:
            # TODO: Fill in this case
            self.config['agent']['conditions'] += self.config['num_conds']
            self.config['agent']['x0'].extend(x0s)

        # action_cost_wp = np.ones((self.config['agent']['T'], self.dU), dtype='float64')
        state_cost_wp = np.ones((self.symbolic_bound), dtype='float64')
        state_cost_wp[self.state_inds['baxter', 'lArmPose']] *= 0.1
        state_cost_wp[self.state_inds['baxter', 'rArmPose']] *= 0.1
        state_cost_wp[self.state_inds['baxter', 'lGripper']] *= 0.1
        state_cost_wp[self.state_inds['baxter', 'rGripper']] *= 0.1
        # for (param_name, attr) in state_inds:
        #     if not initial_plan.params[param_name].is_symbol() and param_name.startswith('cloth'):
        #         state_cost_wp[:, state_inds[param_name, attr]] *= 4

        # state_cost_wp[:, initial_plan.state_inds['baxter', 'ee_right_pos']] *= 0.2
        # state_cost_wp[:, initial_plan.state_inds['baxter', 'ee_right_rot']] *= 0.2
        # state_cost_wp[:, initial_plan.state_inds['baxter', 'lGripper']] *= 10
        # state_cost_wp[:, initial_plan.state_inds['baxter', 'rGripper']] *= 10
        # action_cost_wp[:, initial_plan.action_inds['baxter', 'ee_right_pos']] *= 0.75
        # action_cost_wp[:, initial_plan.action_inds['baxter', 'ee_right_rot']] *= 0.75
        # action_cost_wp[:, initial_plan.action_inds['baxter', 'lGripper']] *= 10
        # action_cost_wp[:, initial_plan.action_inds['baxter', 'rGripper']] *= 10
        # state_cost_wp[-1, :] = 0
        # action_cost_wp[-1, :] = 0
        for cond in range(len(plans)):
            traj_cost = {
                            'type': CostState,
                            'data_types': {
                                utils.STATE_ENUM: {
                                    'wp': state_cost_wp,
                                    'target_state': np.zeros((1, self.symbolic_bound)),
                                    'wp_final_multiplier': 1.0,
                                }
                            },
                            'ramp_option': RAMP_LINEAR
                        }
            action_cost = {
                            'type': CostAction,
                            'data_types': {
                                utils.ACTION_ENUM: {
                                    'wp': np.ones((1, self.dU), dtype='float64'),
                                    'target_state': np.zeros((1, self.dU)),
                                }
                            },
                            'ramp_option': RAMP_CONSTANT
                         }

            self.config['algorithm']['cost'].append({
                                                        'type': CostSum,
                                                        'costs': [traj_cost, action_cost],
                                                        'weights': [1.0, 0.0],
                                                    })

        self.config['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = self.dU
        self.config['algorithm']['init_traj_distr']['dt'] = 1.0

        self.config['algorithm']['policy_opt'] = {
            'type': MultiHeadPolicyOptTf,
            'network_params': {
                'obs_include': [utils.STATE_ENUM],
                'obs_vector_data': [utils.STATE_ENUM],
                'obs_image_data': [],
                'image_width': utils.IM_W,
                'image_height': utils.IM_H,
                'image_channels': utils.IM_C,
                'sensor_dims': sensor_dims,
                'n_layers': 2,
                'num_filters': [5,10],
                'dim_hidden': [200, 200]
            },
            'lr': 1e-3,
            'network_model': tf_network,
            'primitive_network_model': tf_classification_network,
            'iterations': 10000,
            'batch_size': 100,
            'weight_decay': 0.05,
            'weights_file_prefix': EXP_DIR + 'policy',
            'image_width': utils.IM_W,
            'image_height': utils.IM_H,
            'image_channels': utils.IM_C,
            'task_list': self.task_list
        }

        alg_map = {}
        for task in self.task_list:
            alg_map[task] = self.config['algorithm']

        self.config['algorithm'] = alg_map

        if not self.gps:
            self.gps = GPSMain(self.config)
            for plan in plans:
                self._add_policy_constraints_to_plan(plan, list(state_vector_include.keys()))
        # else:
        #     # TODO: Handle this case
        #     self._update_agent(x0s)
        #     self._update_algorithm(self.config['algorithm']['cost'][-len(x0s):])

        self.gps.run()


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
            print("Constraining actions {0} and {1} against the global policy.\n".format(a_num, a_num+1))
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


    # def _solve_policy_opt_prob(self, plan, global_traj_mean, base_t, active_ts, resample):
    #     self.plan = plan
    #     priority = 3
    #     robot = plan.params['baxter']
    #     body = plan.env.GetRobot("baxter")
    #     plan.save_free_attrs()
    #     model = grb.Model()
    #     model.params.OutputFlag = 0
    #     self._prob = Prob(model)
    #     self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
    #     model.update()
    #     initial_trust_region_size = self.initial_trust_region_size
    #     tol=1e-3

    #     if resample:
    #         obj_bexprs = []
    #         failed_preds = plan.get_failed_preds(active_ts = active_ts, priority=priority, tol = tol)
    #         rs_obj = self._resample(plan, failed_preds, sample_all = True)
    #         obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
    #         obj_bexprs.extend(self._traj_policy_opt(plan, global_traj_mean, active_ts[0], active_ts[1], base_t))

    #         self._add_all_timesteps_of_actions(plan, priority=priority,
    #             add_nonlin=False, active_ts=active_ts)
    #         obj_bexprs.extend(rs_obj)
    #         self._add_obj_bexprs(obj_bexprs)
    #         initial_trust_region_size = 1e3
    #     else:
    #         self._bexpr_to_pred = {}
    #         obj_bexprs = self._traj_policy_opt(plan, global_traj_mean, active_ts[0], active_ts[1], base_t)
    #         obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
    #         # obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
    #         self._add_obj_bexprs(obj_bexprs)
    #         self._add_all_timesteps_of_actions(plan, priority=3, add_nonlin=True,
    #                                            active_ts=active_ts)

    #     solv = Solver()
    #     solv.initial_trust_region_size = initial_trust_region_size

    #     solv.initial_penalty_coeff = self.init_penalty_coeff

    #     solv.max_merit_coeff_increases = self.max_merit_coeff_increases

    #     success = solv.solve(self._prob, method='penalty_sqp', tol=tol)
    #     self._update_ll_params()

    #     if resample:
    #         if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
    #             reward = 0
    #             if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
    #                 reward = len(plan.actions)
    #             else:
    #                 failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
    #                 for i in range(len(plan.actions)):
    #                     if failed_t > plan.actions[i].active_timesteps[1]:
    #                         reward += 1
    #             plan.sampling_trace[-1]['reward'] = reward
    #     ##Restore free_attrs values
    #     plan.restore_free_attrs()

    #     self.reset_variable()
    #     return success


    def _traj_policy_opt(self, plan, traj_mean, start_t, end_t):
        transfer_objs = []
        base_t = plan.actions[self.gps.agent.init_plan_states[self.gps.agent.current_cond][1][0]].active_timesteps[0]
        for param_name, attr_name in list(self.action_inds.keys()):
            param = plan.params[param_name]
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            T = end_t - start_t + 1
            attr_val = traj_mean[self.action_inds[(param_name, attr_name)], start_t-base_t:end_t-base_t+1].T
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
            policy_transfer_coeff = self.policy_transfer_coeff / float(traj_mean.shape[1])

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(2*policy_transfer_coeff*Q,
                                 policy_transfer_coeff*A, policy_transfer_coeff*b)
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)
        return transfer_objs


    def _add_policy_constraints_to_plan(self, plan, param_names):
        params = [plan.params[name] for name in param_names if name in plan.params]
        state_inds = self.state_inds
        action_inds = self.action_inds
        dX = self.symbolic_bound
        dU = self.dU
        cur_task_ind = 0
        next_t, task = plan.task_breaks[cur_task_ind]
        for action in plan.actions:
            if action.active_timesteps[0] >= next_t:
                cur_task_ind += 1
                next_t, task = plan.task_breaks[cur_task_ind]

            policy_func = lambda x: self.gps.alg_map[task].policy_opt.task_map['policy'].act(x, x, 0, 0) # The global policy should be time independent
            # pred = BaxterPolicyEEPredicate('PolicyPred', params, state_inds, action_inds, policy_func, dX, dU, self.config['policy_coeff'])
            pred = BaxterPolicyPredicate('PolicyPred', params, state_inds, action_inds, policy_func, dX, dU, self.config['policy_coeff'])
            pred_dict = {'hl_info': 'policy', 'pred': pred, 'negated': False, 'active_timesteps': action.active_timesteps}
            action.preds.append(pred_dict)


    def _solve_policy_opt_prob(self, plan, global_traj_mean, active_ts, resample):
        self.plan = plan
        priority = 4
        robot = plan.params['baxter']
        body = plan.env.GetRobot("baxter")
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
            if len(global_traj_mean):
                obj_bexprs.extend(self._traj_policy_opt(plan, global_traj_mean, active_ts[0], active_ts[1]))
            failed_preds = plan.get_failed_preds(active_ts = active_ts, priority=priority, tol = tol)
            rs_obj = self._resample(plan, failed_preds, sample_all = True)
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
            self._add_all_timesteps_of_actions(plan, priority=3,
                add_nonlin=False, active_ts=active_ts)
            self._add_policy_preds(plan, active_ts)
            obj_bexprs.extend(rs_obj)
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e3
        else:
            self._bexpr_to_pred = {}
            obj_bexprs = []
            if len(global_traj_mean):
                obj_bexprs.extend(self._traj_policy_opt(plan, global_traj_mean, active_ts[0], active_ts[1]))
                self._add_obj_bexprs(obj_bexprs)
            # self.transfer_coeff *= 1e1
            # obj_bexprs = self._get_transfer_obj(plan, self.transfer_norm)
            # self.transfer_coeff *= 1e-1
            # self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=3, add_nonlin=True,
                                               active_ts=active_ts)
            self._add_policy_preds(plan, active_ts)

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


    def _add_policy_preds(self, plan, effective_timesteps):
        for action in plan.actions:
            if action.active_timesteps[1] <= effective_timesteps[0]: continue
            if action.active_timesteps[0] >= effective_timesteps[1]: continue
            for pred_dict in action.preds:
                if pred_dict['hl_info'] == 'policy':
                    self._add_policy_pred_dict(pred_dict, effective_timesteps)


    def _add_policy_pred_dict(self, pred_dict, effective_timesteps):
        """
            This function creates constraints for the predicate and added to
            Prob class in sco.
        """
        start, end = pred_dict['active_timesteps']
        active_range = list(range(start, end+1))
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
        x = np.empty(pred.dX, dtype=object)
        v = np.empty(pred.dX)
        for param in pred.params:
            for attr in const.ATTR_MAP[param._type]:
                if (param.name, attr[0]) in pred.state_inds and (param.name != 'baxter' or attr[0] == 'pose'):
                    ll_p = self._param_to_ll[param]
                    if (param.name, attr[0]) in pred.action_inds:
                        x[pred.state_inds[(param.name, attr[0])]] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                        v[pred.state_inds[(param.name, attr[0])]] = getattr(param, attr[0])[attr[1], t]
                    else:
                        inds = pred.state_inds[(param.name, attr[0])] - pred.act_offset
                        x[inds] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                        v[inds] = getattr(param, attr[0])[attr[1], t]
                if param.name == 'baxter' and attr[0] == 'lArmPose':
                    ll_p = self._param_to_ll[param]
                    x[-16:-9] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                    v[-16:-9] = getattr(param, attr[0])[attr[1], t]
                if param.name == 'baxter' and attr[0] == 'rArmPose':
                    ll_p = self._param_to_ll[param]
                    x[-8:-1] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                    v[-8:-1] = getattr(param, attr[0])[attr[1], t]
                if param.name == 'baxter' and attr[0] == 'lGripper':
                    ll_p = self._param_to_ll[param]
                    x[-9:-8] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                    v[-9:-8] = getattr(param, attr[0])[attr[1], t]
                if param.name == 'baxter' and attr[0] == 'rGripper':
                    ll_p = self._param_to_ll[param]
                    x[-1:] = getattr(ll_p, attr[0])[attr[1], t-self.ll_start]
                    v[-1:] = getattr(param, attr[0])[attr[1], t]

        x = x.reshape((-1,1))
        v = v.reshape((-1,1))

        return Variable(x, v)


if __name__ == '__main__':
    PS = BaxterPolicySolver()
    PS.train_policy(2)
