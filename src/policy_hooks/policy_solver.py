import os
import time

import numpy as np

import gurobipy as grb

from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.prob import Prob
from sco.solver import Solver
from sco.variable import Variable

from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
# from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *

from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.policy_solver_utils import *
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

def get_base_solver(parent_class):
    class PolicySolver(parent_class):
        def __init__(self, hyperparams=None, early_converge=False, transfer_norm='min-vel'):
            super(PolicySolver, self).__init__(early_converge, transfer_norm)
            self.config = None
            self.gps = None
            self.transfer_coeff = 1e0
            self.strong_transfer_coeff = 1e2
            self.rs_coeff = 2e2
            self.trajopt_coeff = 1e3 # 1e2
            self.policy_pred = None
            self.transfer_always = False
            self.gps = None
            self.agent = None
            self.policy_fs = {}
            self.policy_inf_fs = {}

            self.hyperparams = hyperparams
            if hyperparams != None:
                self.dX = hyperparams['dX']
                self.dU = hyperparams['dU']
                self.symbolic_bound = hyperparams['symbolic_bound']
                self.state_inds = hyperparams['state_inds']
                self.action_inds = hyperparams['action_inds']
                self.policy_out_coeff = hyperparams['policy_out_coeff']
                self.policy_inf_coeff = hyperparams['policy_inf_coeff']
                self.target_inds = hyperparams['target_inds']
                self.target_dim = hyperparams['target_dim']
                self.task_list = hyperparams['task_list']
                self.robot_name = hyperparams['robot_name']


        # TODO: Add hooks for online policy learning
        def train_policy(self, num_cans, hyperparams=None):
            raise NotImplementedError


        def _backtrack_solve(self, plan, callback=None, anum=0, verbose=False, amax=None, n_resamples=5, inf_f=None, traj_mean=[], task=None, start_time=None):
            if amax is None:
                amax = len(plan.actions) - 1

            if anum > amax:
                return True

            if start_time is None:
                start_time = time.time()
            a = plan.actions[anum]
            print "backtracking Solve on {}".format(a.name)
            active_ts = a.active_timesteps
            inits = {}
            rs_param = self.get_resample_param(a)

            base_t = active_ts[0]
            if len(traj_mean):
                self.transfer_always = True
                for t in range(base_t, min(active_ts[1], len(traj_mean))):
                    for param_name, attr in plan.state_inds:
                        param = plan.params[param_name]
                        if hasattr(param, attr):
                            if param.is_symbol(): continue
                            getattr(param, attr)[:, base_t+1] = traj_mean[t, plan.state_inds[param_name, attr]]
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
                self.child_solver = self.__class__(self.hyperparams)
                self.child_solver.dX = self.dX
                self.child_solver.dU = self.dU
                self.child_solver.symbolic_bound = self.symbolic_bound
                self.child_solver.state_inds = self.state_inds
                self.child_solver.action_inds = self.action_inds
                self.child_solver.agent = self.agent
                self.child_solver.policy_out_coeff = self.policy_out_coeff
                self.child_solver.policy_inf_coeff = self.policy_inf_coeff
                self.child_solver.gps = self.gps
                self.child_solver.policy_fs = self.policy_fs
                self.child_solver.policy_inf_fs = self.policy_inf_fs
                self.child_solver.target_inds = self.target_inds
                self.child_solver.target_dim = self.target_dim
                self.child_solver.robot_name = self.robot_name
                success = self.child_solver._backtrack_solve(plan, callback=callback, anum=anum+1, verbose=verbose, amax=amax, 
                                                             n_resamples=n_resamples, inf_f=inf_f, traj_mean=traj_mean, task=task)

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
                self.child_solver = self.__class__(self.hyperparams)
                self.child_solver.dX = self.dX
                self.child_solver.dU = self.dU
                self.child_solver.symbolic_bound = self.symbolic_bound
                self.child_solver.state_inds = self.state_inds
                self.child_solver.action_inds = self.action_inds
                self.child_solver.agent = self.agent
                self.child_solver.policy_out_coeff = self.policy_out_coeff
                self.child_solver.policy_inf_coeff = self.policy_inf_coeff
                self.child_solver.gps = self.gps
                self.child_solver.policy_fs = self.policy_fs
                self.child_solver.policy_inf_fs = self.policy_inf_fs
                self.child_solver.target_inds = self.target_inds
                self.child_solver.target_dim = self.target_dim
                self.child_solver.robot_name = self.robot_name
                success = self.child_solver.solve(plan, callback=callback_a, n_resamples=n_resamples,
                                                  active_ts=active_ts, verbose=verbose, force_init=True, 
                                                  inf_f=inf_f, traj_mean=traj_mean, task=task)
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
            robot_poses = self.obj_pose_suggester(plan, anum, resample_size=1)

            """
            sampler end
            """

            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None

            success = False
            for rp in robot_poses:
                for attr, val in rp.iteritems():
                    setattr(rs_param, attr, val)

                success = False
                self.child_solver = self.__class__(self.hyperparams)
                self.child_solver.dX = self.dX
                self.child_solver.dU = self.dU
                self.child_solver.symbolic_bound = self.symbolic_bound
                self.child_solver.state_inds = self.state_inds
                self.child_solver.action_inds = self.action_inds
                self.child_solver.agent = self.agent
                self.child_solver.policy_out_coeff = self.policy_out_coeff
                self.child_solver.policy_inf_coeff = self.policy_inf_coeff
                self.child_solver.gps = self.gps
                self.child_solver.policy_fs = self.policy_fs
                self.child_solver.policy_inf_fs = self.policy_inf_fs
                self.child_solver.target_inds = self.target_inds
                self.child_solver.target_dim = self.target_dim
                self.child_solver.robot_name = self.robot_name
                success = self.child_solver.solve(plan, callback=callback_a, n_resamples=n_resamples,
                                                  active_ts = active_ts, verbose=verbose,
                                                  force_init=True, inf_f=inf_f, traj_mean=traj_mean, task=task)
                if success:
                    if recursive_solve():
                        break
                    else:
                        success = False

            rs_param._free_attrs = rs_free
            return success


        def solve(self, plan, callback=None, n_resamples=5, active_ts=None,
                  verbose=False, force_init=False, inf_f=None, traj_mean=[], task=None):
            success = False
            if callback is not None:
                viewer = callback()
            if force_init or not plan.initialized:
                self._solve_opt_prob(plan, priority=-2, callback=callback,
                    active_ts=active_ts, traj_mean=traj_mean, verbose=verbose, task=task)
                # self._solve_opt_prob(plan, priority=-1, callback=callback,
                #     active_ts=active_ts, verbose=verbose)
                plan.initialized=True

            if success or len(plan.get_failed_preds(active_ts=active_ts, tol=1e-3)) == 0:
                return True

            for priority in self.solve_priorities:
                for attempt in range(n_resamples):
                    ## refinement loop
                    success = self._solve_opt_prob(plan, priority=priority,
                                    callback=callback, active_ts=active_ts, verbose=verbose, 
                                    inf_f=inf_f, traj_mean=traj_mean, task=task)

                    try:
                        if DEBUG: plan.check_cnt_violation(active_ts=active_ts, priority=priority, tol=1e-3)
                    except:
                        print "error in predicate checking"
                    if success:
                        break

                    self._solve_opt_prob(plan, priority=priority, callback=callback, active_ts=active_ts, 
                            verbose=verbose, resample=True, inf_f=inf_f, traj_mean=traj_mean, task=task)

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


        def _policy_inference_obj(self, plan, task, start_t, end_t, inf_f):
            transfer_objs = []

            T = end_t-start_t+1
            sample = Sample(self.agent)
            state = np.zeros((self.symbolic_bound, T))
            for p_name, a_name in self.state_inds:
                p = plan.params[p_name]
                if p.is_symbol(): continue
                state[self.state_inds[p_name, a_name], :] = getattr(p, a_name)[:, start_t:end_t+1]

            sample.set(STATE_ENUM, state.T)
            # for t in range(0, T):
            #     sample = self.agent.fill_sample(0, sample, state[:,t], t, task)

            mu, sig, use_state, use_action = inf_f(sample)
            def attr_moments(param, attr_name, inds, sample):
                attr_mu = mu[:, inds]
                attr_sig = sig[:, inds][:, :, inds]
                active_ts = (start_t, end_t)
                if not hasattr(param, attr_name):
                    attr_name, attr_mu, attr_sig = self.convert_attrs(attr_name, attr_mu, attr_sig, param, active_ts, sample)
                prec = np.linalg.inv(attr_sig)
                attr_type = param.get_attr_type(attr_name)
                return attr_name, attr_type.dim, attr_attr_mu, attr_sig, prec

            def inf_expr(Q, v):
                return QuadExpr(coeff*Q, -2*coeff*v.T.dot(Q), coeff*v.T.dot(Q).dot(v))

            if use_action:
                for param_name, attr_name in self.action_inds:
                    act_inds = self.action_inds[(param_name, attr_name)]
                    start_ind = sample.dX if use_state else 0
                    param = plan.params[param_name]
                    param_ll = self._param_to_ll[param]

                    attr_name, K, attr_mu, attr_sig, prec = attr_moments(param, attr_name, start_ind+act_inds, sample)
                    attr_val = getattr(param, attr_name)[:, start_t:end_t+1]
                    attr_val[:, :-1] -= attr_val[:, 1:]

                    KT = K*T

                    Q = np.zeros(KT, KT)
                    v = attr_mu.reshape((KT, 1))
                    for t in range(T-1):
                        Q[t:t+1,t:t+1] = prec[t]

                    coeff = self.policy_inf_coeff / T
                    quad_expr = inf_expr(Q, v)

                    bexpr = self.gen_bexpr(param_ll, attr_name, KT, quad_expr)
                    transfer_objs.append(bexpr)

            if use_state:
                for param_name, attr_name in self.state_inds:
                    state_inds = self.state_inds[param_name, attr_name]
                    param = plan.params[param_name]
                    param_ll = self._param_to_ll[param]

                    attr_name, K, attr_mu, attr_sig, prec = attr_moments(param, attr_name, state_inds)
                    attr_val = getattr(param, attr_name)[:, start_t:end_t+1]

                    KT = K*T

                    Q = np.zeros(KT, KT)
                    v = attr_mu.reshape((KT, 1))
                    for t in range(T):
                        Q[t:t+1,t:t+1] = prec[t]

                    coeff = self.policy_inf_coeff / T
                    quad_expr = inf_Expr(Q, v)

                    bexpr = self.gen_bexpr(param_ll, attr_name, KT, quad_expr)
                    transfer_objs.append(bexpr)

            return transfer_objs


        def gen_bexpr(self, param_ll, attr_name, KT, expr):
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            cur_val = attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            return BoundExpr(quad_expr, sco_var)


        def _policy_inf_func(self, obs, x0, task):
            mu, sig, prec, det_sig = self.policy_opt.traj_prob(np.array([obs]), task)
            for p_name, a_name in self.action_inds:
                mu[:, self.action_inds[p_name, a_name]] += x0[self.state_inds[p_name, a_name]]
            return mu, sig, prec, det_sig


        def set_gps(self, gps):
            self.agent = gps.agent
            self._store_policy_fs()


        # def _store_policy_fs(self):
        #     if hasattr(self, 'agent'):
        #         for i in range(len(self.agent.task_list)):
        #             for j in range(len(self.agent.targ_list)):
        #                 pol_inf_f = lambda o, x: self._policy_inf_func(o, x, self.agent.task_list[i])
        #                 self.policy_inf_fs[(i, j)] = pol_inf_f
        #     else:
        #         print '{0} has no attribute agent; not setting policy functions.'.format(self.__class__)


        def _solve_opt_prob(self, plan, active_ts, priority, inf_f=None, traj_mean=[], task=None, resample=False, callback=None, verbose=False):
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
                failed_preds = plan.get_failed_preds(active_ts = active_ts, priority=priority, tol = tol)
                rs_obj = self._resample(plan, failed_preds, sample_all = True)
                obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
                self._add_all_timesteps_of_actions(plan, priority=3,
                    add_nonlin=True, active_ts=active_ts)
                obj_bexprs.extend(rs_obj)
                if inf_f is not None:
                    obj_bexprs.extend(self._policy_inference_obj(plan, task, active_ts[0], active_ts[1]), inf_F)
                self._add_obj_bexprs(obj_bexprs)
                initial_trust_region_size = 1e3
            else:
                self._bexpr_to_pred = {}
                obj_bexprs = []

                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                if self.transfer_always:
                    obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm, coeff=self.strong_transfer_coeff))
                if task is not None:
                    obj_bexprs.extend(self._policy_inference_obj(plan, task, active_ts[0], active_ts[1]))

                if priority == -2:
                    """
                    Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
                    """
                    self._add_obj_bexprs(obj_bexprs)
                    self._add_first_and_last_timesteps_of_actions(plan,
                        priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose, add_nonlin=False)
                    tol = 1e-3
                    initial_trust_region_size = 1e3
                elif priority == -1:
                    """
                    Solve the optimization problem while enforcing every constraints.
                    """
                    self._add_obj_bexprs(obj_bexprs)
                    self._add_first_and_last_timesteps_of_actions(plan,
                        priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose,
                        add_nonlin=True)
                    tol = 1e-3
                elif priority >= 0:
                    self._add_obj_bexprs(obj_bexprs)
                    self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True,
                                                       active_ts=active_ts, verbose=verbose)
                    tol=1e-3

            solv = Solver()
            solv.initial_trust_region_size = initial_trust_region_size

            solv.initial_penalty_coeff = self.init_penalty_coeff

            solv.max_merit_coeff_increases = self.max_merit_coeff_increases

            success = solv.solve(self._prob, method='penalty_sqp', tol=tol)
            if priority == MAX_PRIORITY:
                success = len(plan.get_failed_preds(tol=tol, active_ts=active_ts, priority=priority)) == 0
            self._update_ll_params()

            # if resample:
            #     if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
            #         reward = 0
            #         if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
            #             reward = len(plan.actions)
            #         else:
            #             failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
            #             for i in range(len(plan.actions)):
            #                 if failed_t > plan.actions[i].active_timesteps[1]:
            #                     reward += 1
            #         plan.sampling_trace[-1]['reward'] = reward
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

    return PolicySolver
