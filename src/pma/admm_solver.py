from sco.prob import Prob
from sco.variable import Variable
from sco.expr import BoundExpr, QuadExpr
from sco.solver import Solver

from core.util_classes.matrix import Vector

from pma.ll_solver import NAMOSolver

import gurobipy as grb
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

TOL = 1e-3
RO = 300
EPS_PRIMAL = 1e-5
EPS_DUAL = 1e-5
MAX_ADMM_ITERS = 5

class ADMMHelper(object):
    def __init__(self, consensus_dict, nonconsensus_dict):
        recursive_default_dict = lambda: defaultdict(recursive_default_dict)
        # lp = local param or param copy, lts = local timestep
        self._lp_attr_lts_x_bar = recursive_default_dict()
        self._lp_attr_lts_y = recursive_default_dict()
        self._param_attr_ts_x_bar = recursive_default_dict()
        self._param_attr_ts_primal_res = recursive_default_dict()
        self._param_attr_ts_dual_res = recursive_default_dict()
        for param, ts_to_triples in consensus_dict.iteritems():
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    attr_val = getattr(param, attr_name)
                    dim = attr_type.dim
                    for ts, triples in ts_to_triples.iteritems():
                        val = attr_val[:, ts]
                        self._param_attr_ts_x_bar[param][attr_name][ts] = val
                        for plan, param_copy, local_ts in triples:
                            # adding entry in self._param_attr_ts_x_bar
                            self._lp_attr_lts_x_bar[param_copy][attr_name][local_ts] = val
                            self._lp_attr_lts_y[param_copy][attr_name][local_ts] = np.zeros(dim)
        """
        param_ts_(plan, param_copy, local_ts) is good for updates
        x_bar mapping param_copy, attr_name, local_ts
        y mapping param_copy, attr_name, local_ts
        """
        self._consensus_dict = consensus_dict
        self._nonconsensus_dict = nonconsensus_dict

    def admm_step(self, ro=RO, verbose=False):
        """
        Updates x_bar and y
        """
        converged = True
        for param, ts_to_triple in self._consensus_dict.iteritems():
            if verbose: print "param: {}".format(param.name)
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    attr_val = getattr(param, attr_name)
                    dim = attr_type.dim
                    for ts, triples in ts_to_triple.iteritems():
                        val = attr_val[:, ts]
                        # assert len(triples) >= 2

                        x_bar = np.zeros(dim)
                        x_bar_old = self._param_attr_ts_x_bar[param][attr_name][ts]
                        primal_res = 0.0
                        N = len(triples)
                        if N == 0:
                            continue
                        for plan, param_copy, local_ts in triples:
                            if verbose:
                                print getattr(param_copy, attr_name)[:, local_ts]
                            x = getattr(param_copy, attr_name)[:, local_ts]
                            x_bar += x
                            primal_res += norm(x-x_bar_old, 2)**2
                        x_bar = x_bar/N

                        dual_res = N*(ro**2)*(norm(x_bar-x_bar_old, 2)**2)
                        if primal_res > EPS_PRIMAL or dual_res > EPS_DUAL:
                            converged = False
                        self._param_attr_ts_primal_res[param][attr_name][ts] = primal_res
                        self._param_attr_ts_dual_res[param][attr_name][ts] = dual_res

                        self._param_attr_ts_x_bar[param][attr_name][ts] = x_bar

                        for plan, param_copy, local_ts in triples:
                            self._lp_attr_lts_x_bar[param_copy][attr_name][local_ts] = x_bar
                            y_old = self._lp_attr_lts_y[param_copy][attr_name][local_ts]
                            x = getattr(param_copy, attr_name)[:, local_ts]
                            y = y_old + ro*(x - x_bar)
                            self._lp_attr_lts_y[param_copy][attr_name][local_ts] = y
        return converged

    def update_params(self):
        self._update_consensus()
        self._update_nonconsensus()

    def _update_consensus(self):
        for param, attr_ts_x_bar in self._param_attr_ts_x_bar.iteritems():
            for attr_name, ts_x_bar in attr_ts_x_bar.iteritems():
                for ts, x_bar in ts_x_bar.iteritems():
                    p_attr = getattr(param, attr_name)
                    p_attr[:, ts] = x_bar

    def _update_nonconsensus(self):
        for param, ts_triple_dict in self._nonconsensus_dict.iteritems():
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    for ts, triple in ts_triple_dict.iteritems():
                        start, end = ts
                        if triple is None:
                            continue
                        plan, param_copy, (s, e) = triple
                        p_attr = getattr(param, attr_name)
                        p_attr[:, start:end+1] = getattr(param_copy, attr_name)[:, s:e+1]

    def get_admm_exprs(self, local_param, ll_param, ro, trust_size=None):
        admm_objs = []
        # admm_cnts = []
        attr_ts_x_bar_dict = self._lp_attr_lts_x_bar[local_param]
        for attr_name, ts_x_bar_dict in attr_ts_x_bar_dict.iteritems():
            for local_ts, x_bar in ts_x_bar_dict.iteritems():
                yi = self._lp_attr_lts_y[local_param][attr_name][local_ts]

                attr_val = getattr(local_param, attr_name)
                K, T = attr_val.shape

                # QuadExpr is 0.5*x^Tx + Ax + b
                Q = ro*np.eye(K)
                A = (yi - ro*x_bar).reshape((1,K))
                quad_expr = QuadExpr(Q, A, np.zeros((1,1)))

                ll_attr_val = getattr(ll_param, attr_name)[:, local_ts]

                param_ll_grb_vars = ll_attr_val.reshape((K, 1), order='F')
                sco_var = Variable(param_ll_grb_vars, x_bar.reshape((K,1)))
                bexpr = BoundExpr(quad_expr, sco_var)
                admm_objs.append(bexpr)

        return admm_objs

class NAMOADMMSolver(NAMOSolver):
    def _compute_shared_timesteps(self, plan):
        shared_timesteps = []
        unshared_ranges = []

        if len(plan.actions) == 1:
            unshared_ranges.append((0, plan.horizon-1))
        elif len(plan.actions) > 1:
            for i, action in enumerate(plan.actions):
                start, end = action.active_timesteps
                if i > 0:
                    shared_timesteps.append(start)

                if i == 0:
                    unshared_ranges.append((0, end-1))
                elif i < len(plan.actions) - 1:
                    unshared_ranges.append((start+1, end-1))
                elif i == len(plan.actions) - 1:
                    unshared_ranges.append((start+1, end))
        return shared_timesteps, unshared_ranges

    def _classify_variables(self, plan):
        """
        Returns two dictionaries, a consensus dictionary and a nonconsensus
        dictionary. The purpose is to track the mapping between the plan
        parameters and their local action counterparts.

        consensus_dict: {parameter: {timestep: []}}
        nonconsensus_dict: {parameter: {(start ts, end ts): None}}
        The consensus dictionary keys and values are respectively a parameter
        and a dictionary with the timestep as the key and a list as the value.
        The nonconsensus dictionary keys and values are respectively a parameter
        and a dictionary with a tuple indicating the time range as the key and
        a value of None.

        Note: symbols are assumed to be consensus variables.
        """
        consensus_dict = {}
        nonconsensus_dict = {}
        shared_timesteps, unshared_ranges = self._compute_shared_timesteps(plan)
        for param in plan.params.values():
            if param.is_symbol():
                # assumes symbols are shared between different actions
                consensus_dict[param] = {0: []}
            else:
                # object parameters with shared timesteps are consensus variables
                if len(shared_timesteps) > 0:
                    consensus_dict[param] = {t: [] for t in shared_timesteps}
                if len(unshared_ranges) > 0:
                    nonconsensus_dict[param] = {r: None for r in unshared_ranges}
        return consensus_dict, nonconsensus_dict

    def solve(self, plan, callback=None, n_resamples=5, active_ts=None, verbose=False, force_init=False):
        success = False

        if force_init or not plan.initialized:
             ## solve at priority -1 to get an initial value for the parameters
            success = self._solve_opt_prob(plan, priority=-1, callback=callback, active_ts=active_ts, verbose=verbose)
            if not success:
                return success
            plan.initialized=True
        success = self._solve_opt_prob(plan, priority=1, callback=callback, active_ts=active_ts, verbose=verbose)
        success = self._solve_opt_prob(plan, priority=2, callback=callback, active_ts=active_ts, verbose=verbose)
        success = plan.satisfied(active_ts)
        if success:
            return success


        for _ in range(n_resamples):
        ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints
            self._solve_opt_prob(plan, priority=0, callback=callback, active_ts=active_ts, verbose=verbose)
            success = self._solve_opt_prob(plan, priority=2, callback=callback, active_ts=active_ts, verbose=verbose)
            success = plan.satisfied(active_ts)
            if success:
                return success
        return success

    def _solve_opt_prob(self, plan, priority, ro=RO, callback=None, active_ts=None,
                        verbose=False, resample=True):
        """
        Solves optimization problem of a plan at a given priority.

        callback(solver, plan) is a function that takes in the solver and plan
        and plots the current solution.
        """
        # for priority -1 and 0, we use the default solve method (not ADMM)
        def callback_no_args():
            self._update_ll_params()
            return callback(self, plan)

        def callback_no_args_admm():
            # no update because LL params aren't updated during the admm solve.
            # The update will cause the plan to have the values from the
            # initialization.
            return callback(self, plan)

        if priority == -1:
            return super(NAMOADMMSolver, self)._solve_opt_prob(plan, priority=-1,
                callback=callback_no_args, active_ts=active_ts, verbose=verbose,
                resample=resample)
        elif priority == 0:
            return super(NAMOADMMSolver, self)._solve_opt_prob(plan, priority=0,
                callback=callback_no_args, active_ts=active_ts, verbose=verbose,
                resample=resample)
        elif priority == 1:

            ## active_ts is the inclusive timesteps to include
            ## in the optimization
            if active_ts==None:
                active_ts = (0, plan.horizon-1)

            model = grb.Model()
            model.params.OutputFlag = 0
            self._prob = Prob(model, callback=callback_no_args)
            # param_to_ll_old = self._param_to_ll.copy()
            self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
            model.update()

            self._bexpr_to_pred = {}

            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(
                plan, priority=1, add_nonlin=True, verbose=verbose)
            self._add_all_timesteps_of_actions(
                plan, priority=0, add_nonlin=False, verbose=verbose)

            solv = Solver()
            solv.initial_penalty_coeff = self.init_penalty_coeff
            success = solv.solve(self._prob, method='penalty_sqp', tol=1e-3, verbose=verbose)
            self._update_ll_params()
            self._failed_groups = self._prob.nonconverged_groups
            return success

        elif priority == 2:
            # http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf page 48
            # Global variable consensus optimization
            consensus_dict, nonconsensus_dict = self._classify_variables(plan)

            action_plans = plan.get_action_plans(consensus_dict, nonconsensus_dict)
            admm_help = ADMMHelper(consensus_dict, nonconsensus_dict)
            callback(self, plan, clear=True)
            for i in range(MAX_ADMM_ITERS):
                if verbose: print "ADMM iteration {}".format(i)
                for action_plan in action_plans:
                    self._solve_admm_subproblem(action_plan, admm_help, ro=ro,
                        verbose=verbose, callback=callback)

                # compute x_bar and y's
                converged = admm_help.admm_step(ro=ro)
                admm_help.update_params()
                # callback_no_args_admm()
                if converged:
                    break
            # update variable values in the consensus variable optimization

    def _solve_admm_subproblem(self, plan, admm_help, ro=RO, tol=TOL,
                               verbose=False, callback=None):
        def callback_no_args():
            namo_solver._update_ll_params()
            callback(namo_solver, plan)
        active_ts = (0, plan.horizon-1)
        model = grb.Model()
        model.params.OutputFlag = 0
        namo_solver = NAMOSolver()
        namo_solver._prob = Prob(model, callback=callback_no_args)
        namo_solver._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        namo_solver._bexpr_to_pred = {}

        obj_bexprs = namo_solver._get_trajopt_obj(plan, active_ts)
        for param in plan.params.values():
            ll_param = namo_solver._param_to_ll[param]
            admm_objs = admm_help.get_admm_exprs(param, ll_param, ro)
            obj_bexprs.extend(admm_objs)
        namo_solver._add_obj_bexprs(obj_bexprs)

        namo_solver._add_all_timesteps_of_actions(plan, priority=1,
            add_nonlin=True, active_ts=active_ts, verbose=verbose)

        solv = Solver()
        solv.initial_penalty_coeff = namo_solver.init_penalty_coeff
        success = solv.solve(namo_solver._prob, method='penalty_sqp', tol=tol, verbose=verbose)
        namo_solver._update_ll_params()
        namo_solver._failed_groups = namo_solver._prob.nonconverged_groups
        return success
