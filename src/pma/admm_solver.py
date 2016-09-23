from sco.prob import Prob
from sco.variable import Variable
from sco.expr import BoundExpr, QuadExpr
from sco.solver import Solver

from core.util_classes.matrix import Vector

from pma.ll_solver import NAMOSolver

from collections import defaultdict
from copy import deepcopy

class ADMMHelper(object):
    def __init__(self, consensus_dict, nonconsensus_dict):
        recursive_default_dict = lambda: defaultdict(recursive_default_dict)
        # lp = local param or param copy, lts = local timestep
        self._lp_attr_lts_x_bar = recursive_default_dict()
        self._lp_attr_lts_y = recursive_default_dict()
        self._param_attr_ts_x_bar = recursive_default_dict()
        for param, ts_to_triples in consensus_dict.iteritems():
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    attr_val = getattr(param, attr_name)
                    val = attr_val[:, ts]
                    dim = attr_type.dim
                    for ts, triples in ts_to_triple.iteritems():
                        self._param_attr_ts_x_bar[param][attr_name][ts] = val
                        for plan, param_copy, local_ts in triples:
                            # adding entry in self._param_attr_ts_x_bar
                            self._lp_attr_lts_x_bar[param_copy][attr_name][local_ts] = val
                            self._lp_attr_lts_y[param_copy][attr_name][local_ts] = np.zeros(3)
        """
        param_ts_(plan, param_copy, local_ts) is good for updates
        x_bar mapping param_copy, attr_name, local_ts
        y mapping param_copy, attr_name, local_ts
        """
        self._consensus_dict = copy.deepcopy(consensus_dict)
        self._nonconsensus_dict = copy.deepcopy(nonconsensus_dict)

    def admm_step(self):
        """
        Updates x_bar and y
        """
        for param, ts_to_triple in self._consensus_dict.iteritems():
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    attr_val = getattr(param, attr_name)
                    val = attr_val[:, ts]
                    dim = attr_type.dim
                    for ts, triples in ts_to_triple.iteritems():
                        assert len(triples) >= 2

                        x_bar = np.zeros(dim)
                        for plan, param_copy, local_ts in triples:
                            x_bar += getattr(param_copy, attr_name)[:, local_ts]
                        x_bar = x_bar/len(triples)
                        self._param_attr_ts_x_bar[param][attr_name][ts] = x_bar

                        for plan, param_copy, local_ts in triples:
                            self._lp_attr_lts_x_bar[param_copy][attr_name][local_ts] = x_bar
                            y_old = self._lp_attr_lts_y[param_copy][attr_name][local_ts]
                            x = getattr(param_copy, attr_name)[:, local_ts]
                            y = y_old + ro*(x_bar - x)
                            self._lp_attr_lts_y[param_copy][attr_name][local_ts] = y

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
        for param, ts_triple_dict in nonconsensus_dict.iteritems():
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    for ts, triple in ts_triple_dict.iteritems():
                        start, end = ts
                        triple = (plan, param_copy, (s, e))
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
                A = np.transpose(yi - 2*x_bar)
                quad_expr = QuadExpr(Q, A, np.zeros((1,1)))

                ll_attr_val = getattr(ll_param, attr_name)[:, local_ts]

                param_ll_grb_vars = ll_attr_val.reshape((K, 1), order='F')
                sco_var = Variable(param_ll_grb_vars, x_bar)
                bexpr = BoundExpr(quad_expr, sco_var)
                admm_objs.append(bexpr)

        return admm_objs

class NAMOADMMSolver(NAMOSolver):
    def _param_in_multiple_actions(self, plan, param):
        in_one_action = False
        for action in plan.actions:
            if param in action.params:
                if in_one_action:
                    return True
                else:
                    in_one_action = True
        return False

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
        Returns a dictionary where the keys are parameters which contain
        consensus variables and the values are the timesteps for the consensus
        variables.
        """
        consensus_dict = {}
        nonconsensus_dict = {}
        shared_timesteps, unshared_ranges = self._compute_shared_timesteps(plan)
        shared_ts_dict = {t: [] for t in shared_timesteps}
        unshared_ts_dict = {r: None for r in unshared_ranges}
        for param in plan.params.values():
            if param.is_symbol():
                # loop through each symbolic parameter and check if its in multiple actions
                if self._param_in_multiple_actions(plan, param):
                    consensus_dict[param] = {0: []}
                else:
                    assert param not in nonconsensus_dict
                    nonconsensus_dict[param] = {(0,0): None}
            else:
                # object parameters with shared timesteps are consensus variables
                if len(shared_ts_dict) > 0:
                    consensus_dict[param] = shared_ts_dict.copy()
                if len(unshared_ts_dict) > 0:
                    nonconsensus_dict[param] = unshared_ts_dict.copy()
        return consensus_dict, nonconsensus_dict

    # http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf page 48
    # Global variable consensus optimization
    def admm_solve(self, plan, callback, verbose=False):
        penalty_coeff = 1e2
        consensus_trust = 0.1
        eps_primal = 1e-3
        eps_dual = 1e-3
        ro = 300
        MAX_ITERS = 5
        consensus_dict, nonconsensus_dict = self._classify_variables(plan)

        solv = Solver()
        solv.initial_penalty_coeff = self.init_penalty_coeff
        action_solvers = self._setup_action_solvers(plan)
        action_plans = plan.get_action_plans(consensus_dict, nonconsensus_dict)
        admm_help = ADMMHelper(consensus_dict, nonconsensus_dict)
        for i in range(MAX_ITERS):
            for action_plan in action_plans:
                self._solve_admm_subproblem(action_plan, admm_help)

            # compute x_bar and y's
            admm_help.admm_step()
            # update variable values in the consensus variable optimization
            admm_help.update_params()

    def _solve_admm_subproblem(self, plan, admm_help, ro):
        active_ts = (0, plan.horizon-1)
        model = grb.Model()
        model.params.OutputFlag = 0
        namo_solver = NAMOSolver()
        namo_solver._prob = Prob(model, callback=callback)
        # param_to_ll_old = self._param_to_ll.copy()
        namo_solver._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()

        obj_bexprs = namo_solver._get_trajopt_obj(plan, active_ts)
        cnt_bexprs = []
        for param in plan.params:
            ll_param = namo_solver._param_to_ll[param]
            admm_objs = admm_help.get_admm_exprs(param, ll_param, ro)
            obj_bexprs.extend(admm_objs)
        namo_solver._add_obj_bexprs(obj_bexprs)

        solv = Solver()
        solv.initial_penalty_coeff = namo_solver.init_penalty_coeff
        success = solv.solve(namo_solver._prob, method='penalty_sqp', tol=tol, verbose=verbose)
        namo_solver._update_ll_params()
        namo_solver._failed_groups = namo_solver._prob.nonconverged_groups
        return success
