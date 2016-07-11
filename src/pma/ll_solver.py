from sco.prob import Prob
from sco.variable import Variable
from sco.expr import BoundExpr, QuadExpr
from sco.solver import Solver

from core.util_classes import common_predicates
from core.util_classes.matrix import Vector2d

import gurobipy as grb
import numpy as np
GRB = grb.GRB
from IPython import embed as shell

class LLSolver(object):
    """
    LLSolver solves the underlying optimization problem using motion planning.
    This is where different refinement strategies (e.g. backtracking,
    randomized), different motion planners, and different optimization
    strategies (global, sequential) are implemented.
    """
    def solve(self, plan):
        raise NotImplementedError("Override this.")

class LLParam(object):
    """
    LLParam creates the low-level representation of parameters (Numpy array of
    Gurobi variables) that is used in the optimization. For every numerical
    attribute in the original parameter, LLParam has the corresponding
    attribute where the value is a numpy array of Gurobi variables. LLParam
    updates its parameter based off of the value of the Gurobi variables.

    Create create_grb_vars and batch_add_cnts aren't included in the
    initialization because a Gurobi model update needs to be done before the
    batch_add_cnts call. Model updates can be very slow, so we want to create
    all the Gurobi variables for all the parameters before adding all the
    constraints.
    """
    def __init__(self, model, param, horizon):
        self._model = model
        self._param = param
        self._horizon = horizon
        self._num_attrs = []

    def create_grb_vars(self):
        """
        Creates Gurobi variables for attributes of certain types.
        """
        for k, _ in self._param.__dict__.items():
            rows = None
            if self._param.get_attr_type(k) == Vector2d:
                rows = 2

            if rows is not None:
                self._num_attrs.append(k)

                shape = None
                value = None
                if self._param.is_symbol():
                    shape = (2, 1)
                else:
                    shape = (2, self._horizon)

                x = np.empty(shape, dtype=object)

                for index in np.ndindex(shape):
                    # Note: it is easier for the sco code and update_param to
                    # handle things if everything is a Gurobi variable
                    x[index] = self._model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                setattr(self, k, x)

    def batch_add_cnts(self):
        """
        Adds all the equality constraints.
        """
        if self._param.is_defined():
            for attr in self._num_attrs:
                grb_vars = getattr(self, attr)
                value = getattr(self._param, attr)
                for index, value in np.ndenumerate(value):
                    if not np.isnan(value):
                        self._model.addConstr(grb_vars[index], GRB.EQUAL, value)

    def grb_val_dict(self):
        val_dict = {}
        for attr in self._num_attrs:
            val_dict[attr] = self._get_attr_val(attr)

    def _get_attr_val(self, attr):
        grb_vars = getattr(self, attr)

        value = np.zeros(grb_vars.shape)
        for index, var in np.ndenumerate(grb_vars):
            value[index] = var.X
        return value

    def update_param(self):
        """
        Updates the numerical attributes in the original parameter based off of
        the attribute's corresponding Gurobi variables.
        """
        for attr in self._num_attrs:
            value = self._get_attr_val(attr)
            setattr(self._param, attr, value)


class NAMOSolver(LLSolver):

    # def solve(self, plan):
    #
    #     model = grb.Model()
    #     # model.params.OutputFlag = 0
    #     self._prob = Prob(model)
    #
    #     self._spawn_parameter_to_ll_mapping(model, plan)
    #     model.update()
    #
    #     self._add_actions_to_sco_prob(plan)
    #     obj_bexprs = self._get_trajopt_obj(plan)
    #     self._add_obj_bexprs(obj_bexprs)
    #
    #     solv = Solver()
    #     solv.solve(self._prob, method='penalty_sqp')
    #     # self._update_ll_params()

    def solve(self, plan, callback=None):
        success = False
        # while not success:
            # self._solve_opt_prob(plan, priority=-1)
        success = self._solve_opt_prob(plan, priority=1, callback=None)

    def _solve_opt_prob(self, plan, priority, callback=None):
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        self._initialize_params(plan)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan)
        model.update()

        if priority == -1:
            obj_bexprs = self._get_trajopt_obj(plan) + self._get_init_obj(plan)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan)
        elif priority == 1:
            obj_bexprs = self._get_trajopt_obj(plan)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan)

        solv = Solver()
        success = solv.solve(self._prob, method='penalty_sqp')
        return success
        # self._update_ll_params()

    def _initialize_params(self, plan):
        self._init_values = {}
        for param in plan.params.values():
            self._resample(param)

    def _resample(self, param):
        if param.is_symbol():
            shape = param.value.shape
        else:
            shape = param.pose.shape
        self._init_values[param] = np.random.rand(*shape)

    def _add_pred_dict(self, pred_dict, effective_timesteps):
        if not pred_dict['hl_info'] == "hl_state":
            print "pred being added: ", pred_dict
            start, end = pred_dict['active_timesteps']
            active_range = range(start, end+1)
            for t in effective_timesteps:
                if t in active_range:
                    negated = pred_dict['negated']
                    pred = pred_dict['pred']
                    assert isinstance(pred, common_predicates.ExprPredicate)
                    expr = pred.get_expr(negated)
                    if expr is not None:
                        print "expr being added at time ", t
                        var = self._spawn_sco_var_for_pred(pred, t)
                        bexpr = BoundExpr(expr, var)
                        self._prob.add_cnt_expr(bexpr)

    def _add_first_and_last_timesteps_of_actions(self, plan):
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, [action_start, action_end])

    def _add_all_timesteps_of_actions(self, plan):
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            timesteps = range(action_start, action_end+1)
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps)

    # def _add_actions_to_sco_prob(self, plan):
    #     # needs to be modified to add only first and last time steps during initialization
    #     for action in plan.actions:
    #         for pred_dict in action.preds:
    #             if pred_dict['hl_info'] == "hl_state":
    #                 continue
    #             pred = pred_dict['pred']
    #             start, end = pred_dict['active_timesteps']
    #             for t in xrange(start, end+1):
    #                 assert isinstance(pred, common_predicates.ExprPredicate)
    #                 expr = pred.expr
    #                 if expr is not None:
    #                     var = self._spawn_sco_var_for_pred(pred, t)
    #                     bexpr = BoundExpr(expr, var)
    #                     self._prob.add_cnt_expr(bexpr)

    def _update_ll_params(self):
        for ll_param in self._param_to_ll.values():
            ll_param.update_param()

    def _spawn_parameter_to_ll_mapping(self, model, plan):
        horizon = plan.horizon
        self._param_to_ll = {}
        for param in plan.params.values():
            ll_param = LLParam(model, param, horizon)
            ll_param.create_grb_vars()
            self._param_to_ll[param] = ll_param
        model.update()
        for ll_param in self._param_to_ll.values():
            ll_param.batch_add_cnts()

    def _add_obj_bexprs(self, obj_bexprs):
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    # def _add_dynamic_cnts(self, plan):
    #     for param in plan.params.values():
    #         if param._type == 'Can':
    #             ll_param = self._param_to_ll[param]
    #             rows, cols = ll_param.pose.shape


    def _get_trajopt_obj(self, plan):
        traj_objs = []
        for param in plan.params.values():
            if param._type == 'Robot':
                T = plan.horizon
                K = 2
                pose = param.pose
                assert (K, T) == pose.shape
                KT = K*T
                v = -1 * np.ones((KT - K, 1))
                d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                # [:,0] allows numpy to see v and d as one-dimensional so
                # that numpy will create a diagonal matrix with v and d as a diagonal
                P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                Q = np.dot(np.transpose(P), P)

                quad_expr = QuadExpr(Q, np.zeros((1,KT)), np.zeros((1,1)))
                robot_ll = self._param_to_ll[param]
                robot_ll_grb_vars = robot_ll.pose.reshape((KT, 1), order='F')
                bexpr = BoundExpr(quad_expr, Variable(robot_ll_grb_vars))
                traj_objs.append(bexpr)
        return traj_objs

    def _get_init_obj(self, plan):
        init_objs = []
        for param in plan.params.values():
            if param._type == 'Grasp':
                value = self._init_values[param]
                ll_param = self._param_to_ll[param]
                g_var = ll_param.value
                assert g_var.shape == (2,1)
                assert value.shape == (2,1)
                Q = np.eye(2)
                A = -2*value.T
                b = np.zeros((1,1))
                quad_expr = QuadExpr(Q, A, b)
                bexpr = BoundExpr(quad_expr, Variable(g_var))
                init_objs.append(bexpr)
        return init_objs

    def _spawn_sco_var_for_pred(self, pred, t):
        i = 0
        x = np.empty(pred.x_dim , dtype=object)
        for p in pred.params:
            for attr, ind_arr in pred.attr_inds[p]:
                n_vals = len(ind_arr)
                ll_p = self._param_to_ll[p]
                if p.is_symbol():
                    x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, 0]
                else:
                    x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, t]
                i += n_vals
        x = x.reshape((pred.x_dim, 1))
        return Variable(x)

class CanSolver(LLSolver):
    pass

class DummyLLSolver(LLSolver):
    def solve(self, plan):
        return "solve"
