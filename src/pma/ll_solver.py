from sco.prob import Prob
from sco.variable import Variable
from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.solver import Solver

from core.util_classes import common_predicates
from core.util_classes.matrix import Vector2d
from core.util_classes.namo_predicates import StationaryW

import gurobipy as grb
import numpy as np
GRB = grb.GRB
from IPython import embed as shell

MAX_PRIORITY=5

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
                name = "({}-{}-{})"

                for index in np.ndindex(shape):
                    # Note: it is easier for the sco code and update_param to
                    # handle things if everything is a Gurobi variable
                    x[index] = self._model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                  name=name.format(self._param.name, k, index))
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
                    # TODO: what's the purpose of _free_attrs (should they be the indices)?
                    if not self._param._free_attrs[attr][index]:
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

    def __init__(self):
        self.transfer_coeff = 1e1
        self.rs_coeff = 1e6
        self.init_penalty_coeff = 1e2

        self._bexpr_to_pred = {}
        self._ec_violated_pred = None

    def solve(self, plan, callback=None, n_resamples=5):
        success = False

        if not plan.initialized:
             ## solve at priority -1 to get an initial value for the parameters
            self._solve_opt_prob(plan, priority=-1, callback=callback)
            plan.initialized=True
        success = self._solve_opt_prob(plan, priority=1, callback=callback)
        if success:
            return success


        for _ in range(n_resamples):
        ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints
            self._solve_opt_prob(plan, priority=0, callback=callback)
            success = self._solve_opt_prob(plan, priority=1, callback=callback)
            if success:
                return success
        return success


    def _solve_opt_prob(self, plan, priority, callback=None):
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan)
        model.update()

        self._bexpr_to_pred = {}

        if priority == -1:
            obj_bexprs = self._get_trajopt_obj(plan)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan, priority=-1)
            tol = 1e-1
        elif priority == 0:
            # failed_preds = None
            failed_preds = plan.get_failed_preds()
            # if self._ec_violated_pred is None:
            #     failed_preds = plan.get_failed_preds()
            # else:
            #     failed_preds = [self._ec_violated_pred]
            ## this is an objective that places
            ## a high value on matching the resampled values
            obj_bexprs = self._resample(plan, failed_preds)
            ## solve an optimization movement primitive to
            ## transfer current trajectories
            obj_bexprs.extend(self._get_transfer_obj(plan, 'min-vel'))
            self._add_obj_bexprs(obj_bexprs)
            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False)
            self._add_first_and_last_timesteps_of_actions(
                plan, priority=-1, add_nonlin=True)
            self._add_all_timesteps_of_actions(
                plan, priority=0, add_nonlin=False)
            tol = 1e-1
        elif priority == 1:
            obj_bexprs = self._get_trajopt_obj(plan)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=1, add_nonlin=True)
            tol=1e-3

        solv = Solver()
        solv.initial_penalty_coeff = self.init_penalty_coeff
        # success, violated_bexpr = solv.solve(self._prob, method='penalty_sqp', tol=tol)
        success, violated_bexpr = solv.solve(self._prob, method='penalty_sqp_early_converge', tol=tol)
        if violated_bexpr is not None:
            self._ec_violated_pred = self._bexpr_to_pred[violated_bexpr]
        else:
            self._ec_violated_pred = None
        self._update_ll_params()
        return success

    def _get_transfer_obj(self, plan, norm):
        transfer_objs = []
        if norm == 'min-vel':
            for param in plan.params.values():
                if param._type in ['Robot', 'Can']:
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
                    cur_pose = pose.reshape((KT, 1), order='F')
                    A = -2*cur_pose.T.dot(Q)
                    b = cur_pose.T.dot(Q.dot(cur_pose))
                    # QuadExpr is 0.5*x^Tx + Ax + b
                    quad_expr = QuadExpr(2*self.transfer_coeff*Q,
                                         self.transfer_coeff*A, self.transfer_coeff*b)
                    ll_param = self._param_to_ll[param]
                    ll_grb_vars = ll_param.pose.reshape((KT, 1), order='F')
                    bexpr = BoundExpr(quad_expr, Variable(ll_grb_vars, cur_pose))
                    transfer_objs.append(bexpr)
        else:
            raise NotImplemented
        return transfer_objs

    def _resample(self, plan, preds):
        val, attr_inds = None, None
        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            val, attr_inds = pred.resample(negated, t, plan)
            ## no resample defined for that pred
            if val is not None: break
        if val is None:
            return None
        t_local = t
        bexprs = []
        i = 0
        for p in attr_inds:
                ## get the ll_param for p and gurobi variables
            ll_p = self._param_to_ll[p]
            if p.is_symbol(): t_local = 0
            n_vals = 0
            grb_vars = []
            for attr, ind_arr in attr_inds[p]:
                n_vals += len(ind_arr)
                grb_vars.extend(
                    list(getattr(ll_p, attr)[ind_arr, t_local]))

            for j, grb_var in enumerate(grb_vars):
                ## create an objective saying stay close to this value
                ## e(x) = x^2 - 2*val[i+j]*x + val[i+j]^2
                Q = np.eye(1)
                A = -2*val[i+j]*np.ones((1, 1))
                b = np.ones((1, 1))*np.power(val[i+j], 2)
                # QuadExpr is 0.5*x^Tx + Ax + b
                quad_expr = QuadExpr(2*Q*self.rs_coeff, A*self.rs_coeff, b*self.rs_coeff)
                v_arr = np.array([grb_var]).reshape((1, 1), order='F')
                init_val = np.ones((1, 1))*val[i+j]
                bexpr = BoundExpr(quad_expr,
                                  Variable(v_arr, val[i+j].reshape((1, 1))))

                bexprs.append(bexpr)

            i += n_vals
        return bexprs

    def _add_pred_dict(self, pred_dict, effective_timesteps, add_nonlin=True, priority=MAX_PRIORITY):
        ## for debugging
        ignore_preds = []
        priority = np.maximum(priority, 0)
        if not pred_dict['hl_info'] == "hl_state":
            # print "pred being added: ", pred_dict
            start, end = pred_dict['active_timesteps']
            active_range = range(start, end+1)
            # print active_range, effective_timesteps
            negated = pred_dict['negated']
            pred = pred_dict['pred']

            if pred.get_type() in ignore_preds:
                return

            # if isinstance(pred, StationaryW):
            #     print "stationary added!!!"
            #     import pdb; pdb.set_trace()



            if pred.priority > priority: return

            # if pred.get_type() == 'InContact':
            #     import pdb; pdb.set_trace()

            assert isinstance(pred, common_predicates.ExprPredicate)
            expr = pred.get_expr(negated)


            for t in effective_timesteps:
                if t in active_range:
                    if expr is not None:
                        if add_nonlin or isinstance(expr.expr, AffExpr):
                            # print "expr being added at time ", t
                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)
                            self._bexpr_to_pred[bexpr] = (negated, pred, t)
                            self._prob.add_cnt_expr(bexpr)

    def _add_first_and_last_timesteps_of_actions(self, plan, priority = MAX_PRIORITY, add_nonlin=False):
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, [action_start, action_end], priority=priority, add_nonlin=add_nonlin)
            ## add all of the linear ineqs
            timesteps = range(action_start+1, action_end)
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps, add_nonlin=False, priority=priority)


    def _add_all_timesteps_of_actions(self, plan, priority=MAX_PRIORITY, add_nonlin=True):
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            timesteps = range(action_start, action_end+1)
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps, priority=priority, add_nonlin=add_nonlin)

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


    def _get_trajopt_obj(self, plan):
        traj_objs = []
        for param in plan.params.values():
            if param._type in ['Robot', 'Can', 'Obstacle']:
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
                bexpr = BoundExpr(quad_expr, Variable(robot_ll_grb_vars, robot_ll._param.pose.reshape((KT, 1), order='F')))
                traj_objs.append(bexpr)
        return traj_objs

    def _spawn_sco_var_for_pred(self, pred, t):

        i = 0
        x = np.empty(pred.x_dim , dtype=object)
        v = np.empty(pred.x_dim)
        for p in pred.attr_inds:
            for attr, ind_arr in pred.attr_inds[p]:
                n_vals = len(ind_arr)
                ll_p = self._param_to_ll[p]
                if p.is_symbol():
                    x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, 0]
                    v[i:i+n_vals] = getattr(p, attr)[ind_arr, 0]
                else:
                    x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, t]
                    v[i:i+n_vals] = getattr(p, attr)[ind_arr, t]
                i += n_vals
        if pred.dynamic:
            ## include the variables from the next time step
            for p in pred.attr_inds:
                for attr, ind_arr in pred.attr_inds[p]:
                    n_vals = len(ind_arr)
                    ll_p = self._param_to_ll[p]
                    if p.is_symbol():
                        x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, 0]
                        v[i:i+n_vals] = getattr(p, attr)[ind_arr, 0]
                    else:
                        x[i:i+n_vals] = getattr(ll_p, attr)[ind_arr, t+1]
                        v[i:i+n_vals] = getattr(p, attr)[ind_arr, t]
                    i += n_vals
        assert i >= pred.x_dim
        x = x.reshape((pred.x_dim, 1))
        v = v.reshape((pred.x_dim, 1))
        return Variable(x, v)

class CanSolver(LLSolver):
    pass

class DummyLLSolver(LLSolver):
    def solve(self, plan):
        return "solve"
