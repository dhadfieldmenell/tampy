from sco.prob import Prob
from sco.variable import Variable
from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.solver import Solver
from openravepy import matrixFromAxisAngle
from core.util_classes import common_predicates
from core.util_classes import baxter_predicates
from core.util_classes.matrix import Vector
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from ll_solver import LLSolver, LLParam
import itertools, random
import gurobipy as grb
import numpy as np
GRB = grb.GRB
from IPython import embed as shell
from core.util_classes import sampling
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes import baxter_sampling


MAX_PRIORITY=5
BASE_MOVE_COEFF = 10
TRAJOPT_COEFF=1e3
SAMPLE_SIZE = 5
BASE_SAMPLE_SIZE = 5


attr_map = {'Robot': ['lArmPose', 'lGripper','rArmPose', 'rGripper', 'pose'],
            'RobotPose':['lArmPose', 'lGripper','rArmPose', 'rGripper', 'value'],
            'EEPose': ['value', 'rotation'],
            'Can': ['pose', 'rotation'],
            'Target': ['value', 'rotation'],
            'Obstacle': ['pose', 'rotation']}

class RobotLLSolver(LLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        self.transfer_coeff = 1e1
        self.rs_coeff = 1e10
        self.initial_trust_region_size = 1e-2
        self.init_penalty_coeff = 1e1
        # self.init_penalty_coeff = 1e5
        self.max_merit_coeff_increases = 5
        self._param_to_ll = {}
        self.early_converge=early_converge
        self.child_solver = None
        self.solve_priorities = [2]
        self.transfer_norm = transfer_norm


    def _solve_helper(self, plan, callback, active_ts, verbose):
        # certain constraints should be solved first
        success = False
        for priority in self.solve_priorities:
            success = self._solve_opt_prob(plan, priority=priority,
                            callback=callback, active_ts=active_ts,
                            verbose=verbose)
            # if not success:
            #     return success
        # return True
        return success

    def solve(self, plan, callback=None, n_resamples=20, active_ts=None,
              verbose=False, force_init=False):
        success = False
        if force_init or not plan.initialized:
             ## solve at priority -1 to get an initial value for the parameters
            self._solve_opt_prob(plan, priority=-2, callback=callback,
                active_ts=active_ts, verbose=verbose)
            self._solve_opt_prob(plan, priority=-1, callback=callback,
                active_ts=active_ts, verbose=verbose)
            plan.initialized=True

        success = self._solve_helper(plan, callback=callback,
            active_ts=active_ts, verbose=verbose)
        if success:
            return success

        for _ in range(20):
            ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints

            self._solve_opt_prob(plan, priority=0, callback=callback,
                                 active_ts=active_ts, verbose=verbose)
            import ipdb; ipdb.set_trace()
            success = self._solve_opt_prob(plan, priority=1,
                            callback=callback, active_ts=active_ts, verbose=verbose)
            if success:
                return success
        return success

    def _solve_opt_prob(self, plan, priority, callback=None, init=True,
                        active_ts=None, verbose=False):
        robot = plan.params['baxter']
        body = plan.env.GetRobot("baxter")
        viewer = callback()
        def draw(t):
            viewer.draw_plan_ts(plan, t)


        ## active_ts is the inclusive timesteps to include
        ## in the optimization
        if active_ts==None:
            active_ts = (0, plan.horizon-1)

        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # _free_attrs is paied attentioned in here
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()


        self._bexpr_to_pred = {}
        if priority == -2:
            """
            Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
            """
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan,
                priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose, add_nonlin=False)
            tol = 1e-1
        elif priority == -1:
            """
            Solve the optimization problem while enforcing every constraints.
            """
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan,
                priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose,
                add_nonlin=True)
            plan.save_free_attrs()
            tol = 1e-1
        elif priority == 0:
            """
            When Optimization fails, resample new values for certain timesteps
            of the trajectory and solver as initialization
            """
            ## this should only get called with a full plan for now
            # assert active_ts == (0, plan.horizon-1)

            plan.restore_free_attrs()
            plan.save_free_attrs()

            failed_preds = plan.get_failed_preds()
            if len(failed_preds) <= 0:
                return True
            import ipdb; ipdb.set_trace()

            print "{} predicates fails, resampling process begin...\n \
                   Checking {}".format(len(failed_preds), failed_preds[0])

            ## this is an objective that places
            ## a high value on matching the resampled values
            obj_bexprs = []
            rs_obj = self._resample(plan, failed_preds)
            obj_bexprs.extend(rs_obj)
            # _get_transfer_obj returns the expression saying the current trajectory should be close to it's previous trajectory.
            # obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))
            # obj_bexprs.extend(self._get_unfree_obj(plan, active_ts))

            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=1,
                add_nonlin=True, active_ts= active_ts, verbose=verbose)
            tol = 1e-3
        elif priority >= 1:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True,
                                               active_ts=active_ts, verbose=verbose)
            tol=1e-3

        solv = Solver()
        solv.initial_trust_region_size = self.initial_trust_region_size
        solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases
        success = solv.solve(self._prob, method='penalty_sqp', tol=tol, verbose=True)
        self._update_ll_params()
        print "priority: {}".format(priority)
        # if callback is not None: callback(True)
        # if priority >= 1:
            ##Restore free_attrs values
        return success


    def _get_transfer_obj(self, plan, norm):
        """
            This function returns the expression e(x) = P|x - cur|^2
            Which says the optimized trajectory should be close to the
            previous trajectory.
            Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """

        transfer_objs = []
        if norm == 'min-vel':
            for param in plan.params.values():
                # if param._type in ['Robot', 'Can', 'EEPose']:
                for attr_name in param.__dict__.iterkeys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        if param.is_symbol():
                            T = 1
                        else:
                            T = plan.horizon
                        K = attr_type.dim
                        attr_val = getattr(param, attr_name)

                        # pose = param.pose
                        assert (K, T) == attr_val.shape
                        KT = K*T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        # P = np.eye(KT)
                        Q = np.dot(np.transpose(P), P)
                        cur_val = attr_val.reshape((KT, 1), order='F')
                        A = -2*cur_val.T.dot(Q)
                        b = cur_val.T.dot(Q.dot(cur_val))
                        # QuadExpr is 0.5*x^Tx + Ax + b
                        quad_expr = QuadExpr(2*self.transfer_coeff*Q,
                                             self.transfer_coeff*A, self.transfer_coeff*b)
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
                        bexpr = BoundExpr(quad_expr, Variable(param_ll_grb_vars, cur_val))
                        transfer_objs.append(bexpr)
        else:
            raise NotImplemented
        return transfer_objs

    def _resample(self, plan, preds):
        """
            This function first calls fail predicate's resample function,
            then, uses the resampled value to create a square difference cost
            function e(x) = |x - rs_val|^2 that will be minimized later.
            rs_val is the resampled value
        """
        val, attr_inds = None, None
        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            val, attr_inds = pred.resample(negated, t, plan)
            ## if no resample defined for that pred, continue
            if val is not None: break
        if val is None:
            return []

        bexprs, i = [], 0
        for p in attr_inds:
            ## get the ll_param for p and gurobi variables
            ll_p = self._param_to_ll[p]
            n_vals = 0
            grb_vars = []
            for attr, ind_arr, t in attr_inds[p]:
                n_vals += len(ind_arr)
                grb_vars.extend(
                    list(getattr(ll_p, attr)[ind_arr, t].flatten()))

            for j, grb_var in enumerate(grb_vars):
                ## create an objective saying stay close to the resampled value
                ## e(x) = (x - val[i+j])**2
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

    def _get_unfree_obj(self, plan, active_ts):
        """
            This function returns the expression e(x) = |x_fix - cur_fix|^2
            where x_fix are fixed value specified by parameter's _free_attrs map
            This objective function will keep the fixed trajectory same as before.
        """

        transfer_objs = []
        for param in plan.params.values():
            if param.is_symbol():
                continue
            for attr_name in param.__dict__.iterkeys():
                attr_type = param.get_attr_type(attr_name)
                if issubclass(attr_type, Vector):
                    fixed_inds = list(set(np.where(param._free_attrs[attr_name] == 0)[1]))
                    T = len(fixed_inds)
                    if T == 1:
                        continue
                    K = attr_type.dim
                    attr_val = getattr(param, attr_name)[:, fixed_inds]
                    # pose = param.pose
                    assert (K, T) == attr_val.shape
                    KT = K*T
                    Q = np.eye(KT)*10
                    cur_val = attr_val.reshape((KT, 1), order='F')
                    A = -2*cur_val.T.dot(Q)
                    b = cur_val.T.dot(Q.dot(cur_val))
                    # QuadExpr is 0.5*x^Tx + Ax + b
                    quad_expr = QuadExpr(2**Q, A, b)
                    param_ll = self._param_to_ll[param]
                    ll_attr_val = getattr(param_ll, attr_name)
                    param_ll_grb_vars = ll_attr_val[:, fixed_inds].reshape((KT, 1), order='F')
                    bexpr = BoundExpr(quad_expr, Variable(param_ll_grb_vars, cur_val))
                    transfer_objs.append(bexpr)
        return transfer_objs



    def _add_pred_dict(self, pred_dict, effective_timesteps, add_nonlin=True,
                       priority=MAX_PRIORITY, verbose=False):
        """
            This function creates constraints for the predicate and added to
            Prob class in sco.
        """
        ## for debugging
        ignore_preds = []
        priority = np.maximum(priority, 0)
        if not pred_dict['hl_info'] == "hl_state":
            start, end = pred_dict['active_timesteps']
            active_range = range(start, end+1)
            if verbose:
                print "pred being added: ", pred_dict
                print active_range, effective_timesteps
            negated = pred_dict['negated']
            pred = pred_dict['pred']

            if pred.get_type() in ignore_preds:
                return

            if pred.priority > priority: return
            assert isinstance(pred, common_predicates.ExprPredicate)
            expr = pred.get_expr(negated)

            for t in effective_timesteps:
                if t in active_range:
                    if expr is not None:
                        if add_nonlin or isinstance(expr.expr, AffExpr):
                            if verbose:
                                print "expr being added at time ", t
                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)
                            # TODO: REMOVE line below, for tracing back predicate for debugging.
                            bexpr.pred = pred
                            self._bexpr_to_pred[bexpr] = (negated, pred, t)
                            groups = ['all']
                            if self.early_converge:
                                ## this will check for convergence per parameter
                                ## this is good if e.g., a single trajectory quickly
                                ## gets stuck
                                groups.extend([param.name for param in pred.params])
                            self._prob.add_cnt_expr(bexpr, groups)

    def _add_first_and_last_timesteps_of_actions(self, plan, priority = MAX_PRIORITY,
                                                 add_nonlin=False, active_ts=None, verbose=False):
        """
            This function adds all linear predicates and first and last timestep
            non-linear predicates from actions that are active within the range of active_ts.
        """
        if active_ts==None:
            active_ts = (0, plan.horizon-1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            ## only add an action
            if action_start >= active_ts[1] and action_start > active_ts[0]: continue
            if action_end < active_ts[0]: continue
            for pred_dict in action.preds:
                if action_start >= active_ts[0]:
                    self._add_pred_dict(pred_dict, [action_start],
                                        priority=priority, add_nonlin=add_nonlin, verbose=verbose)
                if action_end <= active_ts[1]:
                    self._add_pred_dict(pred_dict, [action_end],
                                        priority=priority, add_nonlin=add_nonlin, verbose=verbose)
            ## add all of the linear ineqs
            timesteps = range(max(action_start+1, active_ts[0]),
                              min(action_end, active_ts[1]))
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps, add_nonlin=False,
                                    priority=priority, verbose=verbose)

    def _add_all_timesteps_of_actions(self, plan, priority=MAX_PRIORITY,
                                      add_nonlin=True, active_ts=None, verbose=False):
        """
            This function adds both linear and non-linear predicates from
            actions that are active within the range of active_ts.
        """
        if active_ts==None:
            active_ts = (0, plan.horizon-1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            if action_start >= active_ts[1] and action_start > active_ts[0]: continue
            if action_end < active_ts[0]: continue

            timesteps = range(max(action_start, active_ts[0]),
                              min(action_end, active_ts[1])+1)
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps, priority=priority,
                                    add_nonlin=add_nonlin, verbose=verbose)

    def _update_ll_params(self):
        """
            update plan's parameters from low level grb_vars.
            expected to be called after each optimization.
        """
        for ll_param in self._param_to_ll.values():
            ll_param.update_param()
        if self.child_solver:
            self.child_solver._update_ll_params()

    def _spawn_parameter_to_ll_mapping(self, model, plan, active_ts=None):
        """
            This function creates low level parameters for each parameter in the plan,
            initialized he corresponding grb_vars for each attributes in each timestep,
            update the grb models
            adds in equality constraints,
            construct a dictionary as param-to-ll_param mapping.
        """
        if active_ts == None:
            active_ts=(0, plan.horizon-1)
        horizon = active_ts[1] - active_ts[0] + 1
        self._param_to_ll = {}
        self.ll_start = active_ts[0]
        for param in plan.params.values():
            ll_param = LLParam(model, param, horizon, active_ts)
            ll_param.create_grb_vars()
            self._param_to_ll[param] = ll_param
        model.update()
        for ll_param in self._param_to_ll.values():
            ll_param.batch_add_cnts()

    def _add_obj_bexprs(self, obj_bexprs):
        """
            This function adds objective bounded expressions to the Prob class
            in sco.
        """
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    def _get_trajopt_obj(self, plan, active_ts=None):
        """
            This function selects parameter of type Robot and Can and returns
            the expression e(x) = |Px|^2
            Which optimize trajectory so that robot and can's attributes in
            current timestep is close to that of next timestep.
            forming a straight line between each end points.

            Where P is the KT x KT matrix, where Px is the difference of
            value in current timestep compare to next timestep
        """
        if active_ts == None:
            active_ts = (0, plan.horizon-1)
        start, end = active_ts
        traj_objs = []
        for param in plan.params.values():
            if param not in self._param_to_ll:
                continue
            if param._type in ['Robot', 'Can']:
                for attr_name in param.__dict__.iterkeys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        T = end - start + 1
                        K = attr_type.dim
                        attr_val = getattr(param, attr_name)
                        KT = K*T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        Q *= TRAJOPT_COEFF

                        quad_expr = None
                        if attr_name == 'pose' and param._type == 'Robot':
                            quad_expr = QuadExpr(BASE_MOVE_COEFF*Q,
                                                 np.zeros((1,KT)),
                                                 np.zeros((1,1)))
                        else:
                            quad_expr = QuadExpr(Q, np.zeros((1,KT)), np.zeros((1,1)))
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start:end+1].reshape((KT, 1), order='F')
                        bexpr = BoundExpr(quad_expr, Variable(param_ll_grb_vars, init_val))
                        # bexpr = BoundExpr(quad_expr, Variable(param_ll_grb_vars))
                        traj_objs.append(bexpr)
        return traj_objs
