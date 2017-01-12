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
    def __init__(self, early_converge=False):
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

    def _solve_helper(self, plan, callback, active_ts, verbose):
        # certain constraints should be solved first
        success = False
        for priority in self.solve_priorities:
            success = self._solve_opt_prob(plan, priority=priority, callback=callback, active_ts=active_ts, verbose=verbose)
            # if not success:
            #     return success
        # return True
        return success

    def solve(self, plan, callback=None, n_resamples=20, active_ts=None, verbose=False, force_init=False):
        success = False
        if force_init or not plan.initialized:
             ## solve at priority -1 to get an initial value for the parameters
            self._solve_opt_prob(plan, priority=-2, callback=callback, active_ts=active_ts, verbose=verbose)
            self._solve_opt_prob(plan, priority=-1, callback=callback, active_ts=active_ts, verbose=verbose)
            plan.initialized=True

        success = self._solve_helper(plan, callback=callback, active_ts=active_ts, verbose=verbose)
        # fp = plan.get_failed_preds()
        # if len(fp) == 0:
        #     return True
        if success:
            return success

        for _ in range(n_resamples):
            ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints

            self._solve_opt_prob(plan, priority=0, callback=callback, active_ts=active_ts, verbose=verbose)
            success = self._solve_opt_prob(plan, priority=3, callback=callback, active_ts=active_ts, verbose=verbose)
            # self._solve_opt_prob(plan, priority=0, callback=callback, active_ts=active_ts, verbose=verbose)
            #
            # self._solve_opt_prob(plan, priority=1, callback=callback, active_ts=active_ts, verbose=verbose)
            # success = self._solve_opt_prob(plan, priority=2, callback=callback, active_ts=active_ts, verbose=verbose)
            # fp = plan.get_failed_preds()
            # if len(fp) == 0:
            #     return True
            if success:
                return success
        return success

    def _solve_opt_prob(self, plan, priority, callback=None, init=True, active_ts=None,
                        verbose=False):
        robot = plan.params['baxter']
        body = plan.env.GetRobot("baxter")
        viewer = callback()
        ## Backup the free_attrs value
        plan.save_free_attrs()

        ## active_ts is the inclusive timesteps to include
        ## in the optimization
        if active_ts==None:
            active_ts = (0, plan.horizon-1)

        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        self._bexpr_to_pred = {}
        if priority == -2:
            """
            Initialize the trajectory while enforce only the linear constraints
            """
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan, priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose, add_nonlin=False)
            tol = 1e-1
        elif priority == -1:
            """
            Solve the optimization problem while enforcing every constraints.
            """
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(plan, priority=MAX_PRIORITY, active_ts=active_ts, verbose=verbose, add_nonlin=True)
            tol = 1e-1
        elif priority == 0:
            """
            When Optimization fails, resample new values for certain timesteps of the trajectory and solver as initialization
            """
            ## this should only get called with a full plan for now
            # assert active_ts == (0, plan.horizon-1)
            failed_preds = plan.get_failed_preds()
            if len(failed_preds) <= 0:
                return True
            print "{} predicates fails, resampling process begin...\n Checking {}".format(len(failed_preds), failed_preds[0])
            # import ipdb; ipdb.set_trace()
            # random.shuffle(failed_preds)
            ## this is an objective that places
            ## a high value on matching the resampled values
            obj_bexprs = []
            obj_bexprs.extend(self._resample(plan, failed_preds))
            ## solve an optimization movement primitive to
            ## transfer current trajectories
            obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            # obj_bexprs.extend(self._get_transfer_obj(plan, 'min-vel'))

            # active_dof = body.GetManipulator("right_arm").GetArmIndices()
            # init_dof = plan.params['robot_init_pose'].rArmPose
            # raw_traj = baxter_sampling.get_rrt_traj(plan.env, body, active_dof, init_dof, baxter.rArmPose[:,17])
            # traj = baxter_sampling.process_traj(raw_traj, 17)
            # baxter.rArmPose[:, 0:17] = traj

            self._add_obj_bexprs(obj_bexprs)



            # self._update_ll_params()

            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False)

            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=-1, add_nonlin=True, verbose=verbose)
            # self._add_all_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False, verbose=verbose)

            self._add_all_timesteps_of_actions(
                plan, priority=1, add_nonlin=True, verbose=verbose)

            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=1, add_nonlin=True, verbose=verbose)
            # self._add_all_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False, verbose=verbose)
            tol = 1e-1
        elif priority == 3:
            """
                Optimization with priority 3 is called right after resample with priority 0
            """
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True, active_ts= None, verbose=verbose)
            tol=1e-3
        elif priority >= 1:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True, active_ts=active_ts, verbose=verbose)
            tol=1e-3

        solv = Solver()
        solv.initial_trust_region_size = self.initial_trust_region_size
        solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases
        success = solv.solve(self._prob, method='penalty_sqp', tol=tol, verbose=True)
        self._update_ll_params()
        print "priority: {}".format(priority)
        # if callback is not None: callback(True)

        if priority >= 1:
            ##Restore free_attrs values
            plan.restore_free_attrs()
        return success


    def _get_transfer_obj(self, plan, norm):
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
        val, attr_inds = None, None

        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            val, attr_inds = pred.resample(negated, t, plan)
            ## no resample defined for that pred
            if val is not None: break
        if val is None:
            return []

        bexprs = []
        i = 0
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

    def _add_pred_dict(self, pred_dict, effective_timesteps, add_nonlin=True, priority=MAX_PRIORITY, verbose=False):
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

    def _add_first_and_last_timesteps_of_actions(self, plan, priority = MAX_PRIORITY, add_nonlin=False, active_ts=None, verbose=False):
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
                self._add_pred_dict(pred_dict, timesteps, add_nonlin=False, priority=priority, verbose=verbose)

    def _add_all_timesteps_of_actions(self, plan, priority=MAX_PRIORITY, add_nonlin=True, active_ts=None, verbose=False):
        if active_ts==None:
            active_ts = (0, plan.horizon-1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            if action_start >= active_ts[1] and action_start > active_ts[0]: continue
            if action_end < active_ts[0]: continue

            timesteps = range(max(action_start, active_ts[0]),
                              min(action_end, active_ts[1])+1)
            for pred_dict in action.preds:
                self._add_pred_dict(pred_dict, timesteps, priority=priority, add_nonlin=add_nonlin, verbose=verbose)

    def _update_ll_params(self):
        for ll_param in self._param_to_ll.values():
            ll_param.update_param()
        if self.child_solver:
            self.child_solver._update_ll_params()

    def _spawn_parameter_to_ll_mapping(self, model, plan, active_ts=None):
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
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    def _get_trajopt_obj(self, plan, active_ts=None):
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
                            quad_expr = QuadExpr(BASE_MOVE_COEFF*Q, np.zeros((1,KT)), np.zeros((1,1)))
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
