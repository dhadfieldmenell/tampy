from sco.sco_gurobi.prob import Prob
from sco.sco_gurobi.variable import Variable
from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.sco_gurobi.solver import Solver
from openravepy import matrixFromAxisAngle
from core.internal_repr.parameter import Object
from core.util_classes import (
    baxter_predicates,
    common_predicates,
    robot_predicates,
    baxter_sampling,
    baxter_constants,
)
import core.util_classes.baxter_solve_enums as solve_enums
from core.util_classes.matrix import Vector
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.plan_hdf5_serialization import PlanSerializer
from .ll_solver import LLSolver, LLParam
import itertools, random
import gurobipy as grb
import numpy as np

GRB = grb.GRB
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes import baxter_sampling


MAX_PRIORITY = 3
BASE_MOVE_COEFF = 10
TRAJOPT_COEFF = 1e-1
SAMPLE_SIZE = 5
BASE_SAMPLE_SIZE = 5
DEBUG = True

# used for pose suggester
RESAMPLE_FACTOR = baxter_constants.RESAMPLE_FACTOR

attr_map = {}


class DrivingSolver(LLSolver):
    def __init__(self, early_converge=False, transfer_norm="min-vel"):
        # To avoid numerical difficulties during optimization, try keep
        # range of coefficeint within 1e9
        # (largest_coefficient/smallest_coefficient < 1e9)
        self.transfer_coeff = 1e0
        self.rs_coeff = 5e1
        self.trajopt_coeff = 1e0
        self.initial_trust_region_size = 1e2
        self.init_penalty_coeff = 4e3
        self.smooth_penalty_coeff = 7e4
        self.max_merit_coeff_increases = 5
        self._param_to_ll = {}
        self.early_converge = early_converge
        self.child_solver = None
        self.solve_priorities = [0, 1, 2, 3]
        self.transfer_norm = transfer_norm
        self.grb_init_mapping = {}
        self.var_list = []
        self._grb_to_var_ind = {}
        self.tol = 1e-3

    def _solve_helper(self, plan, callback, active_ts, verbose):
        # certain constraints should be solved first
        success = False
        for priority in self.solve_priorities:
            success = self._solve_opt_prob(
                plan,
                priority=priority,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )

        return success

    def backtrack_solve(self, plan, callback=None, verbose=False, n_resamples=5):
        plan.save_free_attrs()
        success = self._backtrack_solve(
            plan, callback, anum=0, verbose=verbose, n_resamples=n_resamples
        )
        # plan.restore_free_attrs()
        return success

    # @profile
    def _backtrack_solve(
        self, plan, callback=None, anum=0, verbose=False, amax=None, n_resamples=1
    ):
        # if anum == 2:
        #     import ipdb; ipdb.set_trace()
        if amax is None:
            amax = len(plan.actions) - 1

        if anum > amax:
            return True

        a = plan.actions[anum]
        print("backtracking Solve on {}".format(a.name))
        active_ts = a.active_timesteps
        inits = {}
        if a.name == "drive_down_road":
            rs_param = a.params[3]
        else:
            raise NotImplemented

        def recursive_solve():
            # import ipdb; ipdb.set_trace()
            ## don't optimize over any params that are already set
            old_params_free = {}
            for p in plan.params.values():
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in list(old_params_free[p].keys()):
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, active_ts[1]].copy()
                        p._free_attrs[attr][:, active_ts[1]] = 0
            self.child_solver = DrivingSolver()
            success = self.child_solver._backtrack_solve(
                plan, callback=callback, anum=anum + 1, verbose=verbose, amax=amax
            )

            # reset free_attrs
            for p in plan.params.values():
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, active_ts[1]] = old_params_free[p][attr]
            return success

        # if there is no parameter to resample or some part of rs_param is fixed, then go ahead optimize over this action
        if rs_param is None or sum(
            [
                not np.all(rs_param._free_attrs[attr])
                for attr in list(rs_param._free_attrs.keys())
            ]
        ):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = DrivingSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=n_resamples,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
            )

            if not success:
                ## if planning fails we're done
                return False
            ## no other options, so just return here
            return recursive_solve()

        ## so that this won't be optimized over
        rs_free = rs_param._free_attrs
        rs_param._free_attrs = {}
        for attr in list(rs_free.keys()):
            rs_param._free_attrs[attr] = np.zeros(rs_free[attr].shape)

        """
        sampler_begin
        """
        vehicle_poses = self.obj_pose_suggester(plan, anum, resample_size=3)
        if not vehicle_poses:
            success = False
            # print "Using Random Poses"
            # robot_poses = self.random_pose_suggester(plan, anum, resample_size = 5)

        """
        sampler end
        """

        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        for rp in vehicle_poses:
            for attr, val in list(rp.items()):
                setattr(rs_param, attr, val)

            success = False
            self.child_solver = DrivingSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=n_resamples,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
            )
            if success:
                if recursive_solve():
                    break
                else:
                    success = False

        rs_param._free_attrs = rs_free
        return success

    # @profile
    def random_pose_suggester(self, plan, anum, resample_size=5):
        pass

    # @profile
    def obj_pose_suggester(self, plan, anum, resample_size=20):
        vehicle_poses = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum + 1]
        else:
            act, next_act = plan.actions[anum], None

        vehicle = act.params[0]

        start_ts, end_ts = act.active_timesteps

        for i in range(resample_size):
            if act.name == "drive_down_road":
                road = act.params[1]
                init_pos = act.params[2]

                direction = road.geom.direction
                dist = road.geom.length / 2 - 1
                x = road.geom.x
                y = road.geom.y
                final_xy = np.array(
                    [[x + dist * np.cos(direction)], [y + dist * np.sin(direction)]]
                )

                vehicle_poses.append(
                    {
                        "xy": final_xy,
                        "theta": np.array([[direction]]),
                        "vel": np.zeros((1, 1)),
                        "phi": np.zeros((1, 1)),
                        "u1": np.zeros((1, 1)),
                        "u2": np.zeros((1, 1)),
                        "value": np.zeros((1, 1)),
                    }
                )
            else:
                raise NotImplementedError
        if not vehicle_poses:
            print("Unable to find IK")
            # import ipdb; ipdb.set_trace()

        return vehicle_poses

    # @profile
    def solve(
        self,
        plan,
        callback=None,
        n_resamples=5,
        active_ts=None,
        verbose=False,
        force_init=False,
    ):
        success = False
        if callback is not None:
            viewer = callback()
        if force_init or not plan.initialized:
            self._solve_opt_prob(
                plan,
                priority=-2,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            # self._solve_opt_prob(plan, priority=-1, callback=callback,
            #     active_ts=active_ts, verbose=verbose)
            plan.initialized = True

        if success or len(plan.get_failed_preds(active_ts=active_ts)) == 0:
            return True

        for priority in self.solve_priorities:
            for attempt in range(n_resamples):
                ## refinement loop
                success = self._solve_opt_prob(
                    plan,
                    priority=priority,
                    callback=callback,
                    active_ts=active_ts,
                    verbose=verbose,
                )

                try:
                    if DEBUG:
                        plan.check_cnt_violation(
                            active_ts=active_ts, priority=priority, tol=1e-3
                        )
                except:
                    print("error in predicate checking")
                if success:
                    break

                success = self._solve_opt_prob(
                    plan,
                    priority=priority,
                    callback=callback,
                    active_ts=active_ts,
                    verbose=verbose,
                    resample=True,
                )

                # if len(plan.get_failed_preds(active_ts=active_ts, tol=1e-3)) > 9:
                #     break

                print("resample attempt: {}".format(attempt))

                try:
                    if DEBUG:
                        plan.check_cnt_violation(
                            active_ts=active_ts, priority=priority, tol=1e-3
                        )
                except:
                    print("error in predicate checking")

                assert not (
                    success
                    and not len(
                        plan.get_failed_preds(
                            active_ts=active_ts, priority=priority, tol=1e-3
                        )
                    )
                    == 0
                )

                if success:
                    break

            if not success:
                return False

        return success

    # @profile
    def _solve_opt_prob(
        self,
        plan,
        priority,
        callback=None,
        init=True,
        active_ts=None,
        verbose=False,
        resample=False,
        smoothing=False,
    ):
        if callback is not None:
            viewer = callback()
        self.plan = plan
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        plan.save_free_attrs()
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        initial_trust_region_size = self.initial_trust_region_size
        if resample:
            tol = 1e-3

            def variable_helper():
                error_bin = []
                for sco_var in self._prob._vars:
                    for grb_var, val in zip(
                        sco_var.get_grb_vars(), sco_var.get_value()
                    ):
                        grb_name = grb_var[0].VarName
                        one, two = grb_name.find("-"), grb_name.find("-(")
                        param_name = grb_name[1:one]
                        attr = grb_name[one + 1 : two]
                        index = eval(grb_name[two + 1 : -1])
                        param = plan.params[param_name]
                        if not np.allclose(val, getattr(param, attr)[index]):
                            error_bin.append(
                                (grb_name, val, getattr(param, attr)[index])
                            )
                if len(error_bin) != 0:
                    print("something wrong")
                    if DEBUG:
                        import ipdb

                        ipdb.set_trace()

            """
            When Optimization fails, resample new values for certain timesteps
            of the trajectory and solver as initialization
            """
            obj_bexprs = []

            ## this is an objective that places
            ## a high value on matching the resampled values
            failed_preds = plan.get_failed_preds(
                active_ts=active_ts, priority=priority, tol=tol
            )
            rs_obj = self._resample(plan, failed_preds, sample_all=True)
            # import ipdb; ipdb.set_trace()
            # _get_transfer_obj returns the expression saying the current trajectory should be close to it's previous trajectory.
            # obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))

            self._add_all_timesteps_of_actions(
                plan,
                priority=priority,
                add_nonlin=False,
                active_ts=active_ts,
                verbose=verbose,
            )
            obj_bexprs.extend(rs_obj)
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e1
            # import ipdb; ipdb.set_trace()
        else:
            self._bexpr_to_pred = {}
            if priority == -2:
                """
                Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
                """
                obj_bexprs = self._get_trajopt_obj(plan, active_ts)
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=False,
                )
                tol = 1e-3
                initial_trust_region_size = 1e1
            elif priority == -1:
                """
                Solve the optimization problem while enforcing every constraints.
                """
                obj_bexprs = self._get_trajopt_obj(plan, active_ts)
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=True,
                )
                tol = 1e-3
            elif priority >= 0:
                obj_bexprs = self._get_trajopt_obj(plan, active_ts)
                self._add_obj_bexprs(obj_bexprs)
                self._add_all_timesteps_of_actions(
                    plan,
                    priority=priority,
                    add_nonlin=True,
                    active_ts=active_ts,
                    verbose=verbose,
                )
                tol = 1e-3

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size

        if smoothing:
            solv.initial_penalty_coeff = self.smooth_penalty_coeff
        else:
            solv.initial_penalty_coeff = self.init_penalty_coeff

        solv.max_merit_coeff_increases = self.max_merit_coeff_increases

        success = solv.solve(self._prob, method="penalty_sqp", tol=tol, verbose=verbose)
        success = (
            len(plan.get_failed_preds(tol=tol, active_ts=active_ts, priority=priority))
            == 0
        )
        self._update_ll_params()

        if DEBUG:
            assert not plan.has_nan(active_ts)

        # if resample:
        #     # During resampling phases, there must be changes added to sampling_trace
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
        print("priority: {}     success: {}\n".format(priority, success))

        return success

    # @profile
    def traj_smoother(self, plan, callback=None, n_resamples=5, verbose=False):
        # plan.save_free_attrs()
        a_num = 0
        success = True
        while a_num < len(plan.actions) - 1:
            act_1 = plan.actions[a_num]
            act_2 = plan.actions[a_num + 1]
            active_ts = (act_1.active_timesteps[0], act_2.active_timesteps[1])
            # print active_ts
            old_params_free = {}
            for p in plan.params.values():
                if p.is_symbol():
                    if p in act_1.params or p in act_2.params:
                        continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in list(old_params_free[p].keys()):
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = [
                            p._free_attrs[attr][:, : active_ts[0]].copy(),
                            p._free_attrs[attr][:, active_ts[1] :].copy(),
                        ]
                        p._free_attrs[attr][:, active_ts[1] :] = 0
                        p._free_attrs[attr][:, : active_ts[0]] = 0
            success = self._traj_smoother(
                plan, callback, n_resamples, active_ts, verbose
            )
            # reset free_attrs
            for p in plan.params.values():
                if p.is_symbol():
                    if p in act_1.params or p in act_2.params:
                        continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, : active_ts[0]] = old_params_free[p][
                            attr
                        ][0]
                        p._free_attrs[attr][:, active_ts[1] :] = old_params_free[p][
                            attr
                        ][1]

            if not success:
                return success
            print(
                "Actions: {} and {}".format(
                    plan.actions[a_num].name, plan.actions[a_num + 1].name
                )
            )
            a_num += 1
        # try:
        #     success = self._traj_smoother(plan, callback, n_resamples, active_ts, verbose)
        # except:
        #     print "Error occured during planning, but not catched"
        #     return False
        # plan.restore_free_attrs()
        return success

    # @profile
    def _traj_smoother(
        self, plan, callback=None, n_resamples=5, active_ts=None, verbose=False
    ):
        print("Smoothing Trajectory...")
        priority = MAX_PRIORITY
        for attempt in range(n_resamples):
            # refinement loop
            print("Smoother iteration #: {}\n".format(attempt))
            success = self._solve_opt_prob(
                plan,
                priority=priority,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
                resample=False,
                smoothing=True,
            )
            if success:
                break
            plan.check_cnt_violation(tol=1e-3)
            self._solve_opt_prob(
                plan,
                priority=priority,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
                resample=True,
                smoothing=True,
            )
        return success

    # @profile
    def _get_transfer_obj(self, plan, norm):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """

        transfer_objs = []
        if norm == "min-vel":
            for param in list(plan.params.values()):
                for attr_name in param.__dict__.keys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        param_ll = self._param_to_ll[param]
                        if param.is_symbol():
                            T = 1
                            attr_val = getattr(param, attr_name)
                        else:
                            T = param_ll._horizon
                            attr_val = getattr(param, attr_name)[
                                :, param_ll.active_ts[0] : param_ll.active_ts[1] + 1
                            ]
                        K = attr_type.dim

                        # pose = param.pose
                        if DEBUG:
                            assert (K, T) == attr_val.shape
                        KT = K * T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        # P = np.eye(KT)
                        Q = (
                            np.dot(np.transpose(P), P)
                            if not param.is_symbol()
                            else np.eye(KT)
                        )
                        cur_val = attr_val.reshape((KT, 1), order="F")
                        A = -2 * cur_val.T.dot(Q)
                        b = cur_val.T.dot(Q.dot(cur_val))
                        transfer_coeff = self.transfer_coeff / float(plan.horizon)

                        # QuadExpr is 0.5*x^Tx + Ax + b
                        quad_expr = QuadExpr(
                            2 * transfer_coeff * Q,
                            transfer_coeff * A,
                            transfer_coeff * b,
                        )
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
                        sco_var = self.create_variable(param_ll_grb_vars, cur_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        transfer_objs.append(bexpr)
        else:
            raise NotImplemented
        return transfer_objs

    # @profile
    def create_variable(self, grb_vars, init_vals, save=False):
        """
        if save is Ture
        Update the grb_init_mapping so that each grb_var is mapped to
        the right initial values.
        Then find the sco variables that includes the grb variables we are updating and change the corresponding initial values inside of it.
        if save is False
        Iterate the var_list and use the last initial value used for each gurobi, and construct the sco variables
        """
        sco_var, grb_val_map, ret_val = None, {}, []

        for grb, v in zip(grb_vars.flatten(), init_vals.flatten()):
            grb_name = grb.VarName
            if save:
                self.grb_init_mapping[grb_name] = v
            grb_val_map[grb_name] = self.grb_init_mapping.get(grb_name, v)
            ret_val.append(grb_val_map[grb_name])
            if grb_name in list(self._grb_to_var_ind.keys()):
                for var, i in self._grb_to_var_ind[grb_name]:
                    var._value[i] = grb_val_map[grb_name]
                    if np.all(var._grb_vars is grb_vars):
                        sco_var = var

        if sco_var is None:
            sco_var = Variable(grb_vars, np.array(ret_val).reshape((len(ret_val), 1)))
            self.var_list.append(sco_var)
            for i, grb in enumerate(grb_vars.flatten()):
                index_val_list = self._grb_to_var_ind.get(grb.VarName, [])
                index_val_list.append((sco_var, i))
                self._grb_to_var_ind[grb.VarName] = index_val_list

        if DEBUG:
            self.check_sync()
        return sco_var

    # @profile
    def check_grb_sync(self, grb_name):
        for var, i in self._grb_to_var_ind[grb_name]:
            print(var._grb_vars[i][0].VarName, var._value[i])

    # @profile
    def check_sync(self):
        """
        This function checks whether all sco variable are synchronized
        """
        grb_val_map = {}
        correctness = True
        for grb_name in list(self._grb_to_var_ind.keys()):
            for var, i in self._grb_to_var_ind[grb_name]:
                try:
                    correctness = np.allclose(
                        grb_val_map[grb_name], var._value[i], equal_nan=True
                    )
                except KeyError:
                    grb_val_map[grb_name] = var._value[i]
                except:
                    print("something went wrong")
                    import ipdb

                    ipdb.set_trace()
                if not correctness:
                    import ipdb

                    ipdb.set_trace()

    def reset_variable(self):
        self.grb_init_mapping = {}
        self.var_list = []
        self._grb_to_var_ind = {}

    def monitor_update(
        self,
        plan,
        update_values,
        callback=None,
        n_resamples=5,
        active_ts=None,
        verbose=False,
    ):
        print("Resolving after environment update...\n")
        if callback is not None:
            viewer = callback()
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        plan.save_free_attrs()
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # _free_attrs is paied attentioned in here
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        self._bexpr_to_pred = {}
        tol = 1e-3
        obj_bexprs = []
        rs_obj = self._update(plan, update_values)
        obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))

        self._add_all_timesteps_of_actions(
            plan,
            priority=MAX_PRIORITY,
            add_nonlin=True,
            active_ts=active_ts,
            verbose=verbose,
        )
        obj_bexprs.extend(rs_obj)
        self._add_obj_bexprs(obj_bexprs)
        initial_trust_region_size = 1e-2

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size
        solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases
        success = solv.solve(self._prob, method="penalty_sqp", tol=tol, verbose=verbose)
        self._update_ll_params()

        if DEBUG:
            assert not plan.has_nan(active_ts)

        plan.restore_free_attrs()

        self.reset_variable()
        print("monitor_update\n")
        return success

    def _update(self, plan, update_values):
        bexprs = []
        for val, attr_inds in update_values:
            if val is not None:
                for p in attr_inds:
                    ## get the ll_param for p and gurobi variables
                    ll_p = self._param_to_ll[p]
                    n_vals, i = 0, 0
                    grb_vars = []
                    for attr, ind_arr, t in attr_inds[p]:
                        for j, grb_var in enumerate(
                            getattr(ll_p, attr)[
                                ind_arr, t - ll_p.active_ts[0]
                            ].flatten()
                        ):
                            Q = np.eye(1)
                            A = -2 * val[p][i + j] * np.ones((1, 1))
                            b = np.ones((1, 1)) * np.power(val[p][i + j], 2)
                            resample_coeff = self.rs_coeff / float(plan.horizon)
                            # QuadExpr is 0.5*x^Tx + Ax + b
                            quad_expr = QuadExpr(
                                2 * Q * resample_coeff,
                                A * resample_coeff,
                                b * resample_coeff,
                            )
                            v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                            init_val = np.ones((1, 1)) * val[p][i + j]
                            sco_var = self.create_variable(
                                v_arr,
                                np.array([val[p][i + j]]).reshape((1, 1)),
                                save=True,
                            )
                            bexpr = BoundExpr(quad_expr, sco_var)
                            bexprs.append(bexpr)
                        i += len(ind_arr)
        return bexprs

    # @profile
    def _resample(self, plan, preds, sample_all=False):
        """
        This function first calls fail predicate's resample function,
        then, uses the resampled value to create a square difference cost
        function e(x) = |x - rs_val|^2 that will be minimized later.
        rs_val is the resampled value
        """
        bexprs = []
        val, attr_inds = None, None
        pred_type = {}
        for negated, pred, t in preds:
            ## returns a vector of new values and an
            ## attr_inds (OrderedDict) that gives the mapping
            ## to parameter attributes
            # if pred_type.get(pred.get_type, False):
            #     continue
            val, attr_inds = pred.resample(negated, t, plan)
            if val is not None:
                pred_type[pred.get_type] = True
            ## if no resample defined for that pred, continue
            if val is not None:
                for p in attr_inds:
                    ## get the ll_param for p and gurobi variables
                    ll_p = self._param_to_ll[p]
                    n_vals, i = 0, 0
                    grb_vars = []
                    for attr, ind_arr, t in attr_inds[p]:
                        for j, grb_var in enumerate(
                            getattr(ll_p, attr)[
                                ind_arr, t - ll_p.active_ts[0]
                            ].flatten()
                        ):
                            Q = np.eye(1)
                            A = -2 * val[p][i + j] * np.ones((1, 1))
                            b = np.ones((1, 1)) * np.power(val[p][i + j], 2)
                            resample_coeff = self.rs_coeff / float(plan.horizon)
                            # QuadExpr is 0.5*x^Tx + Ax + b
                            quad_expr = QuadExpr(
                                2 * Q * resample_coeff,
                                A * resample_coeff,
                                b * resample_coeff,
                            )
                            v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                            init_val = np.ones((1, 1)) * val[p][i + j]
                            sco_var = self.create_variable(
                                v_arr,
                                np.array([val[p][i + j]]).reshape((1, 1)),
                                save=True,
                            )
                            bexpr = BoundExpr(quad_expr, sco_var)
                            bexprs.append(bexpr)
                        i += len(ind_arr)
                if not sample_all:
                    break
        return bexprs

    # @profile
    def _add_pred_dict(
        self,
        pred_dict,
        effective_timesteps,
        add_nonlin=True,
        priority=MAX_PRIORITY,
        verbose=False,
    ):
        """
        This function creates constraints for the predicate and added to
        Prob class in sco.
        """
        ## for debugging
        ignore_preds = []
        priority = np.maximum(priority, 0)
        if not pred_dict["hl_info"] == "hl_state":
            start, end = pred_dict["active_timesteps"]
            active_range = list(range(start, end + 1))
            if verbose:
                print("pred being added: ", pred_dict)
                print(active_range, effective_timesteps)
            negated = pred_dict["negated"]
            pred = pred_dict["pred"]

            if pred.get_type() in ignore_preds:
                return

            if pred.priority > priority:
                return
            if DEBUG:
                assert isinstance(pred, common_predicates.ExprPredicate)
            expr = pred.get_expr(negated)

            if expr is not None:
                if add_nonlin or isinstance(expr.expr, AffExpr):
                    for t in effective_timesteps:
                        if t in active_range:
                            if verbose:
                                print("expr being added at time ", t)
                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)

                            # TODO: REMOVE line below, for tracing back predicate for debugging.
                            if DEBUG:
                                bexpr.source = (negated, pred, t)
                            self._bexpr_to_pred[bexpr] = (negated, pred, t)
                            groups = ["all"]
                            if self.early_converge:
                                ## this will check for convergence per parameter
                                ## this is good if e.g., a single trajectory quickly
                                ## gets stuck
                                groups.extend([param.name for param in pred.params])
                            self._prob.add_cnt_expr(bexpr, groups)

    # @profile
    def _add_first_and_last_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=False,
        active_ts=None,
        verbose=False,
    ):
        """
        Adding only non-linear constraints on the first and last timesteps of each action.
        """
        if active_ts is None:
            active_ts = (0, plan.horizon - 1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            ## only add an action
            if action_start >= active_ts[1] and action_start > active_ts[0]:
                continue
            if action_end < active_ts[0]:
                continue
            for pred_dict in action.preds:
                if action_start >= active_ts[0]:
                    self._add_pred_dict(
                        pred_dict,
                        [action_start],
                        priority=priority,
                        add_nonlin=add_nonlin,
                        verbose=verbose,
                    )
                if action_end <= active_ts[1]:
                    self._add_pred_dict(
                        pred_dict,
                        [action_end],
                        priority=priority,
                        add_nonlin=add_nonlin,
                        verbose=verbose,
                    )
            ## add all of the linear ineqs
            timesteps = list(
                range(
                    max(action_start + 1, active_ts[0]), min(action_end, active_ts[1])
                )
            )
            for pred_dict in action.preds:
                self._add_pred_dict(
                    pred_dict,
                    timesteps,
                    add_nonlin=False,
                    priority=priority,
                    verbose=verbose,
                )

    # @profile
    def _add_all_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=True,
        active_ts=None,
        verbose=False,
    ):
        """
        This function adds both linear and non-linear predicates from
        actions that are active within the range of active_ts.
        """
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        for action in plan.actions:
            action_start, action_end = action.active_timesteps
            if action_start >= active_ts[1] and action_start > active_ts[0]:
                continue
            if action_end < active_ts[0]:
                continue

            timesteps = list(
                range(
                    max(action_start, active_ts[0]), min(action_end, active_ts[1]) + 1
                )
            )
            for pred_dict in action.preds:
                self._add_pred_dict(
                    pred_dict,
                    timesteps,
                    priority=priority,
                    add_nonlin=add_nonlin,
                    verbose=verbose,
                )

    # @profile
    def _update_ll_params(self):
        """
        update plan's parameters from low level grb_vars.
        expected to be called after each optimization.
        """
        for ll_param in list(self._param_to_ll.values()):
            ll_param.update_param()
        if self.child_solver:
            self.child_solver._update_ll_params()

    # @profile
    def _spawn_parameter_to_ll_mapping(self, model, plan, active_ts=None):
        """
        This function creates low level parameters for each parameter in the plan,
        initialized he corresponding grb_vars for each attributes in each timestep,
        update the grb models
        adds in equality constraints,
        construct a dictionary as param-to-ll_param mapping.
        """
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        horizon = active_ts[1] - active_ts[0] + 1
        self._param_to_ll = {}
        self.ll_start = active_ts[0]
        for param in list(plan.params.values()):
            ll_param = LLParam(model, param, horizon, active_ts)
            ll_param.create_grb_vars()
            self._param_to_ll[param] = ll_param
        model.update()
        for ll_param in list(self._param_to_ll.values()):
            ll_param.batch_add_cnts()

    # @profile
    def _add_obj_bexprs(self, obj_bexprs):
        """
        This function adds objective bounded expressions to the Prob class
        in sco.
        """
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    # @profile
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
            active_ts = (0, plan.horizon - 1)
        start, end = active_ts
        traj_objs = []
        for param in list(plan.params.values()):
            if param not in self._param_to_ll:
                continue
            if isinstance(param, Object):
                for attr_name in param.__dict__.keys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        T = end - start + 1
                        K = attr_type.dim
                        attr_val = getattr(param, attr_name)
                        KT = K * T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        Q *= self.trajopt_coeff / float(plan.horizon)

                        quad_expr = None
                        quad_expr = QuadExpr(Q, np.zeros((1, KT)), np.zeros((1, 1)))
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start : end + 1].reshape(
                            (KT, 1), order="F"
                        )
                        sco_var = self.create_variable(param_ll_grb_vars, init_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        traj_objs.append(bexpr)
        return traj_objs
