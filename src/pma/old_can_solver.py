from sco.sco_gurobi.prob import Prob
from sco.sco_gurobi.variable import Variable
from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.sco_gurobi.solver import Solver
from openravepy import matrixFromAxisAngle
from core.util_classes import common_predicates
from core.util_classes.matrix import Vector
from core.util_classes.pr2 import PR2
from core.util_classes.namo_predicates import StationaryW, InContact
from core.util_classes.openrave_body import OpenRAVEBody
from .ll_solver import LLSolver, LLParam
import itertools, random
import gurobipy as grb
import numpy as np

GRB = grb.GRB
from IPython import embed as shell
from core.util_classes import pr2_sampling
from core.util_classes.viewer import OpenRAVEViewer

MAX_PRIORITY = 5
BASE_MOVE_COEFF = 10
TRAJOPT_COEFF = 1e3
SAMPLE_SIZE = 5
BASE_SAMPLE_SIZE = 5
DOWN_ROT = np.array([0, np.pi / 2, 0])
GRIPPER_OPEN_VAL = 0.02
GRIPPER_CLOSE_VAL = 0.015

attr_map = {
    "Robot": ["backHeight", "lArmPose", "lGripper", "rArmPose", "rGripper", "pose"],
    "RobotPose": [
        "backHeight",
        "lArmPose",
        "lGripper",
        "rArmPose",
        "rGripper",
        "value",
    ],
    "EEPose": ["value", "rotation"],
    "Can": ["pose", "rotation"],
    "Target": ["value", "rotation"],
    "Obstacle": ["pose", "rotation"],
}


class CanSolver(LLSolver):
    def __init__(self, early_converge=False):
        self.transfer_coeff = 1e1
        self.rs_coeff = 1e10
        self.initial_trust_region_size = 1e-2
        self.init_penalty_coeff = 1e1
        # self.init_penalty_coeff = 1e5
        self.max_merit_coeff_increases = 5
        self._param_to_ll = {}
        self.early_converge = early_converge
        self.child_solver = None
        self.solve_priorities = [2]

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
            # if not success:
            #     return success
        # return True
        return success

    def backtrack_solve(self, plan, callback=None, anum=0, verbose=False):
        if anum > len(plan.actions) - 1:
            return True
        a = plan.actions[anum]
        active_ts = a.active_timesteps
        inits = {}
        # Moveto -> Grasp -> Movetoholding -> Putdown
        robot = a.params[0]
        lookup_preds = "EEReachable"
        if a.name == "moveto":
            ## moveto: (?robot - Robot ?start - RobotPose ?end - RobotPose)
            ## sample ?end - RobotPose
            rs_param = a.params[2]
        elif a.name == "movetoholding":
            ## movetoholding: (?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can)
            ## sample ?end - RobotPose
            rs_param = a.params[2]
        elif a.name == "grasp":
            ## grasp: (?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)
            ## sample ?ee - EEPose
            rs_param = a.params[4]
            lookup_preds = "InContact"
        elif a.name == "putdown":
            ## putdown: (?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)
            ## sample ?ep - RobotPose
            rs_param = a.params[5]
        elif a.name == "move_both_holding_cloth":
            rs_param = a.params[4]
        else:
            raise NotImplemented

        def recursive_solve():
            ## don't optimize over any params that are already set
            old_params_free = {}
            for p in plan.params.values():
                old_param_map = {}
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    for attr in attr_map[getattr(p, "_type")]:
                        old_param_map[attr] = p._free_attrs[attr].copy()
                        p._free_attrs[attr][:] = 0
                else:
                    for attr in attr_map[getattr(p, "_type")]:
                        old_param_map[attr] = p._free_attrs[attr][
                            :, active_ts[1]
                        ].copy()
                        p._free_attrs[attr][:, active_ts[1]] = 0
                old_params_free[p] = old_param_map
            self.child_solver = CanSolver()
            if self.child_solver.backtrack_solve(
                plan, callback=callback, anum=anum + 1, verbose=verbose
            ):
                return True
            ## reset free_attrs
            for p in a.params:
                if p.is_symbol():
                    if p not in a.params:
                        continue
                    for attr in attr_map[getattr(p, "_type")]:
                        p._free_attrs[attr] = old_params_free[p][attr]
                else:
                    for attr in attr_map[getattr(p, "_type")]:
                        p._free_attrs[attr][:, active_ts[1]] = old_params_free[p][attr]
            return False

        if rs_param.is_fixed(attr_map[getattr(rs_param, "_type")]):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = CanSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=0,
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
        rs_free = {}

        for attr in attr_map[getattr(rs_param, "_type")]:
            rs_free[attr] = rs_param._free_attrs[attr].copy()
            rs_param._free_attrs[attr][:] = 0

        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        ######################################################################
        # Want robot_poses to be a list of dict, mapping from attr to value
        robot_poses = []

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum + 1]
        else:
            act, next_act = plan.actions[anum], None

        robot = plan.params["pr2"]
        robot_body = robot.openrave_body
        start_ts, end_ts = act.active_timesteps
        old_l_arm_pose = robot.lArmPose[:, start_ts].reshape((7, 1))
        old_r_arm_pose = robot.rArmPose[:, start_ts].reshape((7, 1))
        old_pose = robot.pose[:, start_ts].reshape((3, 1))
        old_backHeight = robot.backHeight[:, start_ts].reshape((1, 1))
        if next_act and (
            next_act.name == "right_grasp" or next_act.name == "right_putdown"
        ):
            target = next_act.params[2]
            target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
            next_act.params[1].openrave_body.set_pose(
                target.pose[:, 0], target.rotation[:, 0]
            )
            robot.openrave_body.set_dof({"lArmPose": old_l_arm_pose})
            rposes = robot.openrave_body.get_ik_from_pose(
                target_pos, DOWN_ROT, "rightarm_torso"
            )
            if not len(rposes):
                pass

            closest_ind = np.argmin(np.sum(np.abs(rposes - old_r_arm_pose), axis=1))
            rpose = rposes[closest_ind]

            robot_poses.append(
                {
                    "lArmPose": old_l_arm_pose,
                    "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "rArmPose": rpose,
                    "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "pose": old_pose,
                    "backHeight": old_backHeight,
                }
            )
        elif next_act and (
            next_act.name == "left_grasp" or next_act.name == "left_putdown"
        ):
            target = next_act.params[2]
            target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
            next_act.params[1].openrave_body.set_pose(
                target.pose[:, 0], target.rotation[:, 0]
            )
            robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
            lposes = robot.openrave_body.get_ik_from_pose(
                target_pos, DOWN_ROT, "leftarm_torso"
            )
            if not len(lposes):
                pass

            closest_ind = np.argmin(np.sum(np.abs(lposes - old_l_arm_pose), axis=1))
            lpose = lposes[closest_ind]

            robot_poses.append(
                {
                    "lArmPose": lpose,
                    "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "rArmPose": old_r_arm_pose,
                    "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "pose": old_pose,
                    "backHeight": old_backHeight,
                }
            )
        elif act.name == "right_grasp" or act.name == "right_putdown":
            target = act.params[2]
            target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
            act.params[1].openrave_body.set_pose(
                target.pose[:, 0], target.rotation[:, 0]
            )
            robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
            rposes = robot.openrave_body.get_ik_from_pose(
                target_pos, DOWN_ROT, "rightarm_torso"
            )
            if not len(rposes):
                pass

            closest_ind = np.argmin(np.sum(np.abs(rposes - old_r_arm_pose), axis=1))
            rpose = rposes[closest_ind]

            robot_poses.append(
                {
                    "lArmPose": old_l_arm_pose,
                    "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "rArmPose": rpose,
                    "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "pose": old_pose,
                    "backHeight": old_backHeight,
                }
            )
        elif act.name == "left_grasp" or act.name == "left_putdown":
            target = act.params[2]
            target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
            next_act.params[1].openrave_body.set_pose(
                target.pose[:, 0], target.rotation[:, 0]
            )
            robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
            lposes = robot.openrave_body.get_ik_from_pose(
                target_pos, DOWN_ROT, "leftarm_torso"
            )
            if not len(lposes):
                pass

            closest_ind = np.argmin(np.sum(np.abs(lposes - old_l_arm_pose), axis=1))
            lpose = lposes[closest_ind]

            robot_poses.append(
                {
                    "lArmPose": lpose,
                    "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "rArmPose": old_r_arm_pose,
                    "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                    "pose": old_pose,
                    "backHeight": old_backHeight,
                }
            )
        else:
            raise NotImplemented

        ##########################################################################

        for rp in robot_poses:
            for attr in attr_map[rs_param._type]:
                dim = len(rp[attr])
                setattr(rs_param, attr, rp[attr].reshape((dim, 1)))
            success = False
            self.child_solver = CanSolver()
            success = self.child_solver.solve(
                plan,
                callback=callback_a,
                n_resamples=0,
                active_ts=active_ts,
                verbose=verbose,
                force_init=True,
            )
            if success:
                if recursive_solve():
                    break
                else:
                    success = False

        for attr in attr_map[getattr(rs_param, "_type")]:
            rs_param._free_attrs[attr] = rs_free[attr]

        return success

    def sample_rp_from_ee(t, plan, ee_poses, robot):
        target_pose = target.value[:, 0]
        dummy_body = OpenRAVEBody(plan.env, robot.name + "_dummy", robot.geom)
        possible_rps = []
        for ee_pose in ee_poses:
            base_pose = pr2_sampling.get_col_free_base_pose_around_target(
                t, plan, ee_pose.value, robot
            ).reshape((1, 3))
            ik_arm_poses = dummy_body.ik_arm_pose(ee_pose[0], ee_pose[1])
            cur_arm_poses = np.c_[robot.backHeight, robot.rArmPose].reshape((8,))
            chosen_pose = pr2_sampling.closest_arm_pose(ik_arm_poses, cur_arm_poses)
            torso_pose, arm_pose = chosen_pose[0], chosen_pose[1:]
            possible_rp = np.hstack(
                (
                    base_pose,
                    torso_pose,
                    robot.lArmPose[:, t],
                    robot.lGripper[:, t],
                    rArmPose,
                    robot.rGripper[:, t],
                )
            )
            possible_rps.append(possible_rp)

        dummy_body.delete()
        if len(possible_rps) <= 0:
            print("something went wrong")
            # import ipdb; ipdb.set_trace()

        return possible_rps

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

        if force_init or not plan.initialized:
            ## solve at priority -1 to get an initial value for the parameters
            self._solve_opt_prob(
                plan,
                priority=-2,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            self._solve_opt_prob(
                plan,
                priority=-1,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            plan.initialized = True

        success = self._solve_helper(
            plan, callback=callback, active_ts=active_ts, verbose=verbose
        )
        # fp = plan.get_failed_preds()
        # if len(fp) == 0:
        #     return True
        if success:
            return success

        for _ in range(n_resamples):
            ## refinement loop
            ## priority 0 resamples the first failed predicate in the plan
            ## and then solves a transfer optimization that only includes linear constraints
            self._solve_opt_prob(
                plan,
                priority=0,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )

            # self._solve_opt_prob(plan, priority=1, callback=callback, active_ts=active_ts, verbose=verbose)
            success = self._solve_opt_prob(
                plan,
                priority=2,
                callback=callback,
                active_ts=active_ts,
                verbose=verbose,
            )
            # fp = plan.get_failed_preds()
            # if len(fp) == 0:
            #     return True
            if success:
                return success
        return success

    def _solve_opt_prob(
        self, plan, priority, callback=None, init=True, active_ts=None, verbose=False
    ):
        ## active_ts is the inclusive timesteps to include
        ## in the optimization
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        # param_to_ll_old = self._param_to_ll.copy()
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()

        self._bexpr_to_pred = {}

        if priority == -2:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(
                plan,
                priority=MAX_PRIORITY,
                active_ts=active_ts,
                verbose=verbose,
                add_nonlin=False,
            )
            # self._add_all_timesteps_of_actions(plan, priority=1, add_nonlin=False, verbose=verbose)
            tol = 1e-1
        elif priority == -1:
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_first_and_last_timesteps_of_actions(
                plan,
                priority=MAX_PRIORITY,
                active_ts=active_ts,
                verbose=verbose,
                add_nonlin=True,
            )
            tol = 1e-1
        elif priority == 0:
            ## this should only get called with a full plan for now
            assert active_ts == (0, plan.horizon - 1)

            failed_preds = plan.get_failed_preds()
            ## this is an objective that places
            ## a high value on matching the resampled values
            obj_bexprs = []
            obj_bexprs.extend(self._resample(plan, failed_preds))
            ## solve an optimization movement primitive to
            ## transfer current trajectories
            obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            # obj_bexprs.extend(self._get_transfer_obj(plan, 'min-vel'))
            self._add_obj_bexprs(obj_bexprs)
            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False)

            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=-1, add_nonlin=True, verbose=verbose)
            # self._add_all_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False, verbose=verbose)

            self._add_all_timesteps_of_actions(
                plan, priority=1, add_nonlin=True, verbose=verbose
            )

            # self._add_first_and_last_timesteps_of_actions(
            #     plan, priority=1, add_nonlin=True, verbose=verbose)
            # self._add_all_timesteps_of_actions(
            #     plan, priority=0, add_nonlin=False, verbose=verbose)
            tol = 1e-1
        elif priority >= 1:
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
        solv.initial_trust_region_size = self.initial_trust_region_size
        solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases
        success = solv.solve(self._prob, method="penalty_sqp", tol=tol, verbose=True)
        self._update_ll_params()
        print("priority: {}".format(priority))
        # if callback is not None: callback(True)
        return success

    def _get_transfer_obj(self, plan, norm):
        transfer_objs = []
        if norm == "min-vel":
            for param in list(plan.params.values()):
                # if param._type in ['Robot', 'Can', 'EEPose']:
                for attr_name in param.__dict__.keys():
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
                        KT = K * T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        cur_val = attr_val.reshape((KT, 1), order="F")
                        A = -2 * cur_val.T.dot(Q)
                        b = cur_val.T.dot(Q.dot(cur_val))
                        # QuadExpr is 0.5*x^Tx + Ax + b
                        quad_expr = QuadExpr(
                            2 * self.transfer_coeff * Q,
                            self.transfer_coeff * A,
                            self.transfer_coeff * b,
                        )
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
                        bexpr = BoundExpr(
                            quad_expr, Variable(param_ll_grb_vars, cur_val)
                        )
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
            if val is not None:
                break
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
                grb_vars.extend(list(getattr(ll_p, attr)[ind_arr, t].flatten()))

            for j, grb_var in enumerate(grb_vars):
                ## create an objective saying stay close to this value
                ## e(x) = x^2 - 2*val[i+j]*x + val[i+j]^2
                Q = np.eye(1)
                A = -2 * val[i + j] * np.ones((1, 1))
                b = np.ones((1, 1)) * np.power(val[i + j], 2)
                # QuadExpr is 0.5*x^Tx + Ax + b
                quad_expr = QuadExpr(
                    2 * Q * self.rs_coeff, A * self.rs_coeff, b * self.rs_coeff
                )
                v_arr = np.array([grb_var]).reshape((1, 1), order="F")
                init_val = np.ones((1, 1)) * val[i + j]
                bexpr = BoundExpr(
                    quad_expr, Variable(v_arr, val[i + j].reshape((1, 1)))
                )

                bexprs.append(bexpr)

            i += n_vals
        return bexprs

    def _add_pred_dict(
        self,
        pred_dict,
        effective_timesteps,
        add_nonlin=True,
        priority=MAX_PRIORITY,
        verbose=False,
    ):
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
            assert isinstance(pred, common_predicates.ExprPredicate)
            expr = pred.get_expr(negated)

            for t in effective_timesteps:
                if t in active_range:
                    if expr is not None:
                        if add_nonlin or isinstance(expr.expr, AffExpr):
                            if verbose:
                                print("expr being added at time ", t)
                            var = self._spawn_sco_var_for_pred(pred, t)
                            bexpr = BoundExpr(expr, var)
                            # TODO: REMOVE line below, for tracing back predicate for debugging.
                            bexpr.pred = pred
                            self._bexpr_to_pred[bexpr] = (negated, pred, t)
                            groups = ["all"]
                            if self.early_converge:
                                ## this will check for convergence per parameter
                                ## this is good if e.g., a single trajectory quickly
                                ## gets stuck
                                groups.extend([param.name for param in pred.params])
                            self._prob.add_cnt_expr(bexpr, groups)

    def _add_first_and_last_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=False,
        active_ts=None,
        verbose=False,
    ):
        if active_ts == None:
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

    def _add_all_timesteps_of_actions(
        self,
        plan,
        priority=MAX_PRIORITY,
        add_nonlin=True,
        active_ts=None,
        verbose=False,
    ):
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

    def _update_ll_params(self):
        for ll_param in list(self._param_to_ll.values()):
            ll_param.update_param()
        if self.child_solver:
            self.child_solver._update_ll_params()

    def _spawn_parameter_to_ll_mapping(self, model, plan, active_ts=None):
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

    def _add_obj_bexprs(self, obj_bexprs):
        for bexpr in obj_bexprs:
            self._prob.add_obj_expr(bexpr)

    def _get_trajopt_obj(self, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)
        start, end = active_ts
        traj_objs = []
        for param in list(plan.params.values()):
            if param not in self._param_to_ll:
                continue
            if param._type in ["Robot", "Can"]:
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
                        Q *= TRAJOPT_COEFF

                        quad_expr = None
                        if attr_name == "pose" and param._type == "Robot":
                            quad_expr = QuadExpr(
                                BASE_MOVE_COEFF * Q, np.zeros((1, KT)), np.zeros((1, 1))
                            )
                        else:
                            quad_expr = QuadExpr(Q, np.zeros((1, KT)), np.zeros((1, 1)))
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start : end + 1].reshape(
                            (KT, 1), order="F"
                        )
                        bexpr = BoundExpr(
                            quad_expr, Variable(param_ll_grb_vars, init_val)
                        )
                        # bexpr = BoundExpr(quad_expr, Variable(param_ll_grb_vars))
                        traj_objs.append(bexpr)
        return traj_objs
