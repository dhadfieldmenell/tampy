import numpy as np
import pybullet as P

from sco.expr import BoundExpr, QuadExpr, AffExpr
from pma import bactrack_ll_solver_gurobi
from core.util_classes.namo_arm_predicates import (
    RETREAT_DIST,
    dsafe,
    opposite_angle,
    gripdist,
    ColObjPred,
    RETREAT_DIST,
)


class NAMOSolver(backtrack_ll_solver_gurobi.BacktrackLLSolver):
    def get_resample_param(self, a):
        return a.params[0]  # Experiment with avoiding robot pose symbols

        if a.name == "moveto":
            ## find possible values for the final pose
            rs_param = None  # a.params[2]
        elif a.name == "movetoholding":
            ## find possible values for the final pose
            rs_param = None  # a.params[2]
        elif a.name.find("grasp") >= 0:
            ## sample the grasp/grasp_pose
            rs_param = a.params[4]
        elif a.name == "putdown":
            ## sample the end pose
            rs_param = None  # a.params[4]
        elif a.name.find("place") >= 0:
            rs_param = a.params[2]
        elif a.name.find("moveto") >= 0:
            rs_param = a.params[4]
            # rs_param = None
        elif a.name.find("place_at") >= 0:
            # rs_param = None
            rs_param = a.params[2]
        elif a.name == "short_grasp":
            rs_param = a.params[4]
            # rs_param = None
        elif a.name == "short_movetograsp":
            rs_param = a.params[4]
            # rs_param = None
        elif a.name == "short_place_at":
            # rs_param = None
            rs_param = a.params[2]
        else:
            raise NotImplementedError

        return rs_param

    def freeze_rs_param(self, act):
        return True  # False

    def obj_pose_suggester(self, plan, anum, resample_size=1):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum + 1]
        else:
            act, next_act = plan.actions[anum], None

        robot = plan.params["pr2"]
        robot_body = robot.openrave_body
        ee_link = robot.geom.link_to_ind["ee"]
        start_ts, end_ts = act.active_timesteps
        robot_body.set_dof(
            {
                "joint1": robot.joint1[:, start_ts],
                "joint2": robot.joint2[:, start_ts],
                "wrist": robot.wrist[:, start_ts],
                "left_grip": robot.gripper[:, start_ts],
                "right_grip": robot.gripper[:, start_ts],
            }
        )
        ee_pos = P.getLinkState(robot_body.body_id, 5)[0][:2]
        ul = robot_body._geom.upper_bounds
        ll = robot_body._geom.lower_bounds
        jnt_rng = ul - ll
        ul = ul.tolist()
        ll = ll.tolist()
        jnt_rng = jnt_rng.tolist()
        rest_pos = (
            np.array(
                [
                    robot.joint1[:, start_ts],
                    robot.joint2[:, start_ts],
                    robot.wrist[:, start_ts],
                    robot.gripper[:, start_ts],
                    robot.gripper[:, start_ts],
                ]
            )
            .flatten()
            .tolist()
        )

        for i in range(resample_size):
            if next_act != None and (
                next_act.name == "grasp" or next_act.name == "putdown"
            ):
                target = next_act.params[2]
                target_pos = target.value - [[0], [0.0]]
                robot_pose.append(
                    {
                        "value": target_pos,
                        "gripper": np.array([[-1.0]])
                        if next_act.name == "putdown"
                        else np.array([[1.0]]),
                    }
                )
            elif (
                act.name == "moveto"
                or act.name == "new_quick_movetograsp"
                or act.name == "quick_moveto"
            ):
                target = act.params[2]
                target_rot = -np.arctan2(
                    target.value[0, 0] - ee_pos[0], target.value[1, 0] - ee_pos[1]
                )
                dist = gripdist
                target_pos = target.value - [
                    [-dist * np.sin(target_rot)],
                    [dist * np.cos(target_rot)],
                ]
                quat = [0, 0, np.sin(target_rot / 2.0), np.cos(target_rot / 2.0)]
                poses = [
                    np.random.normal([0, 0, 0, 0, 0], [3, 1, 1, 0.1, 0.1]).tolist()
                    for _ in range(3)
                ]
                for pos in [rest_pos, np.zeros_like(rest_pos).tolist()] + poses:
                    jnts = P.calculateInverseKinematics(
                        robot_body.body_id,
                        ee_link,
                        np.r_[target.value.flatten(), [0.5]],
                        lowerLimits=ll,
                        upperLimits=ul,
                        jointRanges=jnt_rng,
                        restPoses=pos,
                        maxNumIterations=500,
                    )
                    dof_map = {"joint1": jnts[0], "joint2": jnts[1], "wrist": jnts[2]}
                    robot_body.set_dof(dof_map)
                    new_pos = P.getLinkState(
                        robot_body.body_id, robot_body._geom.ee_link
                    )[0][:2]
                    if np.all(
                        np.abs(target.value.flatten() - np.array(new_pos)) < 0.001
                    ):
                        break
                    dof_map = {"joint1": pos[0], "joint2": pos[1], "wrist": pos[2]}
                    robot_body.set_dof(dof_map)

                if np.any(np.abs(target.value.flatten() - np.array(new_pos)) > 0.001):
                    print("WARNING! IK SOLVE FAILED", new_pos, target.value.flatten())

                robot_pose.append(
                    {
                        "joint1": np.array(jnts[0]).reshape((1, 1)),
                        "joint2": np.array(jnts[1]).reshape((1, 1)),
                        "wrist": np.array(jnts[2]).reshape((1, 1)),
                        "gripper": np.array(jnts[3]).reshape((1, 1)),
                    }
                )
                # robot_pose.append({'pose': target_pos + grasp.value, 'gripper': np.array([[-1.]])})
            elif act.name == "transfer" or act.name == "new_quick_place_at":
                target = act.params[2]
                target_rot = -np.arctan2(target.value[0, 0], target.value[1, 0])
                dist = RETREAT_DIST
                dist = gripdist
                target_pos = target.value - [
                    [-dist * np.sin(target_rot)],
                    [dist * np.cos(target_rot)],
                ]
                quat = [0, 0, np.sin(target_rot / 2.0), np.cos(target_rot / 2.0)]
                poses = [
                    np.random.normal([0, 0, 0, 0, 0], [3, 1, 1, 0.1, 0.1]).tolist()
                    for _ in range(3)
                ]
                for pos in [rest_pos, np.zeros_like(rest_pos).tolist()] + poses:
                    jnts = P.calculateInverseKinematics(
                        robot_body.body_id,
                        ee_link,
                        np.r_[target.value.flatten(), [0.5]],
                        lowerLimits=ll,
                        upperLimits=ul,
                        jointRanges=jnt_rng,
                        restPoses=pos,
                        maxNumIterations=500,
                    )
                    dof_map = {"joint1": jnts[0], "joint2": jnts[1], "wrist": jnts[2]}
                    robot_body.set_dof(dof_map)
                    new_pos = P.getLinkState(
                        robot_body.body_id, robot_body._geom.ee_link
                    )[0][:2]
                    if np.all(
                        np.abs(target.value.flatten() - np.array(new_pos)) < 0.001
                    ):
                        break
                    dof_map = {"joint1": pos[0], "joint2": pos[1], "wrist": pos[2]}
                    robot_body.set_dof(dof_map)

                if np.any(np.abs(target.value.flatten() - np.array(new_pos)) > 0.001):
                    print("WARNING! IK SOLVE FAILED", new_pos, target.value.flatten())

                robot_pose.append(
                    {
                        "joint1": np.array(jnts[0]).reshape((1, 1)),
                        "joint2": np.array(jnts[1]).reshape((1, 1)),
                        "wrist": np.array(jnts[2]).reshape((1, 1)),
                        "gripper": np.array(jnts[3]).reshape((1, 1)),
                    }
                )
            elif act.name == "place":
                target = act.params[4]
                grasp = act.params[5]
                target_rot = old_rot
                dist = -gripdist - dsafe - 1.4
                target_pos = target.value - [
                    [-dist * np.sin(target_rot)],
                    [dist * np.cos(target_rot)],
                ]
                quat = [0, 0, np.sin(target_rot / 2.0), np.cos(target_rot / 2.0)]
                jnts = P.calculateInverseKinematics(
                    robot_body.body_id, ee_link, target_pos, quat
                )
                jnts = P.calculateInverseKinematics(
                    robot_body.body_id, ee_link, np.r_[target.value.flatten(), [0.5]]
                )
                robot_pose.append(
                    {
                        "joint1": np.array(jnts[0]).reshape((1, 1)),
                        "joint2": np.array(jnts[1]).reshape((1, 1)),
                        "wrist": np.array(jnts[2]).reshape((1, 1)),
                        "gripper": np.array(jnts[3]).reshape((1, 1)),
                    }
                )
            else:
                raise NotImplementedError

        return robot_pose

    def _get_col_obj(self, plan, norm, mean, coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        if active_ts is None:
            active_ts = (0, plan.horizon - 1)

        start, end = active_ts
        if coeff is None:
            coeff = self.transfer_coeff

        objs = []
        robot = plan.params["pr2"]
        ll_robot = self._param_to_ll[robot]
        ll_robot_attr_val = getattr(ll_robot, "pose")
        robot_ll_grb_vars = ll_robot_attr_val.reshape((KT, 1), order="F")
        attr_robot_val = getattr(robot, "pose")
        init_robot_val = attr_val[:, start : end + 1].reshape((KT, 1), order="F")
        for robot in self._robot_to_ll:
            param_ll = self._param_to_ll[param]
            if param._type != "Can":
                continue
            attr_type = param.get_attr_type("pose")
            attr_val = getattr(param, "pose")
            init_val = attr_val[:, start : end + 1].reshape((KT, 1), order="F")
            K = attr_type.dim
            T = param_ll._horizon
            KT = K * T
            P = np.c_[np.eye(KT), -np.eye(KT)]
            Q = P.T.dot(P)
            quad_expr = QuadExpr(
                -2 * transfer_coeff * Q, np.zeros((KT)), np.zeros((1, 1))
            )
            ll_attr_val = getattr(param_ll, "pose")
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
            all_vars = np.r_[param_ll_grb_vars, robot_ll_grb_vars]
            sco_var = self.create_variable(all_vars, np.r_[init_val, init_robot_val])
            bexpr = BoundExpr(quad_expr, sco_var)

        for p_name, attr_name in self.state_inds:
            param = plan.params[p_name]
            if param.is_symbol():
                continue
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            attr_val = mean[param_ll.active_ts[0] : param_ll.active_ts[1] + 1][
                :, self.state_inds[p_name, attr_name]
            ]
            K = attr_type.dim
            T = param_ll._horizon

            if DEBUG:
                assert (K, T) == attr_val.shape
            KT = K * T
            v = -1 * np.ones((KT - K, 1))
            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
            # [:,0] allows numpy to see v and d as one-dimensional so
            # that numpy will create a diagonal matrix with v and d as a diagonal
            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
            # P = np.eye(KT)
            Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
            cur_val = attr_val.reshape((KT, 1), order="F")
            A = -2 * cur_val.T.dot(Q)
            b = cur_val.T.dot(Q.dot(cur_val))
            transfer_coeff = coeff / float(plan.horizon)

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(
                2 * transfer_coeff * Q, transfer_coeff * A, transfer_coeff * b
            )
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order="F")
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)

    def _add_col_obj(self, plan, norm="min-vel", coeff=None, active_ts=None):
        """
        This function returns the expression e(x) = P|x - cur|^2
        Which says the optimized trajectory should be close to the
        previous trajectory.
        Where P is the KT x KT matrix, where Px is the difference of parameter's attributes' current value and parameter's next timestep value
        """
        return None
        if active_ts is None:
            active_ts = (0, plan.horizon - 1)

        start, end = active_ts
        if coeff is None:
            coeff = self.col_coeff

        objs = []
        robot = plan.params["pr2"]
        for param in plan.params.values():
            if param._type != "Can":
                continue
            expected_param_types = ["Robot", "Can"]
            params = [robot, param]
            pred = ColObjPred(
                "obstr", params, expected_param_types, plan.env, coeff=coeff
            )
            pred.sess = plan.sess
            for t in range(active_ts[0], active_ts[1] - 1):
                var = self._spawn_sco_var_for_pred(pred, t)
                bexpr = BoundExpr(pred.neg_expr.expr, var)
                objs.append(bexpr)
                self._prob.add_obj_expr(bexpr)
        return objs
