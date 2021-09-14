from core.internal_repr.predicate import Predicate
from core.internal_repr.plan import Plan
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np

import pybullet as p

from collections import OrderedDict

from pma.ll_solver_gurobi import NAMOSolver


"""
This file implements the predicates for the 2D NAMO domain.
"""

dsafe = 1e-3  # 1e-1
# dmove = 1.1e0 # 5e-1
dmove = 1.5e0  # 5e-1
contact_dist = 5e-2  # dsafe

RS_SCALE = 0.5
N_DIGS = 5
GRIP_TOL = 5e-1
COL_TS = 4  # 3
NEAR_TOL = 0.4


ATTRMAP = {
    "Robot": (
        ("pose", np.array(list(range(2)), dtype=np.int)),
        ("gripper", np.array(list(range(1)), dtype=np.int)),
        ("vel", np.array(list(range(2)), dtype=np.int)),
        ("acc", np.array(list(range(2)), dtype=np.int)),
    ),
    "Can": (("pose", np.array(list(range(2)), dtype=np.int)),),
    "Target": (("value", np.array(list(range(2)), dtype=np.int)),),
    "RobotPose": (
        ("value", np.array(list(range(2)), dtype=np.int)),
        ("gripper", np.array(list(range(1)), dtype=np.int)),
    ),
    "Obstacle": (("pose", np.array(list(range(2)), dtype=np.int)),),
    "Grasp": (("value", np.array(list(range(2)), dtype=np.int)),),
}


def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]


def process_traj(raw_traj, timesteps):
    """
    Process raw_trajectory so that it's length is desired timesteps
    when len(raw_traj) > timesteps
        sample Trajectory by space to reduce trajectory size
    when len(raw_traj) < timesteps
        append last timestep pose util the size fits

    Note: result_traj includes init_dof and end_dof
    """
    result_traj = []
    if len(raw_traj) == timesteps:
        result_traj = raw_traj.copy()
    else:
        traj_arr = [0]
        result_traj.append(raw_traj[0])
        # calculate accumulative distance
        for i in range(len(raw_traj) - 1):
            traj_arr.append(
                traj_arr[-1] + np.linalg.norm(raw_traj[i + 1] - raw_traj[i])
            )
        step_dist = traj_arr[-1] / (timesteps - 1)
        process_dist, i = 0, 1
        while i < len(traj_arr) - 1:
            if traj_arr[i] == process_dist + step_dist:
                result_traj.append(raw_traj[i])
                process_dist += step_dist
            elif traj_arr[i] < process_dist + step_dist < traj_arr[i + 1]:
                dist = process_dist + step_dist - traj_arr[i]
                displacement = (
                    (raw_traj[i + 1] - raw_traj[i])
                    / (traj_arr[i + 1] - traj_arr[i])
                    * dist
                )
                result_traj.append(raw_traj[i] + displacement)
                process_dist += step_dist
            else:
                i += 1
    result_traj.append(raw_traj[-1])
    return np.array(result_traj).T


def get_rrt_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    active_dofs = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof)  # set goal to all ones
    # # forces parabolic planning with 40 iterations
    # import ipdb; ipdb.set_trace()
    params.SetExtraParameters(
        """<_postprocessing planner="parabolicsmoother">
        <_nmaxiterations>20</_nmaxiterations>
    </_postprocessing>"""
    )

    planner = RaveCreatePlanner(env, "birrt")
    planner.InitPlan(robot, params)

    traj = RaveCreateTrajectory(env, "")
    result = planner.PlanPath(traj)
    if result == False:
        robot.SetActiveDOFs(active_dofs)
        return None
    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data = traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(
            data, robot, robot.GetActiveDOFIndices()
        )
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(active_dofs)
    return np.array(traj_list)


def get_ompl_rrtconnect_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    dof_inds = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof)  # set goal to all ones
    # forces parabolic planning with 40 iterations
    planner = RaveCreatePlanner(env, "OMPL_RRTConnect")
    planner.InitPlan(robot, params)
    traj = RaveCreateTrajectory(env, "")
    planner.PlanPath(traj)

    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data = traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(
            data, robot, robot.GetActiveDOFIndices()
        )
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(dof_inds)
    return traj_list


def twostep_f(xs, dist, dim, pts=COL_TS, grad=False):
    if grad:
        res = []
        jac = np.zeros((0, 2 * dim))
        for t in range(pts):
            coeff = float(pts - t) / pts
            if len(xs) == 2:
                next_pos = coeff * xs[0] + (1 - coeff) * xs[1]
            else:
                next_pos = xs[0]
            res.append(dist(next_pos)[1])
            # jac = np.r_[jac, np.c_[coeff*res[t], (1-coeff)*res[t]]]
            jac = np.r_[jac, np.c_[res[t], res[t]]]
        return jac

    else:
        res = []
        for t in range(pts):
            coeff = float(pts - t) / pts
            if len(xs) == 2:
                next_pos = coeff * xs[0] + (1 - coeff) * xs[1]
            else:
                next_pos = xs[0]
            res.append(dist(next_pos)[0])
        return np.concatenate(res, axis=0)


class CollisionPredicate(ExprPredicate):
    def __init__(
        self,
        name,
        e,
        attr_inds,
        params,
        expected_param_types,
        dsafe=dsafe,
        debug=False,
        ind0=0,
        ind1=1,
        active_range=(0, 1),
        priority=3,
    ):
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)

        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1

        self._cache = {}
        self.n_cols = 1

        super(CollisionPredicate, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=active_range,
            priority=priority,
        )

    def test(self, time, negated=False, tol=1e-4):
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'." % self)
        try:
            result = self.neg_expr.eval(
                self.get_param_vector(time), tol=tol, negated=(not negated)
            )
            return result
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'." % self)

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.distance_from_obj(self.get_param_vector(t))
        self._debug = _debug

    # @profile
    def distance_from_obj(self, x, n_steps=0):
        flattened = tuple(x.round(N_DIGS).flatten())
        # if flattened in self._cache and self._debug is False:
        #     return self._cache[flattened]
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        # if hasattr(p0.geom, 'radius') and hasattr(p1.geom, 'radius'):
        #     disp = pose1 - pose0
        #     dist = np.linalg.norm(disp)
        #     vals = np.zeros((self.n_cols, 1))
        #     jacs = np.zeros((self.n_cols, 4))
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        pose0 = x[0:2]
        pose1 = x[2:4]
        b0.set_pose(pose0)
        b1.set_pose(pose1)

        collisions = p.getClosestPoints(b0.body_id, b1.body_id, contact_dist)

        # if p1.name == 'obs0':
        #     print b1.env_body.GetLinks()[0].GetCollisionData().vertices

        col_val, jac01 = self._calc_grad_and_val(
            p0.name, p1.name, pose0, pose1, collisions
        )
        # val = np.array([col_val])
        val = col_val
        jac = jac01
        # self._cache[flattened] = (val.copy(), jac.copy())

        return val, jac

    # @profile
    def _calc_grad_and_val(self, name0, name1, pose0, pose1, collisions):
        vals = np.zeros((self.n_cols, 1))
        jacs = np.zeros((self.n_cols, 4))

        val = -1 * float("inf")
        # jac0 = np.zeros(2)
        # jac1 = np.zeros(2)
        results = []
        n_cols = len(collisions)
        assert n_cols <= self.n_cols
        jac = np.zeros((1, 4))

        p0 = filter(lambda p: p.name == name0, list(self._param_to_body.keys()))[0]
        p1 = filter(lambda p: p.name == name1, list(self._param_to_body.keys()))[0]

        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        for i, c in enumerate(collisions):
            linkA, linkB = c[3], c[4]
            # linkA, linkB = c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c[1], c[2]
            # linkAParent, linkBParent = c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 0
            if linkAParent == b0.body_id and linkBParent == b1.body_id:
                # pt0, pt1 = c.positionOnA, c.positionOnB
                pt0, pt1 = c[5], c[6]
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == b0.body_id and linkAParent == b1.body_id:
                # pt0, pt1 = c.positionOnB, c.positionOnA
                pt1, pt0 = c[5], c[6]
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            distance = c[8]  # c.contactDistance
            normal = np.array(c[7])  # c.contactNormalOnB # Pointing towards A
            results.append((pt0, pt1, distance))
            # if distance < self.dsafe and 'obs0' in [name0, name1] and not np.any(np.isnan(pose0)) and not np.any(np.isnan(pose1)):
            #     print(name0, name1, distance, pose0, pose1)

            # plotting
            if self._debug:
                pt0[2] = 1.01
                pt1[2] = 1.01
                self._plot_collision(pt0, pt1, distance)
                print("pt0 = ", pt0)
                print("pt1 = ", pt1)
                print("distance = ", distance)
                print("normal = ", normal)

            vals[i, 0] = self.dsafe - distance
            jacs[i, :2] = -1 * normal[:2]
            jacs[i, 2:] = normal[:2]

        if self._debug:
            print("options: ", results)
            print("selected: ", chosen_pt0, chosen_pt1)
            print("selected distance: ", chosen_distance)
            self._plot_collision(chosen_pt0, chosen_pt1, chosen_distance)

        # if jac0 is None or jac1 is None or val is None:
        #     import ipdb; ipdb.set_trace()
        return np.array(vals).reshape((self.n_cols, 1)), np.array(jacs).reshape(
            (self.n_cols, 4)
        )

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(
                    self._env.drawarrow(p1=ptA, p2=ptB, linewidth=0.01, color=(1, 0, 0))
                )
            else:
                self.handles.append(
                    self._env.drawarrow(p1=ptA, p2=ptB, linewidth=0.01, color=(0, 0, 0))
                )


class HLPoseUsed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        ## At Can Target
        self.pose = params[0]
        if self.pose.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict([(self.pose, [(k, np.array([0, 1], dtype=np.int))])])

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLPoseUsed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.hl_info = True

    def test(self, time, negated=False, tol=1e-4):
        if negated:
            return True
        return super(HLPoseUsed, self).test(time, tol=tol)


class HLPoseAtGrasp(HLPoseUsed):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        k = "pose" if not self.r.is_symbol() else "value"
        attr_inds = OrderedDict(
            [
                (self.r, [(k, np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(HLPoseUsed, self).__init__(
            name, e, attr_inds, params, expected_param_types
        )
        self.hl_info = True


class HLAtGrasp(HLPoseUsed):
    pass


class HLPoseAtGrasp(HLPoseUsed):
    pass


class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        ## At Can Target
        self.can, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class AtNEq(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        ## At Can Target
        self.can, self.eq, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        if self.can is not self.eq:
            A = np.c_[np.eye(2), -np.eye(2)]
            b = np.zeros((2, 1))
            val = np.zeros((2, 1))
        else:
            A = np.zeros((2, 4))
            b = np.ones((2, 1))
            val = np.zeros((2, 1))

        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(AtNEq, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class RobotAt(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.rp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class BoxAt(At):
    pass


class Near(At):
    def __init__(self, name, params, expected_param_types, env=None):
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotNearTarget(At):
    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.t = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.t, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 0.25 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotNear(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 2 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class NotRobotNear(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = -2 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = 1.5e1 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotNearGrasp(At):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        b = np.zeros((4, 1))
        val = 1.5 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotAtGrasp(At):

    # RobotAt Robot Can Grasp

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c, self.g = params
        k = "pose" if not self.r.is_symbol() else "value"
        attr_inds = OrderedDict(
            [
                (self.r, [(k, np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[
            np.r_[np.eye(2), -np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
            np.r_[-np.eye(2), np.eye(2)],
        ]
        b = np.zeros((4, 1))
        val = NEAR_TOL * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class RobotPoseAtGrasp(At):
    pass


class RobotWithinReach(At):

    # RobotAt Robot Target

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.t = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.t, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)]]
        b = np.zeros((4, 1))
        val = 20 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class GripperClosed(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        (self.robot,) = params
        attr_inds = OrderedDict(
            [(self.robot, [("gripper", np.array([0], dtype=np.int))])]
        )
        A = np.ones((1, 1))
        b = np.zeros((1, 1))
        val = np.ones((1, 1))  # (GRIP_TOL + 1e-1) * -np.ones((1,1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        # e = LEqExpr(aff_e, val)

        neg_val = -np.ones((1, 1))  # (GRIP_TOL - 1e-1) * np.ones((1,1))
        neg_aff_e = AffExpr(A, b)
        self.neg_expr = EqExpr(neg_aff_e, neg_val)
        # self.neg_expr = LEqExpr(neg_aff_e, neg_val)
        super(GripperClosed, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return self.expr


class InContact(CollisionPredicate):

    # InContact, Robot, RobotPose, Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.rp, self.targ = params
        attr_inds = OrderedDict(
            [
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
                (self.targ, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.rp.geom),
            self.targ: self.lazy_spawn_or_body(
                self.targ, self.targ.name, self.targ.geom
            ),
        }

        INCONTACT_COEFF = 1e1
        f = lambda x: INCONTACT_COEFF * self.distance_from_obj(x)[0]
        grad = lambda x: INCONTACT_COEFF * self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.ones((1, 1)) * dsafe * INCONTACT_COEFF
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            debug=debug,
            ind0=1,
            ind1=2,
            active_range=(0, 0),
        )

    def test(self, time, negated=False, tol=1e-4):
        return super(CollisionPredicate, self).test(time, negated, tol)


class Collides(CollisionPredicate):

    # Collides Can Obstacle (wall)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict(
            [
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)

        def grad(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        N_COLS = 8

        col_expr = Expr(f, grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.n_cols = N_COLS
        # self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class TargetGraspCollides(Collides):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w, self.g = params
        if self.c.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict(
            [
                (self.c, [(k, np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        dist = 1.5

        def f(x):
            disp = x[:2] + dist * x[4:6]
            new_x = np.concatenate([disp, x[2:4]])
            return -self.distance_from_obj(new_x)[0]

        def grad(x):
            disp = x[:2] + dist * x[4:6]
            new_x = np.concatenate([disp, x[2:4]])
            jac = self.distance_from_obj(new_x)[1]
            return np.c_[np.zeros((8, 2)), jac]

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        N_COLS = 8

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
            priority=2,
        )
        self.n_cols = N_COLS
        # self.priority = 1


class RobotCanGraspCollides(Collides):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w, self.g = params
        if self.c.is_symbol():
            k = "value"
        else:
            k = "pose"
        attr_inds = OrderedDict(
            [
                (self.c, [(k, np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -self.distance_from_obj(x[:4])[0]

        def grad(x):
            jac = self.distance_from_obj(x[:4])[1]
            return np.c_[jac, np.zeros((8, 2))]

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        N_COLS = 8

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
            priority=2,
        )
        self.n_cols = N_COLS
        # self.priority = 1


class CanGraspCollides(TargetGraspCollides):
    pass


class TargetCanGraspCollides(TargetGraspCollides):
    pass


class TargetCollides(Collides):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict(
            [
                (self.c, [("value", np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        def f(x):
            return -self.distance_from_obj(x)[0]

        def grad(x):
            return self.distance_from_obj(x)[1]

        def f_neg(x):
            return -f(x)

        def grad_neg(x):
            return grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        # f_neg = lambda x: self.distance_from_obj(x)[0]
        # def grad_neg(x):
        #     # print self.distance_from_obj(x)
        #     return -self.distance_from_obj(x)[1]

        N_COLS = 8

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 0),
        )
        self.n_cols = N_COLS
        # self.priority = 1


class PoseCollides(TargetCollides):
    pass


class RCollides(CollisionPredicate):

    # RCollides Robot Obstacle (Wall)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.hl_ignore = True
        self.r, self.w = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.w, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = 1e4
        neg_grad_coeff = 1e-3

        """
        ## so we have an expr for the negated predicate
        def f_neg(x):
            d = neg_coeff * self.distance_from_obj(x)[0]
            # if np.any(d > 0):
            #     import ipdb; ipdb.set_trace()
            #     self.distance_from_obj(x)
            return d

        def grad_neg(x):
            # print self.distance_from_obj(x)
            return -neg_grad_coeff * self.distance_from_obj(x)[1]
        """

        def f(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)

        def grad(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return neg_grad_coeff * grad(x)

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        N_COLS = 8
        col_expr = Expr(f, grad)
        val = np.zeros((COL_TS * N_COLS, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(RCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1
        )
        self.n_cols = N_COLS

        # self.priority = 1

    def resample(self, negated, time, plan):
        assert negated
        res = OrderedDict()
        attr_inds = OrderedDict()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]

        if time == plan.actions[a].active_timesteps[1]:
            x = self.get_param_vector(time)
            val, jac = self.distance_from_obj(x, 0)
            jac = jac[0, :2]
        else:
            x1 = self.get_param_vector(time)
            x2 = self.get_param_vector(time + 1)
            jac = -twostep_f([x1, x2], self.distance_from_obj, 4, grad=True)
            jac = np.mean(jac[:, :2], axis=0)

        if np.all(jac == 0):
            return None, None

        jac = jac / (np.linalg.norm(jac) + 1e-3)

        new_robot_pose = self.r.pose[:, time] + np.random.uniform(0.1, 0.5) * jac
        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + (
                    (3.0 - dist) / 3.0
                ) * new_robot_pose

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, RobotPose, Can;

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.hl_ignore = True
        self.r, self.startp, self.endp, self.c = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )
        self._param_to_body = {
            self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
            self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
        }

        self.rs_scale = RS_SCALE

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = 1e2
        neg_grad_coeff = 1e-3
        """
        ## so we have an expr for the negated predicate
        f_neg = lambda x: neg_coeff*self.distance_from_obj(x)[0]
        def grad_neg(x):
            # print self.distance_from_obj(x)
            return neg_grad_coeff*self.distance_from_obj(x)[1]
        """

        def f(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)

        def grad(x):
            return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return -neg_grad_coeff * grad(x)

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Obstructs, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3
        )
        # self.priority=1

    def resample(self, negated, time, plan):
        assert negated
        res = OrderedDict()
        attr_inds = OrderedDict()
        # for param in [self.startp, self.endp]:
        #     val, inds = sample_pose(plan, param, self.r, self.rs_scale)
        #     if val is None:
        #         continue
        #     res.extend(val[inds].flatten().tolist())
        #     # inds[0] returns the x values of the indices which is what we care
        #     # about, because the y values correspond to time.
        #     attr_inds[param] = [('value', inds[0])]
        #     import ipdb; ipdb.set_trace()
        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None

        act = plan.actions[a]

        disp = self.c.pose[:, time] - self.r.pose[:, time]
        use_t = time
        new_disp = disp
        if (
            time < plan.actions[a].active_timesteps[1]
            and np.linalg.norm(disp)
            > self.r.geom.radius + self.c.geom.radius + self.dsafe
        ):
            new_disp = self.c.pose[:, time + 1] - self.r.pose[:, time + 1]

        if (
            np.linalg.norm(new_disp)
            < self.r.geom.radius + self.c.geom.radius + self.dsafe
        ):
            disp = new_disp
        else:
            disp = (disp + new_disp) / 2.0

        if disp[0] == 0:
            orth = np.array([1.0, 0.0])
        elif disp[1] == 0:
            orth = np.array([0.0, 1.0])
        else:
            orth = np.array([1.0 / disp[0], -1.0 / disp[1]])
        disp += 1e-5

        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        long_disp = self.r.pose[:, et] - self.r.pose[:, st]
        long_disp /= np.linalg.norm(long_disp)
        d1, d2 = long_disp.dot(orth), long_disp.dot(-orth)

        if d1 > d2:
            w1, w2 = 0.1, 0.9
        else:
            w1, w2 = 0.9, 0.1
        orth *= np.random.choice([-1.0, 1.0], p=[w1, w2])
        orth = orth / np.linalg.norm(orth)

        rdisp = (
            -(self.c.geom.radius + self.r.geom.radius + self.dsafe + 1e-1)
            * disp
            / np.linalg.norm(disp)
        )
        orth = rdisp  # + np.random.uniform(0.5, 2.) * orth
        # orth *= np.random.uniform(0.7, 1.5) * (self.c.geom.radius + self.r.geom.radius + self.dsafe)
        # orth += np.random.uniform([-0.15, 0.15], [-0.15, 0.15])

        # new_robot_pose = self.r.pose[:, time] + orth
        disp = orth
        st = max(max(time - 3, 1), act.active_timesteps[0] + 1)
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        ref_st = max(max(time - 3, 0), act.active_timesteps[0])
        ref_et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, st] + disp
                )
                inter_rp = (dist / 3.0) * self.r.pose[:, ref_st] + (
                    (3.0 - dist) / 3.0
                ) * (self.r.pose[:, time] + disp)
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, et] + disp
                )
                inter_rp = (dist / 3.0) * self.r.pose[:, ref_et] + (
                    (3.0 - dist) / 3.0
                ) * (self.r.pose[:, time] + disp)

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class WideObstructs(Obstructs):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(WideObstructs, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.2


def sample_pose(plan, pose, robot, rs_scale):
    targets = plan.get_param("InContact", 2, {0: robot, 1: pose})
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    inds = np.where(pose._free_attrs["value"])
    if np.sum(inds) == 0:
        return None, None  ## no resampling for this one
    if len(targets) == 1:
        # print "one target", pose
        random_dir = np.random.rand(2, 1) - 0.5
        random_dir = random_dir / np.linalg.norm(random_dir)
        # assumes targets are symbols
        val = targets[0].value + random_dir * 3 * robot.geom.radius
    elif len(targets) == 0:
        ## old generator -- just add a random perturbation
        # print "no targets", pose
        val = np.random.normal(pose.value[:, 0], scale=rs_scale)[:, None]
    else:
        raise NotImplementedError
    # print pose, val
    pose.value = val

    ## make the pose collision free
    _, collision_preds = plan.get_param("RCollides", 1, negated=True, return_preds=True)
    _, at_preds = plan.get_param(
        "RobotAt", 1, {0: robot, 1: pose}, negated=False, return_preds=True
    )
    preds = [(collision_preds[0], True), (at_preds[0], False)]
    old_pose = robot.pose.copy()
    old_free = robot._free_attrs["pose"].copy()
    robot.pose = pose.value.copy()
    robot._free_attrs["pose"][:] = 1

    wall = collision_preds[0].params[1]
    old_w_pose = wall.pose.copy()
    wall.pose = wall.pose[:, 0][:, None]

    old_priority = [p.priority for p, n in preds]
    for p, n in preds:
        p.priority = -1
    p = Plan.create_plan_for_preds(preds, collision_preds[0]._env)
    s = NAMOSolver(transfer_norm="l2")
    success = s._solve_opt_prob(p, 0, resample=False, verbose=False)

    # print success

    ## debugging
    # import viewer
    # v = viewer.OpenRAVEViewer.create_viewer()
    # v.draw_plan_ts(p, 0)
    # print pose.value, val

    ## restore the old values
    robot.pose = old_pose
    robot._free_attrs["pose"] = old_free
    for i, (p, n) in enumerate(preds):
        p.priority = old_priority[i]

    wall.pose = old_w_pose

    return pose.value, inds


class ObstructsHolding(CollisionPredicate):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can;
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        r, startp, endp, obstr, held = params
        self.r = r
        self.startp, self.endp = startp, endp
        self.obstr = obstr
        self.held = held

        self.rs_scale = RS_SCALE

        attr_inds = OrderedDict(
            [
                (r, [("pose", np.array([0, 1], dtype=np.int))]),
                (obstr, [("pose", np.array([0, 1], dtype=np.int))]),
                (held, [("pose", np.array([0, 1], dtype=np.int))]),
            ]
        )

        self._param_to_body = {
            r: self.lazy_spawn_or_body(r, r.name, r.geom),
            obstr: self.lazy_spawn_or_body(obstr, obstr.name, obstr.geom),
            held: self.lazy_spawn_or_body(held, held.name, held.geom),
        }

        # f = lambda x: -self.distance_from_obj(x)[0]
        # grad = lambda x: -self.distance_from_obj(x)[1]

        neg_coeff = 1e2
        neg_grad_coeff = 1e-3
        ## so we have an expr for the negated predicate
        # f_neg = lambda x: neg_coeff*self.distance_from_obj(x)[0]
        # grad_neg = lambda x: neg_grad_coeff*self.distance_from_obj(x)[1]

        def f(x):
            if self.obstr.name == self.held.name:
                return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4)
            else:
                return -twostep_f([x[:6], x[6:12]], self.distance_from_obj, 6)

        def grad(x):
            if self.obstr.name == self.held.name:
                return -twostep_f([x[:4], x[4:8]], self.distance_from_obj, 4, grad=True)
            else:
                return -twostep_f(
                    [x[:6], x[6:12]], self.distance_from_obj, 6, grad=True
                )

        def f_neg(x):
            return -neg_coeff * f(x)

        def grad_neg(x):
            return -neg_grad_coeff * grad(x)

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        if self.held != self.obstr:
            col_expr_neg = Expr(f_neg, grad_neg)
        else:
            new_f_neg = lambda x: 0.0 * f(x)  # self.distance_from_obj(x)[0]
            new_grad_neg = lambda x: -grad(x)  # self.distance_from_obj(x)[1]
            col_expr_neg = Expr(new_f_neg, new_grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(ObstructsHolding, self).__init__(
            name, e, attr_inds, params, expected_param_types
        )
        # self.priority=1

    def resample(self, negated, time, plan):
        assert negated

        a = 0
        while a < len(plan.actions) and plan.actions[a].active_timesteps[1] <= time:
            a += 1

        if a >= len(plan.actions) or time == plan.actions[a].active_timesteps[0]:
            return None, None
        act = plan.actions[a]

        res = OrderedDict()
        attr_inds = OrderedDict()
        disp1 = self.obstr.pose[:, time] - self.held.pose[:, time]
        disp2 = self.obstr.pose[:, time] - self.r.pose[:, time]
        disp = disp1 if np.linalg.norm(disp1) < np.linalg.norm(disp2) else disp2
        if disp[0] == 0:
            orth = np.array([1.0, 0.0])
        elif disp[1] == 0:
            orth = np.array([0.0, 1.0])
        else:
            orth = np.array([1.0 / disp[0], -1.0 / disp[1]])
        disp += 1e-4

        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        long_disp = self.r.pose[:, et] - self.r.pose[:, st]
        long_disp /= np.linalg.norm(long_disp)
        d1, d2 = long_disp.dot(orth), long_disp.dot(-orth)

        if d1 > d2:
            w1, w2 = 0.1, 0.9
        else:
            w1, w2 = 0.9, 0.1
        orth *= np.random.choice([-1.0, 1.0], p=[w1, w2])
        orth = orth / np.linalg.norm(orth)

        rdisp = (
            -(self.obstr.geom.radius + self.held.geom.radius + self.dsafe + 2e-1)
            * disp
            / np.linalg.norm(disp)
        )
        orth = rdisp + np.random.uniform(0.2, 0.5) * orth
        # orth *= np.random.uniform(1.2, 1.8) * (self.obstr.geom.radius + self.r.geom.radius)
        # orth += np.random.uniform([-0.15, 0.15], [-0.15, 0.15])

        # ## assumes that self.startp, self.endp and target are all symbols
        # t_local = 0
        # for param in [self.startp, self.endp]:
        #     ## there should only be 1 target that satisfies this
        #     ## otherwise, choose to fail here
        #     val, inds = sample_pose(plan, param, self.r, self.rs_scale)
        #     if val is None:
        #         continue
        #     res.extend(val[inds].flatten().tolist())
        #     ## inds[0] returns the x values of the indices which is what we care
        #     ## about, because the y values correspond to time.
        #     attr_inds[param] = [('value', inds[0])]

        new_robot_pose = self.r.pose[:, time] + orth
        new_held_pose = self.held.pose[:, time] + orth
        # add_to_attr_inds_and_res(time, attr_inds, res, self.r, [('pose', new_robot_pose)])
        # add_to_attr_inds_and_res(time, attr_inds, res, self.held, [('pose', new_held_pose)])
        st = max(max(time - 3, 0), act.active_timesteps[0])
        et = min(min(time + 3, plan.horizon - 1), act.active_timesteps[1])
        for i in range(st, et):
            dist = float(np.abs(i - time))
            if i <= time:
                inter_rp = (dist / 3.0) * self.r.pose[:, st] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, st] + orth
                )
                inter_hp = (dist / 3.0) * self.held.pose[:, st] + (
                    (3.0 - dist) / 3.0
                ) * (self.held.pose[:, st] + orth)
            else:
                inter_rp = (dist / 3.0) * self.r.pose[:, et] + ((3.0 - dist) / 3.0) * (
                    self.r.pose[:, et] + orth
                )
                inter_hp = (dist / 3.0) * self.held.pose[:, et] + (
                    (3.0 - dist) / 3.0
                ) * (self.held.pose[:, et] + orth)

            add_to_attr_inds_and_res(i, attr_inds, res, self.r, [("pose", inter_rp)])
            add_to_attr_inds_and_res(i, attr_inds, res, self.held, [("pose", inter_rp)])
        return res, attr_inds

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def distance_from_obj(self, x, n_steps=0):
        # x = [rpx, rpy, obstrx, obstry, heldx, heldy]
        b0 = self._param_to_body[self.r]
        b1 = self._param_to_body[self.obstr]

        pose_r = x[0:2]
        pose_obstr = x[2:4]

        b0.set_pose(pose_r)
        b1.set_pose(pose_obstr)

        collisions1 = p.getClosestPoints(b0.body_id, b1.body_id, contact_dist)
        col_val1, jac01 = self._calc_grad_and_val(
            self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1
        )

        if self.obstr.name == self.held.name:
            ## add dsafe to col_val1 b/c we're allowed to touch, but not intersect
            ## 1e-3 is there because the collision checker's value has some error.
            col_val1 -= self.dsafe + 1e-3
            val = np.array(col_val1)
            jac = jac01

        else:
            b2 = self._param_to_body[self.held]
            pose_held = x[4:6]
            b2.set_pose(pose_held)

            collisions2 = p.getClosestPoints(b2.body_id, b1.body_id, contact_dist)

            col_val2, jac21 = self._calc_grad_and_val(
                self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2
            )

            if col_val1 > col_val2:
                val = np.array(col_val1)
                jac = np.c_[jac01, np.zeros((1, 2))].reshape((1, 6))
            else:
                val = np.array(col_val2)
                jac = np.c_[np.zeros((1, 2)), jac21[:, 2:], jac21[:, :2]].reshape(
                    (1, 6)
                )

        return val, jac


class WideObstructsHolding(ObstructsHolding):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(WideObstructsHolding, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 0.2


class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, self.can, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = 1e1 * np.array([[1, 0, -1, 0, -1, 0], [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(InGripper, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )


class Retreat(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, self.can, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = 1e1 * np.array([[1, 0, 0.5, 0, -1, 0, 0, 0], [0, 1, 0, 0.5, 0, -1, 0, 0]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(Retreat, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            priority=-2,
            active_range=(0, 1),
        )


class GraspValid(ExprPredicate):

    # GraspValid RobotPose Target Grasp

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.rp, self.target, self.grasp = params
        attr_inds = OrderedDict(
            [
                (self.rp, [("value", np.array([0, 1], dtype=np.int))]),
                (self.target, [("value", np.array([0, 1], dtype=np.int))]),
                (self.grasp, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0], [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2, 1)))

        super(GraspValid, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=0
        )


class RobotStationary(ExprPredicate):

    # Stationary, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(RobotStationary, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class RobotRetreat(ExprPredicate):

    # Stationary, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c, self.g = params
        attr_inds = OrderedDict(
            [
                (self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                (self.g, [("value", np.array([0, 1], dtype=np.int))]),
            ]
        )
        A = np.array([[1, 0, 1, 0, -1, 0, 0, 0], [0, 1, 0, 1, 0, -1, 0, 0]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(RobotRetreat, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class Stationary(ExprPredicate):

    # Stationary, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        (self.c,) = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(Stationary, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryNEq(ExprPredicate):

    # StationaryNEq, Can, Can
    # Assuming robot only holding one object,
    # it checks whether the can in the first argument is stationary
    # if that first can is not the second can which robot is holding

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c, self.c_held = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        if self.c.name == self.c_held.name:
            A = np.zeros((1, 4))
            b = np.zeros((1, 1))
        else:
            A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
            b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class StationaryW(ExprPredicate):

    # StationaryW, Wall(Obstacle)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        (self.w,) = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class IsMP(ExprPredicate):

    # IsMP Robot

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]])
        b = np.zeros((4, 1))
        e = LEqExpr(AffExpr(A, b), dmove * np.ones((4, 1)))
        super(IsMP, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class VelWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("vel", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = dmove * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class AccWithinBounds(At):

    # RobotAt Robot Can

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.c = params
        attr_inds = OrderedDict([(self.r, [("acc", np.array([0, 1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((4, 1))
        val = 2.5e1 * np.ones((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)


class VelValid(ExprPredicate):

    # VelValid Robot

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("vel", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 1, 0, 0],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), dmove * np.ones((4, 1)))
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class Decelerating(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("vel", np.array([0, 1], dtype=np.int))]),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), b.copy())
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class Accelerating(ExprPredicate):
    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (self.r, [("vel", np.array([0, 1], dtype=np.int))]),
            ]
        )
        A = np.array(
            [
                [1, 0, -1, 0],
                [0, 1, 0, -1],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), b.copy())
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class VelValid(ExprPredicate):

    # VelValid Robot

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        (self.r,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("pose", np.array([0, 1], dtype=np.int)),
                        ("vel", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )
        A = np.array(
            [
                [-1, 0, 1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 1, 0, 0],
            ]
        )
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), dmove * np.ones((4, 1)))
        super(VelValid, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )


class AccValid(VelValid):

    # AccValid Robot

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, dmove=dmove
    ):
        super(AccValid, self).__init__(
            name, params, expected_param_types, env, debug, dmove
        )
        self.attr_inds = OrderedDict(
            [
                (
                    self.r,
                    [
                        ("vel", np.array([0, 1], dtype=np.int)),
                        ("acc", np.array([0, 1], dtype=np.int)),
                    ],
                ),
            ]
        )
