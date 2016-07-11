from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector3d, PR2PoseVector
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2 import PR2
from sco.expr import AffExpr, EqExpr
import numpy as np


"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.05, debug = False, ind0=0, ind1=1):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types)


    def lazy_spawn_or_body(self, param, name, geom):
        if param.openrave_body is not None:
            assert geom == param.openrave_body._geom
        else:
            param.openrave_body = OpenRAVEBody(self._env, name, geom)
        return param.openrave_body

    def distance_from_obj(self, x):
        # self._cc.SetContactDistance(self.dsafe + .1)
        self._cc.SetContactDistance(np.Inf)
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        pose0 = x[0:3]
        pose1 = x[3:6]
        b0.set_pose(pose0)
        b1.set_pose(pose1)

        collisions = self._cc.BodyVsBody(b0.env_body, b1.env_body)

        col_val, jac0, jac1 = self._calc_grad_and_val(p0.name, p1.name, pose0, pose1, collisions)
        val = np.array([col_val])
        jac = np.r_[jac0, jac1].reshape((1, 6))
        return val, jac


    def _calc_grad_and_val(self, name0, name1, pose0, pose1, collisions):
        val = -1 * float("inf")
        jac0 = None
        jac1 = None
        for c in collisions:
            linkA = c.GetLinkAParentName()
            linkB = c.GetLinkBParentName()

            if linkA == name0 and linkB == name1:
                pt0 = c.GetPtA()
                pt1 = c.GetPtB()
            elif linkB == name0 and linkA == name1:
                pt0 = c.GetPtB()
                pt1 = c.GetPtA()
            else:
                continue

            distance = c.GetDistance()
            normal = c.GetNormal()

            # plotting
            if self._debug:
                pt0[2] = 1.01
                pt1[2] = 1.01
                self._plot_collision(pt0, pt1, distance)
                print "pt0 = ", pt0
                print "pt1 = ", pt1
                print "distance = ", distance

            # if there are multiple collisions, use the one with the greatest penetration distance
            if self.dsafe - distance > val:
                val = self.dsafe - distance
                jac0 = -1 * normal[0:2]
                jac1 = normal[0:2]

        return val, jac0, jac1

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(1, 0, 0)))
            else:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(0, 0, 0)))

class At(ExprPredicate):

    # At, Can, Location

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.can, self.targ = params
        attr_inds = {self.can: [("pose", np.array([0,1,2], dtype=np.int))],
                     self.targ: [("value", np.array([0,1,2], dtype=np.int))],}

        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class RobotAt(At):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.r, self.rp = params
        attr_inds = {self.r: [("pose", np.array([0,1,2], dtype=np.int))],
                     self.rp: [("value", np.array([0,1,2], dtype=np.int))],}

        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class IsGP(CollisionPredicate):

    # IsGP, Robot, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 3
	self._env = env
        self.robot, self.robot_pose, self.can = params
        attr_inds = {self.robot: [("pose", np.array([0,1,2], dtype=np.int))],
                     self.robot_pose: [("value", np.array([0,1,2], dtype=np.int))],
                     self.can: [("value", np.array([0,1,2], dtype=np.int))]}
        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

class IsPDP(CollisionPredicate):

    # IsPDP, Robot, RobotPose, Can, Location

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 4
        self._env = env
        self.robot, self.robot_pose, self.can, self.location = params
        attr_inds = {self.robot: [("pose", np.array([0,1,2], dtype=np.int))],
                     self.robot_pose: [("value", np.array([0,1,2], dtype=np.int))],
                     self.can: [("value", np.array([0,1,2], dtype=np.int))],
                     self.location: [("value", np.array([0,1,2], dtype=np.int))]}
        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.location: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types):
        self.robot, self.can, self.grasp = params
        attr_inds = {self.robot: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     self.can: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1, 2], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((3,1)))

        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types)

    def test(self, time = 0):
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.expr.eval(self.get_param_vector(time))
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 3
        self._env = env
        r, rp, c = params
        attr_inds = {r: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     c: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     rp: []}
        self._param_to_body = {r: self.lazy_spawn_or_body(r, r.name, r.geom),
                               rp: self.lazy_spawn_or_body(rp, rp.name, r.geom),
                               c: self.lazy_spawn_or_body(c, c.name, c.geom)}
        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)
        super(Obstructs, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)
