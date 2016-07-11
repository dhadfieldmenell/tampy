from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector2d
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np
from openravepy import Environment
import ctrajoptpy

"""
This file implements the predicates for the 2D NAMO domain.
"""

DEFAULT_TOL=1e-4

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.00, debug = False, ind0=0, ind1=1):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        tol = DEFAULT_TOL
        super(CollisionPredicate, self).__init__(name, e, attr_inds,tol, params, expected_param_types)

    def distance_from_obj(self, x):
        # self._cc.SetContactDistance(self.dsafe + .1)
        self._cc.SetContactDistance(np.Inf)
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        pose0 = x[0:2]
        pose1 = x[2:4]
        b0.set_pose(pose0)
        b1.set_pose(pose1)

        collisions = self._cc.BodyVsBody(b0.env_body, b1.env_body)

        col_val, jac0, jac1 = self._calc_grad_and_val(p0.name, p1.name, pose0, pose1, collisions)
        val = np.array([col_val])
        jac = np.r_[jac0, jac1].reshape((1, 4))
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
    def __init__(self, name, params, expected_param_types):
        ## At Can Target
        self.can, self.targ = params
        attr_inds = {self.can: [("pose", np.array([0,1], dtype=np.int))],
                     self.targ: [("value", np.array([0,1], dtype=np.int))],}

        A = np.c_[np.eye(dims), -np.eye(dims)]
        b = np.zeros((dims, 1))
        val = np.zeros((dims, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        tol = DEFAULT_TOL
        super(At, self).__init__(name, e, attr_inds,tol, params, expected_param_types)

    def get_expr(self, pred_dict, action_preds):
        if pred_dict['negated']:
            return None
        else:
            return self.expr

class RobotAt(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types):
        ## At Robot RobotPose
        self.r, self.rp = params
        attr_inds = {self.r: [("pose", np.array([0,1], dtype=np.int))],
                     self.rp: [("value", np.array([0,1], dtype=np.int))],}

        A = np.c_[np.eye(dims), -np.eye(dims)]
        b = np.zeros((dims, 1))
        val = np.zeros((dims, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        tol = DEFAULT_TOL
        super(At, self).__init__(name, e, attr_inds,tol, params, expected_param_types)

class InContact(CollisionPredicate):

    # InContact, Robot, RobotPose, Target

    def __init__(self, name, params, expected_param_types, dsafe = 0.00, debug=False):
        self._env = Environment()
        self.robot, rp, targ = params
        attr_inds = {self.robot: [],
                     rp: [("value", np.array([0,1], dtype=np.int))],
                     targ: [("pose", np.array([0,1], dtype=np.int))]}
        self._param_to_body = {rp: OpenRAVEBody(self._env, rp.name, self.robot.geom),
                               targ: OpenRAVEBody(self._env, targ.name, targ.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(IsGP, self).__init__(name, e, attr_inds, params, expected_param_types, dsafe, ind0=1, ind1=2)

class NotObstructs(CollisionPredicate):

    # NotObstructs, Robot, RobotPose, Can;
    def __init__(self, name, params, expected_param_types, dsafe = 0.00, debug=False):
        assert len(params) == 3
        self._env = Environment()
        r, rp, c = params
        attr_inds = {r: [("pose", np.array([0, 1], dtype=np.int))],
                     c: [("pose", np.array([0, 1], dtype=np.int))],
                     rp: []}
        self._param_to_body = {r: OpenRAVEBody(self._env, r.name, r.geom),
                               rp: OpenRAVEBody(self._env, rp.name, r.geom),
                               c: OpenRAVEBody(self._env, c.name, c.geom)}
        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)
        super(NotObstructs, self).__init__(name, e, attr_inds, params, expected_param_types, dsafe, ind0=1, ind1=2)

class NotObstructsHolding(CollisionPredicate):

    # NotObstructsHolding, Robot, RobotPose, Can, Can;
    def __init__(self, name, params, expected_param_types, dsafe = 0.00, debug=False):
        assert len(params) == 4
        self._env = Environment()
        r, rp, obstr, held = params
        self.r = r
        self.obstr = obstr
        self.held = held
        attr_inds = {r: [("pose", np.array([0, 1], dtype=np.int))],
                     obstr: [("pose", np.array([0, 1], dtype=np.int))],
                     holding: [("pose", np.array([0, 1], dtype=np.int))],
                     rp: []}
        self._param_to_body = {r: OpenRAVEBody(self._env, r.name, r.geom),
                               obstr: OpenRAVEBody(self._env, obstr.name, obstr.geom),
                               holding: OpenRAVEBody(self._env, holding.name, holding.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)
        super(NotObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types, dsafe)

    def distance_from_obj(self, x):
        # x = [rpx, rpy, obstrx, obstry, heldx, heldy]
        self._cc.SetContactDistance(np.Inf)
        b0 = self._param_to_body[self.r]
        b1 = self._param_to_body[self.obstr]
        b2 = self._param_to_body[self.held]
        pose_r = x[0:2]
        pose_obstr = x[2:4]
        pose_held = x[4:6]
        b0.set_pose(pose_r)
        b1.set_pose(pose_obstr)
        b2.set_pose(pose_held)

        collisions1 = self._cc.BodyVsBody(b0.env_body, b1.env_body)
        col_val1, jac0, jac1 = self._calc_grad_and_val(self.r.name, self.obstr.name, pose0, pose1, collisions1)
        collisions2 = self._cc.BodyVsBody(b2.env_body, b1.env_body)
        col_val2, jac2, jac1_ = self._calc_grad_and_val(self.held.name, self.obstr.name, pose2, pose1, collisions2)

        val = np.array([col_val1 + col_val2])
        jac = np.r_[jac0, jac1 + jac1_, jac2].reshape((1, 6))

        return val, jac

class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types, debug=False):
        assert len(params) == 3
        self._env = Environment()
        self.r, self.can, self.grasp = params
        attr_inds = {self.r: [("pose", np.array([0, 1], dtype=np.int))],
                     self.can: [("pose", np.array([0, 1], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros(2, 1)

        e = AffExpr(A, b)
        e = EqExpr(e, 0)

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

class GraspValid(ExprPredicate):

    # GraspValid RobotPose Target Grasp

    def __init__(self, name, params, expected_param_types, debug=False):
        self._env = Environment()
        self.rp, self.target,  self.grasp = params
        attr_inds = {self.rp: [("value", np.array([0, 1], dtype=np.int))],
                     self.target: [("value", np.array([0, 1], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros(2, 1)

        e = AffExpr(A, b)
        e = EqExpr(e, 0)

        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)


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
