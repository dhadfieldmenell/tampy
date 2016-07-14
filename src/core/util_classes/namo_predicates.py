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

dsafe = 1e-2
dmove = 1e0


class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = dsafe, debug = False, ind0=0, ind1=1):
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
            assert self._env == param.openrave_body.env_body.GetEnv()
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
        pose0 = x[0:2]
        pose1 = x[2:4]
        b0.set_pose(pose0)
        b1.set_pose(pose1)

        assert b0.env_body.GetEnv() == b1.env_body.GetEnv()

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

        if jac0 is None or jac1 is None or val is None:
            import ipdb; ipdb.set_trace()
        return val, jac0, jac1

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(1, 0, 0)))
            else:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(0, 0, 0)))


class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        ## At Can Target
        self.can, self.targ = params
        attr_inds = {self.can: [("pose", np.array([0,1], dtype=np.int))],
                     self.targ: [("pose", np.array([0,1], dtype=np.int))],}

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class RobotAt(At):

    # RobotAt Robot RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        ## At Robot RobotPose
        self.r, self.rp = params
        attr_inds = {self.r: [("pose", np.array([0,1], dtype=np.int))],
                     self.rp: [("value", np.array([0,1], dtype=np.int))],}

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class InContact(CollisionPredicate):

    # InContact, Robot, RobotPose, Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, rp, targ = params
        attr_inds = {self.robot: [],
                     rp: [("value", np.array([0,1], dtype=np.int))],
                     targ: [("pose", np.array([0,1], dtype=np.int))]}
        self._param_to_body = {rp: self.lazy_spawn_or_body(rp, rp.name, self.robot.geom),
                               targ: self.lazy_spawn_or_body(targ, targ.name, targ.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)


class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, Can;

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 3
        self._env = env
        r, rp, c = params
        attr_inds = {r: [("pose", np.array([0, 1], dtype=np.int))],
                     c: [("pose", np.array([0, 1], dtype=np.int))],
                     rp: []}
        self._param_to_body = {r: self.lazy_spawn_or_body(r, r.name, r.geom),
                               rp: self.lazy_spawn_or_body(rp, rp.name, r.geom),
                               c: self.lazy_spawn_or_body(c, c.name, c.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.distance_from_obj(x)[0]
        def grad_neg(x):
            # print self.distance_from_obj(x)
            return self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)


        super(Obstructs, self).__init__(name, e, attr_inds, params, 
                                        expected_param_types, ind0=0, ind1=2)

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class ObstructsHolding(CollisionPredicate):

    # ObstructsHolding, Robot, RobotPose, Can, Can;
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 4
        self._env = env
        r, rp, obstr, held = params
        self.r = r
        self.obstr = obstr
        self.held = held

        if obstr.name == held.name:
            attr_inds = {r: [("pose", np.array([0, 1], dtype=np.int))],
                         obstr: [("pose", np.array([0, 1], dtype=np.int))]}

            f = lambda x: np.zeros((1, 1))
            grad = lambda x: np.zeros((1, 4))

            f_neg = f
            grad_neg = grad


        else:

            attr_inds = {r: [("pose", np.array([0, 1], dtype=np.int))],
                         obstr: [("pose", np.array([0, 1], dtype=np.int))],
                         held: [("pose", np.array([0, 1], dtype=np.int))],
                         rp: []}

            self._param_to_body = {r: self.lazy_spawn_or_body(r, r.name, r.geom),
                                   obstr: self.lazy_spawn_or_body(obstr, obstr.name, obstr.geom),
                                   held: self.lazy_spawn_or_body(held, held.name, held.geom)}
            f = lambda x: -self.distance_from_obj(x)[0]
            grad = lambda x: -self.distance_from_obj(x)[1]
            
            ## so we have an expr for the negated predicate
            f_neg = lambda x: self.distance_from_obj(x)[0]
            grad_neg = lambda x: self.distance_from_obj(x)[1]


        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(ObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types)

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

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
        col_val1, jac0, jac1 = self._calc_grad_and_val(self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1)
        collisions2 = self._cc.BodyVsBody(b2.env_body, b1.env_body)
        col_val2, jac2, jac1_ = self._calc_grad_and_val(self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2)

        val = np.array([col_val1 + col_val2])
        jac = np.r_[jac0, jac1 + jac1_, jac2].reshape((1, 6))

        return val, jac

class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, self.can, self.grasp = params
        attr_inds = {self.r: [("pose", np.array([0, 1], dtype=np.int))],
                     self.can: [("pose", np.array([0, 1], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2,1)))

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

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.rp, self.target,  self.grasp = params
        attr_inds = {self.rp: [("value", np.array([0, 1], dtype=np.int))],
                     self.target: [("pose", np.array([0, 1], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2,1)))

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

class Stationary(ExprPredicate):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c,  = params
        attr_inds = {self.c: [("pose", np.array([0, 1], dtype=np.int))]}
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryNEq(ExprPredicate):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c, self.c_held = params
        attr_inds = {self.c: [("pose", np.array([0, 1], dtype=np.int))]}
        if self.c.name == self.c_held.name:
            A = np.zeros((1, 4))
            b = np.zeros((1, 1))
        else:
            A = np.array([[1, 0, -1, 0],
                          [0, 1, 0, -1]])
            b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class IsMP(ExprPredicate):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = {self.r: [("pose", np.array([0, 1], dtype=np.int))]}
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1],
                      [-1, 0, 1, 0],
                      [0, -1, 0, 1]])
        b = np.zeros((4, 1))
        
        e = LEqExpr(AffExpr(A, b), dmove*np.ones((4, 1)))
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

    
        
