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

from collections import OrderedDict

"""
This file implements the predicates for the 2D NAMO domain.
"""

dsafe = 1e-1
dmove = 5e-1
contact_dist = 0

RS_SCALE = 0.5


class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = dsafe, debug = False, ind0=0, ind1=1):
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")
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

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.distance_from_obj(self.get_param_vector(t))
        self._debug = _debug

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
        jac0 = np.zeros(2)
        jac1 = np.zeros(2)
        results = []
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
            results.append((pt0, pt1, distance))

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
                chosen_pt0, chosen_pt1 = (pt0, pt1)
                chosen_distance = distance
                val = self.dsafe - distance
                jac0 = -1 * normal[0:2]
                jac1 = normal[0:2]

        if self._debug:
            print "options: ", results
            print "selected: ", chosen_pt0, chosen_pt1
            self._plot_collision(chosen_pt0, chosen_pt1, chosen_distance)

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
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1], dtype=np.int))]),
                                 (self.targ, [("value", np.array([0,1], dtype=np.int))])])

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
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0,1], dtype=np.int))]),
                                 (self.rp, [("value", np.array([0,1], dtype=np.int))])])

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
        attr_inds = OrderedDict([(rp, [("value", np.array([0,1], dtype=np.int))]),
                                 (targ, [("value", np.array([0,1], dtype=np.int))])])
        self._param_to_body = {rp: self.lazy_spawn_or_body(rp, rp.name, self.robot.geom),
                               targ: self.lazy_spawn_or_body(targ, targ.name, targ.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.ones((1, 1))*dsafe
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

class Collides(CollisionPredicate):

    # Collides Can Wall

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        self._param_to_body = {self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
                               self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.distance_from_obj(x)[0]
        def grad_neg(x):
            # print self.distance_from_obj(x)
            return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(Collides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)
        self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class RCollides(CollisionPredicate):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.r, self.w = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        self._param_to_body = {self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
                               self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        def f_neg(x):
            d = self.distance_from_obj(x)[0]
            # if d > 0:
            #     import pdb; pdb.set_trace()
            #     self.distance_from_obj(x)
            return d

        def grad_neg(x):
            # print self.distance_from_obj(x)
            return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)

        self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None



class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, RobotPose, Can;

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.r, self.startp, self.endp, self.c = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        self._param_to_body = {self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
                               self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom)}

        self.rs_scale = RS_SCALE

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
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(Obstructs, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=3)
        self.priority=1

    def resample(self, negated, time, plan):
        assert negated
        res = []
        attr_inds = OrderedDict()
        for param in [self.startp, self.endp]:
            ## there should only be 1 target that satisfies this
            ## otherwise, choose to fail here
            targets  = plan.get_param(InContact, 2, {0: self.r, 1:param})
            inds = param._free_attrs['value']
            if np.sum(inds) == 0: continue ## no resampling for this one
            if len(targets) == 1:
                random_dir = np.random.rand(2)
                random_dir = random_dir/np.linalg.norm(random_dir)
                val = targets[0] + random_dir*2*self.r.geom.radius
            elif len(targets) == 0:
                ## old generator -- just add a random perturbation
                val = np.random.normal(param.value[:, 0], scale=self.rs_scale)
            else:
                raise NotImplemented
            self.param.value[inds] = val[inds]
            res.extend(val[inds].flatten().tolist())
            attr_inds[param] = [('value', inds)]
        return np.array(res), attr_inds


    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

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

        attr_inds = OrderedDict([(r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (obstr, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (held, [("pose", np.array([0, 1], dtype=np.int))])
                                 ])

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
        self.priority=1

    def resample(self, negated, time, plan):

        assert negated
        res = []
        attr_inds = OrderedDict()
        # assumes that self.startp, self.endp and target are all symbols
        t_local = 0
        for param in [self.startp, self.endp]:
            ## there should only be 1 target that satisfies this
            ## otherwise, choose to fail here
            targets  = plan.get_param(InContact, 2, {0: self.r, 1:param})
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
            inds = np.where(param._free_attrs['value'])
            if np.sum(inds) == 0: continue ## no resampling for this one
            if len(targets) == 1:
                random_dir = np.random.rand(2,1)
                random_dir = random_dir/np.linalg.norm(random_dir)
                # assumes targets are symbols
                val = targets[0].value + random_dir*2*self.r.geom.radius
            elif len(targets) == 0:
                ## old generator -- just add a random perturbation
                val = np.random.normal(param.value[:, 0], scale=self.rs_scale)
            else:
                raise NotImplemented
            param.value[inds] = val[inds]
            res.extend(val[inds].flatten().tolist())
            # inds[0] returns the x values of the indices which is what we care
            # about, because the y values correspond to time.
            attr_inds[param] = [('value', inds[0])]
        return np.array(res), attr_inds

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

        pose_r = x[0:2]
        pose_obstr = x[2:4]

        b0.set_pose(pose_r)
        b1.set_pose(pose_obstr)

        collisions1 = self._cc.BodyVsBody(b0.env_body, b1.env_body)
        col_val1, jac0, jac1 = self._calc_grad_and_val(self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1)

        if self.obstr.name == self.held.name:
            ## add dsafe to col_val1 b/c we're allowed to touch, but not intersect
            col_val1 -= 2*self.dsafe
            val = np.array(col_val1)
            jac = np.r_[jac0, jac1].reshape((1, 4))
        else:
            b2 = self._param_to_body[self.held]
            pose_held = x[4:6]
            b2.set_pose(pose_held)

            collisions2 = self._cc.BodyVsBody(b2.env_body, b1.env_body)
            col_val2, jac2, jac1_ = self._calc_grad_and_val(self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2)

            if col_val1 > col_val2:
                val = np.array(col_val1)
                jac = np.r_[jac0, jac1, np.zeros(2)].reshape((1, 6))
            else:
                val = np.array(col_val2)
                jac = np.r_[np.zeros(2), jac1_, jac2].reshape((1, 6))

        return val, jac

class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, self.can, self.grasp = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.grasp, [("value", np.array([0, 1], dtype=np.int))])
                                ])
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2,1)))

        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types)

class GraspValid(ExprPredicate):

    # GraspValid RobotPose Target Grasp

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.rp, self.target,  self.grasp = params
        attr_inds = OrderedDict([(self.rp, [("value", np.array([0, 1], dtype=np.int))]),
                     (self.target, [("value", np.array([0, 1], dtype=np.int))]),
                     (self.grasp, [("value", np.array([0, 1], dtype=np.int))])
                     ])
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.array([[1, 0, -1, 0, -1, 0],
                      [0, 1, 0, -1, 0, -1]])
        b = np.zeros((2, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((2,1)))

        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)

class Stationary(ExprPredicate):

    # Stationary, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c,  = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

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
            A = np.array([[1, 0, -1, 0],
                          [0, 1, 0, -1]])
            b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryW(ExprPredicate):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)



class IsMP(ExprPredicate):

    # IsMP Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.r, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1],
                      [-1, 0, 1, 0],
                      [0, -1, 0, 1]])
        b = np.zeros((4, 1))

        e = LEqExpr(AffExpr(A, b), dmove*np.ones((4, 1)))
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)
