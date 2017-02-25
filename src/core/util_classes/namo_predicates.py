from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.internal_repr.plan import Plan
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector2d
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np
from openravepy import Environment
import ctrajoptpy
from collections import OrderedDict

from pma.ll_solver import NAMOSolver

"""
This file implements the predicates for the 2D NAMO domain.
"""

dsafe = 1e-1
dmove = 7e-1
contact_dist = 0

RS_SCALE = 0.5
N_DIGS = 3


class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = dsafe, debug = False, ind0=0, ind1=1):
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1

        self._cache = {}
        self.n_cols = 1

        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types)

    def test(self, time, negated=False):
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.neg_expr.eval(self.get_param_vector(time), tol=self.tol, negated = (not negated))
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.distance_from_obj(self.get_param_vector(t))
        self._debug = _debug


    # @profile
    def distance_from_obj(self, x):
        flattened = tuple(x.round(N_DIGS).flatten())
        if flattened in self._cache and self._debug is False:
            return self._cache[flattened]
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

        col_val, jac01 = self._calc_grad_and_val(p0.name, p1.name, pose0, pose1, collisions)
        # val = np.array([col_val])
        val = col_val
        jac = jac01
        self._cache[flattened] = (val.copy(), jac.copy())
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
        for i, c in enumerate(collisions):
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
                print "normal = ", normal

            vals[i, 0] = self.dsafe - distance
            jacs[i, :2] = -1*normal[:2]
            jacs[i, 2:] = normal[:2]

        if self._debug:
            print "options: ", results
            print "selected: ", chosen_pt0, chosen_pt1
            print "selected distance: ", chosen_distance
            self._plot_collision(chosen_pt0, chosen_pt1, chosen_distance)

        # if jac0 is None or jac1 is None or val is None:
        #     import ipdb; ipdb.set_trace()

        return np.array(vals).reshape((self.n_cols, 1)), np.array(jacs).reshape((self.n_cols, 4))

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
        self.robot, self.rp, self.targ = params
        attr_inds = OrderedDict([(self.rp, [("value", np.array([0,1], dtype=np.int))]),
                                 (self.targ, [("value", np.array([0,1], dtype=np.int))])])
        self._param_to_body = {self.rp: self.lazy_spawn_or_body(self.rp, self.rp.name, self.robot.geom),
                               self.targ: self.lazy_spawn_or_body(self.targ, self.targ.name, self.targ.geom)}

        INCONTACT_COEFF = 1e1
        f = lambda x: INCONTACT_COEFF*self.distance_from_obj(x)[0]
        grad = lambda x: INCONTACT_COEFF*self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.ones((1, 1))*dsafe*INCONTACT_COEFF
        # val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types, debug=debug, ind0=1, ind1=2)

    def test(self, time, negated=False):
        return super(CollisionPredicate, self).test(time, negated)

class Collides(CollisionPredicate):

    # Collides Can Obstacle (wall)

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


        N_COLS = 8

        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)



        super(Collides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)
        self.n_cols = N_COLS
        # self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class RCollides(CollisionPredicate):

    # RCollides Robot Obstacle (Wall)

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

        N_COLS = 8
        col_expr = Expr(f, grad)
        val = np.zeros((N_COLS,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)
        self.n_cols = N_COLS

        # self.priority = 1

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
        # self.priority=1

    def resample(self, negated, time, plan):
        assert negated
        res = []
        attr_inds = OrderedDict()
        for param in [self.startp, self.endp]:
            val, inds = sample_pose(plan, param, self.r, self.rs_scale)
            if val is None:
                continue
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

def sample_pose(plan, pose, robot, rs_scale):
    targets  = plan.get_param('InContact', 2, {0: robot, 1:pose})
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    inds = np.where(pose._free_attrs['value'])
    if np.sum(inds) == 0: return None, None ## no resampling for this one
    if len(targets) == 1:
        # print "one target", pose
        random_dir = np.random.rand(2,1) - 0.5
        random_dir = random_dir/np.linalg.norm(random_dir)
                # assumes targets are symbols
        val = targets[0].value + random_dir*3*robot.geom.radius
    elif len(targets) == 0:
                ## old generator -- just add a random perturbation
        # print "no targets", pose
        val = np.random.normal(pose.value[:, 0], scale=rs_scale)[:, None]
    else:
        # import pdb; pdb.set_trace()
        raise NotImplementedError
    # print pose, val
    pose.value = val

    ## make the pose collision free
    _, collision_preds = plan.get_param('RCollides', 1, negated=True, return_preds=True)
    _, at_preds = plan.get_param('RobotAt', 1, {0: robot, 1:pose}, negated=False, return_preds=True)
    preds = [(collision_preds[0], True), (at_preds[0], False)]
    old_pose = robot.pose.copy()
    old_free = robot._free_attrs['pose'].copy()
    robot.pose = pose.value.copy()
    robot._free_attrs['pose'][:] = 1

    wall = collision_preds[0].params[1]
    old_w_pose = wall.pose.copy()
    wall.pose = wall.pose[:, 0][:, None]


    old_priority = [p.priority for p, n in preds]
    for p, n in preds:
        p.priority = -1
    p = Plan.create_plan_for_preds(preds, collision_preds[0]._env)
    s = NAMOSolver(transfer_norm='l2')
    success = s._solve_opt_prob(p, 0, resample=False, verbose=False)

    # print success


    ## debugging
    # import viewer
    # v = viewer.OpenRAVEViewer.create_viewer()
    # v.draw_plan_ts(p, 0)
    # print pose.value, val
    # import pdb; pdb.set_trace()



    ## restore the old values
    robot.pose = old_pose
    robot._free_attrs['pose'] = old_free
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
        # self.priority=1

    def resample(self, negated, time, plan):

        assert negated
        res = []
        attr_inds = OrderedDict()
        # assumes that self.startp, self.endp and target are all symbols
        t_local = 0
        for param in [self.startp, self.endp]:
            ## there should only be 1 target that satisfies this
            ## otherwise, choose to fail here
            val, inds = sample_pose(plan, param, self.r, self.rs_scale)
            if val is None:
                continue
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

        assert b0.env_body.GetEnv() == b1.env_body.GetEnv()

        collisions1 = self._cc.BodyVsBody(b0.env_body, b1.env_body)
        col_val1, jac01 = self._calc_grad_and_val(self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1)

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

            collisions2 = self._cc.BodyVsBody(b2.env_body, b1.env_body)
            col_val2, jac21 = self._calc_grad_and_val(self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2)

            if col_val1 > col_val2:
                val = np.array(col_val1)
                jac = np.c_[jac01, np.zeros((1, 2))].reshape((1, 6))
            else:
                val = np.array(col_val2)
                jac = np.c_[np.zeros((1, 2)), jac21[:, 2:], jac21[:, :2]].reshape((1, 6))

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
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

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
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class StationaryW(ExprPredicate):

    # StationaryW, Wall(Obstacle)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))



class IsMP(ExprPredicate):

    # IsMP Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False, dmove=dmove):
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
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
