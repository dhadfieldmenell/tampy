from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.matrix import Vector2d
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np
from openravepy import Environment
import ctrajoptpy
from core.util_classes.circle import GreenCircle

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

DEFAULT_TOL=1e-6

class ExprPredicate(Predicate):

    """
    Predicates which are defined by a target value for a set expression.
    """

    def __init__(self, name, expr, attr_inds, params, expected_param_types):
        """
        attr2inds is a dictionary that maps each parameter name to a
        list of (attr, active_inds) pairs. This defines the mapping
        from the primitive predicates of the params to the inputs to
        expr
        """
        super(ExprPredicate, self).__init__(name, params, expected_param_types)
        self.expr = expr
        self.attr_inds = attr_inds
        self.tol = DEFAULT_TOL

        self.x_dim = sum(len(active_inds)
                         for p_attrs in attr_inds.values()
                         for (_, active_inds) in p_attrs)
        self.x = np.zeros(self.x_dim)


    def get_expr(self, pred_dict, action_preds):
        """
        Returns an expr or None

        pred_dict is a dictionary containing
        - the Predicate object (self)
        - negated (Boolean): whether the predicated is negated
        - hl_info (string) which is "pre", "post" and "hl_state" if the
          predicate is a precondition, postcondition, or neither and part of the
          high level state respectively
        - active_timesteps (tuple of (start_time, end_time))

        action_preds is a list containing all the predicate dictionaries for
            the action get_expr is being called from.
        """
        raise NotImplementedError

    def get_param_vector(self, t):
        i = 0
        for p in self.params:
            for attr, ind_arr in self.attr_inds[p.name]:
                n_vals = len(ind_arr)

                if p.is_symbol():
                    self.x[i:i+n_vals] = getattr(p, attr)[ind_arr, 0]
                else:
                    self.x[i:i+n_vals] = getattr(p, attr)[ind_arr, t]
                i += n_vals
        return self.x

    def test(self, time):
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.expr.eval(self.get_param_vector(time), tol=self.tol)
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

    def unpack(self, y):
        """
        Gradient returned in a similar form to attr_inds
        {param_name: [(attr, (g1,...gi,...gn)]}
        gi are in the same order as the attr_inds list
        """
        res = {}
        i = 0
        for p in self.params:
            res[p.name] = []
            for attr, ind_arr in self.attr_inds[p.name]:
                n_vals = len(ind_arr)
                res[p.name].append((attr, y[i:i+n_vals]))
                i += n_vals
        return res

    def _grad(self, t):
        return self.expr.grad(self.get_param_vector(t))

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.05, debug = False):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types)

    def grasp(self, time):
        return (np.array(self.params[1].pose) - np.array(self.params[0].value))[:,time]

    def distance_from_obj(self, x):
        # self._cc.SetContactDistance(self.dsafe + .1)
        self._cc.SetContactDistance(np.Inf)
        p0 = self.params[0]
        p1 = self.params[1]
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
        assert len(params) == 2
        dims = -1
        attr_inds = {}
        for p in params:
            p_inds = []
            attr = None
            if hasattr(p, "pose"):
                attr = "pose"
            elif hasattr(p, "value"):
                attr = "value"
            if attr is not None:
                if p.is_defined():
                    cur_dim = getattr(p, attr).shape[0]
                else:
                    if p.get_attr_type(attr) is Vector2d:
                        cur_dim = 2
                    else:
                        raise PredicateException("attribute type not supported")
                p_inds.append((attr, np.array(range(cur_dim))))

            if dims == -1:
                dims = cur_dim
            else:
                assert dims == cur_dim
            attr_inds[p.name] = p_inds
        A = np.c_[np.eye(dims), -np.eye(dims)]
        b = np.zeros((dims, 1))
        val = np.zeros((dims, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

    def get_expr(self, pred_dict, action_preds):
        if pred_dict['negated']:
            return None
        else:
            return self.expr

class RobotAt(At):
    pass

class IsGP(CollisionPredicate):

    def __init__(self, name, params, expected_param_types, debug=False):
        #IsGP, Robot, RobotPose, Can
        assert len(params) == 3
        self._env = Environment()
        self.robot = params[0]
        gp_params = params[1:]
        expected_gp_param_types = expected_param_types[1:]
        attr_inds = {}
        self._param_to_body = {}
        for p in gp_params:
            if not p.is_symbol():
                assert hasattr(p, "geom")
                assert p.get_attr_type("pose") is Vector2d
                attr_inds[p.name] = [("pose", np.array([0, 1], dtype=np.int))]
                self._param_to_body[p] = OpenRAVEBody(self._env, p.name, p.geom)
            else:
                assert p.get_attr_type("value") is Vector2d
                attr_inds[p.name] = [("value", np.array([0, 1], dtype=np.int))]
                self._param_to_body[p] = OpenRAVEBody(self._env, p.name, self.robot.geom)

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1, 1))
        e = EqExpr(col_expr, val)
        super(IsGP, self).__init__(name, e, attr_inds, gp_params, expected_gp_param_types)



class IsPDP(CollisionPredicate):
    #IsGP, Robot, RobotPose, Target
    assert len(params) == 3
    self._env = Environment()
    self.robot = params[0]
    self.target_object = params[2]
    pdp_params = params[1:]#Select 2nd and 4th element
    expected_pdp_param_types = expected_param_types[1:]
    attr_inds = {}
    self._param_to_body = {}
    for p in gp_params:
        if not p.is_symbol():
            assert hasattr(p, "geom")
            assert p.get_attr_type("pose") is Vector2d
            attr_inds[p.name] = [("pose", np.array([0, 1], dtype=np.int))]
            self._param_to_body[p] = OpenRAVEBody(self._env, p.name, p.geom)
        else:
            assert p.get_attr_type("value") is Vector2d
            attr_inds[p.name] = [("value", np.array([0, 1], dtype=np.int))]
            self._param_to_body[p] = OpenRAVEBody(self._env, p.name, self.robot.geom)

    f = lambda x: self.distance_from_obj(x)[0]
    grad = lambda x: self.distance_from_obj(x)[1]

    col_expr = Expr(f, grad)
    val = np.zeros((1, 1))
    e = EqExpr(col_expr, val)
    super(IsGP, self).__init__(name, e, attr_inds, gp_params, expected_gp_param_types)

class InGripper(Predicate):
    def test(self, time):
        # TODO
        return False

class Obstructs(Predicate):
    def test(self, time):
        # TODO
        return True

class NotObstructs(CollisionPredicate):
    def __init__(self, name, params, expected_param_types, debug=False):
        assert len(params) == 2
        self._env = Environment()
        attr_inds = {}
        self._param_to_body = {}
        for p in params:
            assert not p.is_symbol()
            assert hasattr(p, "geom")
            assert p.get_attr_type("pose") is Vector2d
            attr_inds[p.name] = [("pose", np.array([0, 1], dtype=np.int))]
            self._param_to_body[p] = OpenRAVEBody(self._env, p.name, p.geom)

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)
        super(NotObstructs, self).__init__(name, e, attr_inds, params, expected_param_types)
