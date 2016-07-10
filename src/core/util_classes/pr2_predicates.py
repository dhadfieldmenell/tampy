from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.matrix import Vector3d, PR2PoseVector
from errors_exceptions import PredicateException
from core.util_classes.pr2 import PR2
from sco.expr import AffExpr, EqExpr
import numpy as np


"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

DEFAULT_TOL=1e-4

class ExprPredicate(Predicate):

    """
    Predicates which are defined by a target value for a set expression.
    """

    def __init__(self, name, expr, attr_inds, tol, params, expected_param_types):
        """
        attr2inds is a dictionary that maps each parameter name to a
        list of (attr, active_inds) pairs. This defines the mapping
        from the primitive predicates of the params to the inputs to
        expr
        """
        super(ExprPredicate, self).__init__(name, params, expected_param_types)
        self.expr = expr
        self.attr_inds = attr_inds
        self.tol = tol

        self.x_dim = sum(len(active_inds)
                         for p_attrs in attr_inds.values()
                         for (_, active_inds) in p_attrs)
        self.x = np.zeros(self.x_dim)

    def get_param_vector(self, t):
        i = 0
        for p in self.params:
            for attr, ind_arr in self.attr_inds[p.name]:
                n_vals = len(ind_arr)
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
                if p.get_attr_type(attr) is not Vector3d:
                    raise PredicateException("attribute type not supported")

                if p.is_defined():
                    cur_dim = getattr(p, attr).shape[0]
                else:
                    cur_dim = 3
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
        tol=DEFAULT_TOL

        super(At, self).__init__(name, e, attr_inds, tol, params, expected_param_types)


class RobotAt(At):
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
                if p.get_attr_type(attr) is not PR2PoseVector:
                    raise PredicateException("attribute type not supported")
                if p.is_defined():
                    cur_dim = getattr(p, attr).shape[0]
                else:
                    cur_dim = 5
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
        tol = DEFAULT_TOL

        super(At, self).__init__(name, e, attr_inds, tol, params, expected_param_types)


class IsGP(Predicate):
    def __init__(self, name, params, expected_param_types):
        # IsGP, RobotPose, Can
        assert len(params) == 2
        for p in params:
            attr = None
            if hasattr(p, "pose"):
                attr = "pose"
            elif hasattr(p, "value"):
                attr = "value"
            if attr is not None:
                type = p.get_attr_type(attr)
                if type is not PR2PoseVector and type is not Vector3d:
                    raise PredicateException("attribute type not supported")
            else:
                raise PredicateException("attribute type not supported")
        super(IsGP, self).__init__(name, params, expected_param_types)



    def test(self, time):
        if not self.is_concrete():
            return False
        #TODO how to check whether this predicate is true?
        return True


class IsPDP(Predicate):
    def __init__(self, name, params, expected_param_types):
        # IsPDP, RobotPose, Location (Both params are symbol)
        assert len(params) == 2
        for p in params:
            attr = None
            if hasattr(p, "value"):
                attr = "value"
            if attr is not None:
                type = p.get_attr_type(attr)
                if type is not PR2PoseVector and type is not Vector3d:
                    raise PredicateException("attribute type not supported")
            else:
                raise PredicateException("attribute type not supported")
        super(IsPDP, self).__init__(name, params, expected_param_types)


class InGripper(Predicate):
    def __init__(self, name, params, expected_param_types):
        # InGripper, Can
        assert len(params) == 1
        for p in params:
            attr = None
            if hasattr(p, "pose"):
                attr = "pose"
            if attr is not None:
                type = p.get_attr_type(attr)
                if type is Vector3d:
                    raise PredicateException("attribute type not supported")
            else:
                raise PredicateException("attribute type not supported")
        super(InGripper, self).__init__(name, params, expected_param_types)


    def test(self, time):
        if not self.is_concrete():
            return False
        # TODO
        return False

class Obstructs(Predicate):
    def test(self, time):
        # TODO
        return True
