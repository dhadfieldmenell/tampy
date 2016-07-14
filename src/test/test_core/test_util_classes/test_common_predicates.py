import unittest
from core.internal_repr import parameter
from core.util_classes.matrix import Vector2d
from core.util_classes import common_predicates
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import circle
from errors_exceptions import PredicateException
from sco import expr
import numpy as np
from openravepy import Environment

N = 10

## exprs for testing
e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestCommonPredicates(unittest.TestCase):

    def test_expr_pred(self):
        ## test initialization: x_dim computed correctly
        ## test
        radius = 1
        attrs = {"name": ["can"], "geom": [radius], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": int, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        p1.pose = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": int, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)
        p2.value = np.array([2, 2, 2])

        ## ExprPred Construction
        attr_inds = [(p1, ("pose", np.array([0], dtype=np.int)))]
        e = expr.EqExpr(e1, np.array([2]))
        pred = common_predicates.ExprPredicate("expr_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])

        # with self.assertRaises(NotImplementedError):
        #     pred.get_expr(None, None)

        ## get_param_vector
        self.assertEqual(pred.x_dim, 1)
        self.assertTrue(np.allclose(pred.get_param_vector(0), [1]))
        self.assertTrue(np.allclose(pred.get_param_vector(1), [2]))
        self.assertTrue(np.allclose(pred.get_param_vector(2), [3]))

        ## unpacking
        unpacked = pred.unpack([10])
        self.assertTrue("can" in unpacked)
        self.assertFalse("sym" in unpacked)
        self.assertEqual(len(unpacked["can"]), 1)
        self.assertEqual(("pose", [10]), unpacked["can"][0])

    def test_expr_pred_eq(self):
        ## test (multiple tols)
        ## grad
        tols = [1e-8, 1e-3, 1e-2]
        radius = 1
        attrs = {"name": ["can"], "geom": [radius], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": float, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        p1.pose = np.array([[1, 2, 3], [1 + tols[1], 2 + tols[1], 3], [1 + tols[2], 2 + tols[2], 3]]).T
        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": float, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)
        # The [2,3] in p2's pose shouldn't be used.
        p2.value = np.array([[1, 2], [2, 3]], dtype=np.float64).T

        ## pred is p1.pose[:1] = p2.value
        attr_inds = [(p1, ("pose", np.array([0, 1], dtype=np.int))),
                     (p2, ("value", np.array([0, 1], dtype=np.int)))]
        A = np.array([[1, 1, -1, -1]])
        b = np.array([0])

        aff_e = expr.AffExpr(A, b)
        e = expr.EqExpr(aff_e, np.array([[0.]]))
        pred0 = common_predicates.ExprPredicate("eq_expr_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred0.test(0))
        self.assertFalse(pred0.test(1))
        self.assertFalse(pred0.test(2))

        pred1 = common_predicates.ExprPredicate("eq_expr_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred1.test(0))
        self.assertFalse(pred1.test(1))
        self.assertFalse(pred1.test(2))

        pred2 = common_predicates.ExprPredicate("eq_expr_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred2.test(1))
        self.assertFalse(pred2.test(2))

    def test_expr_pred_leq(self):
        ## test (multiple tols)
        ## grad
        tols = [1e-8, 1e-4, 1e-2]
        radius = 1
        attrs = {"name": ["can"], "geom": [radius], "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": float, "_type": str}
        p1 = parameter.Object(attrs, attr_types)
        p1.pose = np.array([[1, 2, 3], [1 + tols[1], 2 + tols[1], 3], [1 + tols[2], 2 + tols[2], 3]]).T
        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": float, "_type": str}
        p2 = parameter.Symbol(attrs, attr_types)
        p2.value = np.array([[1, 2], [2, 3]], dtype=np.float64).T


        ## pred is p1.pose[:1] = p2.value
        attr_inds = [(p1, ("pose", np.array([0, 1], dtype=np.int))),
                     (p2, ("value", np.array([0, 1], dtype=np.int)))]
        A = np.array([[1, 1, -1, -1]])
        b = np.array([0])

        aff_e = expr.AffExpr(A, b)
        e = expr.LEqExpr(aff_e, np.array([[0.]]))
        pred0 = common_predicates.ExprPredicate("leq_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred0.test(0))
        self.assertFalse(pred0.test(1))
        self.assertFalse(pred0.test(2))

        pred1 = common_predicates.ExprPredicate("leq_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred1.test(0))
        self.assertFalse(pred1.test(1))
        self.assertFalse(pred1.test(2))

        pred2 = common_predicates.ExprPredicate("leq_pred", e, attr_inds, [p1, p2], ["Can", "Sym"])
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred2.test(1))
        self.assertFalse(pred2.test(2))

        ## its LEq, so increasing value for sym should make everything work
        p2.value += 5
        self.assertTrue(pred0.test(0))
        self.assertTrue(pred0.test(1))
        self.assertTrue(pred0.test(2))
        self.assertTrue(pred1.test(0))
        self.assertTrue(pred1.test(1))
        self.assertTrue(pred1.test(2))
        self.assertTrue(pred2.test(0))
        self.assertTrue(pred2.test(1))
        self.assertTrue(pred2.test(2))




if __name__ is "__main__":
    unittest.main()
