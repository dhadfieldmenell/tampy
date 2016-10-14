import unittest
from core.internal_repr import parameter
from core.util_classes import robot_predicates, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.param_setup import ParamSetup
import numpy as np

class TestRobotPredicates(unittest.TestCase):

    # Begin of the test
    def test_at(self):

        # At, Can, Target

        can = ParamSetup.setup_blue_can()
        target = ParamSetup.setup_target()
        pred = robot_predicates.At("testpred", [can, target], ["Can", "Target"])
        self.assertEqual(pred.get_type(), "At")
        # target is a symbol and doesn't have a value yet
        self.assertFalse(pred.test(time=0))
        can.pose = np.array([[3, 3, 5, 6],
                                  [6, 6, 7, 8],
                                  [6, 6, 4, 2]])
        can.rotation = np.zeros((3, 4))
        target.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())
        # Test timesteps
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At blue_can target)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At blue_can target)'.")
        #
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = robot_predicates.At("testpred", [can, sym], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At blue_can sym)'.")
        # Test rotation
        can.rotation = np.array([[1,2,3,4],
                                      [2,3,4,5],
                                      [3,4,5,6]])

        target.rotation = np.array([[2],[3],[4]])

        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_stationary(self):
        can = ParamSetup.setup_blue_can()
        pred = robot_predicates.Stationary("test_stay", [can], ["Can"])
        self.assertEqual(pred.get_type(), "Stationary")
        # Since pose of can is undefined, predicate is not concrete
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0], [0], [0]])
        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary blue_can)' at the timestep.")
        can.rotation = np.array([[1, 1, 1, 4, 4],
                                      [2, 2, 2, 5, 5],
                                      [3, 3, 3, 6, 6]])
        can.pose = np.array([[1, 2],
                                  [4, 4],
                                  [5, 7]])
        self.assertFalse(pred.test(time = 0))
        can.pose = np.array([[1, 1, 2],
                                  [2, 2, 2],
                                  [3, 3, 7]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary blue_can)' at the timestep.")
        can.pose = np.array([[1, 4, 5, 5, 5],
                                  [2, 5, 6, 6, 6],
                                  [3, 6, 7, 7, 7]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

    def test_stationaryNeq(self):
        can = ParamSetup.setup_blue_can()
        can_held = ParamSetup.setup_green_can()
        pred = robot_predicates.StationaryNEq("test_stay", [can, can_held], ["Can", "Can"])
        self.assertEqual(pred.get_type(), "StationaryNEq")
        # Since pose of can is undefined, predicate is not concrete
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0], [0], [0]])
        can_held.pose = np.zeros((3,5))
        can_held.rotation = np.zeros((3,5))
        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (StationaryNEq blue_can green_can)' at the timestep.")
        can.rotation = np.array([[1, 1, 1, 4, 4],
                                      [2, 2, 2, 5, 5],
                                      [3, 3, 3, 6, 6]])
        can.pose = np.array([[1, 2],
                                  [4, 4],
                                  [5, 7]])
        self.assertFalse(pred.test(time = 0))
        can.pose = np.array([[1, 1, 2],
                                  [2, 2, 2],
                                  [3, 3, 7]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (StationaryNEq blue_can green_can)' at the timestep.")
        can.pose = np.array([[1, 4, 5, 5, 5],
                                  [2, 5, 6, 6, 6],
                                  [3, 6, 7, 7, 7]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

        pred2 = robot_predicates.StationaryNEq("test_stay2", [can, can], ["Can", "Can"])
        self.assertTrue(pred2.test(time = 0))
        self.assertTrue(pred2.test(time = 1))
        self.assertTrue(pred2.test(time = 2))
        self.assertTrue(pred2.test(time = 3))

    def test_stationary_w(self):
        table = ParamSetup.setup_box()
        pred = robot_predicates.StationaryW("test_stay_w", [table], ["Table"])
        self.assertEqual(pred.get_type(), "StationaryW")
        # Since pose of can is undefined, predicate is not concrete
        self.assertFalse(pred.test(0))
        table.pose = np.array([[0], [0], [0]])
        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay_w: (StationaryW box)' at the timestep.")
        table.rotation = np.array([[1, 1, 1, 4, 4],
                                  [2, 2, 2, 5, 5],
                                  [3, 3, 3, 6, 6]])
        table.pose = np.array([[1, 2],
                              [4, 4],
                              [5, 7]])
        self.assertFalse(pred.test(time = 0))
        table.pose = np.array([[1, 1, 2],
                              [2, 2, 2],
                              [3, 3, 7]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay_w: (StationaryW box)' at the timestep.")
        table.pose = np.array([[1, 4, 5, 5, 5],
                              [2, 5, 6, 6, 6],
                              [3, 6, 7, 7, 7]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

    def test_collides(self):
        TEST_GRAD = False
        can = ParamSetup.setup_blue_can("obj")
        table = ParamSetup.setup_table()
        test_env = ParamSetup.setup_env()

        pred = robot_predicates.Collides("test_collides", [can, table], ["Can", "Table"], test_env, debug = True)
        self.assertEqual(pred.get_type(), "Collides")
        # Since parameters are not defined
        self.assertFalse(pred.test(0))
        # pose overlapped, collision should happens
        can.pose = table.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        #This gradient failed, table base link fails
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        can.pose = np.array([[0],[0],[1]])
        self.assertFalse(pred.test(0))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        can.pose = np.array([[0],[0],[.25]])
        self.assertFalse(pred.test(0))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        can.pose = np.array([[1],[1],[-.5]])
        self.assertFalse(pred.test(0))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        can.pose = np.array([[.5],[.5],[-.5]])
        self.assertFalse(pred.test(0))
        # This Gradient test didn't pass
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        can.pose = np.array([[.6],[.5],[-.5]])
        self.assertTrue(pred.test(0))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.rotation = np.array([[1],[0.4],[0.5]])
        self.assertFalse(pred.test(0))
        # This Gradient test passed
        pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.rotation = np.array([[.2],[.4],[.5]])
        self.assertTrue(pred.test(0))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[can].set_pose(can.pose, can.rotation)
        # pred._param_to_body[table].set_pose(table.pose, table.rotation)
        # import ipdb; ipdb.set_trace()
