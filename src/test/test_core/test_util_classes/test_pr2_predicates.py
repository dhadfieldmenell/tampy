import unittest
from core.internal_repr import parameter
from core.util_classes import pr2_predicates, viewer, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.can import BlueCan, RedCan
from core.util_classes.pr2 import PR2
from openravepy import Environment
from sco import expr
import numpy as np

## exprs for testing
e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestPR2Predicates(unittest.TestCase):

    def setup(self):
        # Setting up initialization of basic parameter that will be used for the test
        attrs = {"name": ["pr2"], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": ['../models/pr2/pr2.zae'], "backHeight": [0.31], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        self.robot = parameter.Object(attrs, attr_types)

        attrs = {"name": ["funnyPose"], "value": [(0, 0, 0)], "_type": ["RobotPose"], "backHeight": [0.31], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "value": matrix.Vector3d, "_type": str, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        self.rPose = parameter.Symbol(attrs, attr_types)

        attrs = {"name": ["can"], "geom": (0.04, 0.2), "pose": ["undefined"], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": matrix.Vector3d, "_type": str}
        self.can = parameter.Object(attrs, attr_types)

        attrs = {"name": ["location"], "value": ["undefined"], "_type": ["Location"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "_type": str}
        self.location = parameter.Symbol(attrs, attr_types)

        attrs = {"value": ["undefined"], "_type": ["Grasp"], "name": ["grasp"]}
        attr_types = {"value": matrix.Vector3d, "_type": str, "name": str}
        self.grasp = parameter.Symbol(attrs, attr_types)

        self.test_env = Environment()

    def test_expr_at(self):

        # At, Can, Location
        self.setup()
        pred = pr2_predicates.At("testpred", [self.can, self.location], ["Can", "Location"])
        self.assertEqual(pred.get_type(), "At")

        self.assertFalse(pred.test(time=400))
        self.can.pose = np.array([[3, 4, 5, 6], [6, 5, 7, 8], [6, 3, 4, 2]])
        # location is a symbol and doesn't have a value yet
        #self.assertRaises(PredicateException, pred.test, time = 400)
        self.location.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())

        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = pr2_predicates.At("testpred", [self.can, sym], ["Can", "Location"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")

    def test_expr_robot_at(self):

        # RobotAt, Robot, RobotPose
        self.setup()
        pred = pr2_predicates.RobotAt("testRobotAt", [self.robot, self.rPose], ["Robot", "RobotPose"])

        self.assertEqual(pred.get_type(), "RobotAt")
        self.assertTrue(pred.test(0))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        self.robot.pose = np.array([[3, 4, 5, 3],
                                   [6, 5, 7, 6],
                                   [6, 3, 4, 6]])
        self.rPose.value = np.array([[3, 4, 5, 6],
                                     [6, 5, 7, 1],
                                     [6, 3, 9, 2]])
        self.assertTrue(pred.is_concrete())
        self.robot.lGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        self.robot.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        self.robot.backHeight = np.matrix([0.31, 0.29, 0.2, 0.31])
        self.robot.lArmPose = np.array([[0,0,0,0,0,0,0],
                                       [1,2,3,4,5,6,7],
                                       [7,6,5,4,3,2,1],
                                       [0,0,0,0,0,0,0]]).T
        self.robot.rArmPose = np.array([[0,0,0,0,0,0,0],
                                       [1,2,3,4,5,6,7],
                                       [7,6,5,4,3,2,1],
                                       [0,0,0,0,0,0,0]]).T

        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 funnyPose)'.")

        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_expr_is_gp(self):

        # IsGP, Robot, RobotPose, Can

        self.setup()
        pred = pr2_predicates.IsGP("testgp", [self.robot, self.rPose, self.can], ["Robot", "RobotPose", "Can"], self.test_env)
        self.assertEqual(pred.get_type(), "IsGP")
        # Since the pose of can is not defined, predicate is not concrete for the test
        self.assertFalse(pred.test(0))
        # Can's position overlap with robot's, test should fail
        self.can.pose = np.array([[0],[0],[0]])
        # self.assertFalse(pred.test(0))
        # Set pose of can to be at the center of robot's gripper -> np.array([[0.951],[-0.188],[1.100675]])
        self.can.pose = np.array([[0.951],[-0.188],[1.100675 - pred.dsafe]])
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        # Commented out for test
        # self.rPose.rArmPose = np.array([[0,0,0,0,0,0,1.57]]).T # turn the wrist to the side, test should fail
        # self.assertFalse(pred.test(0))

        # Test gradient
        pred.expr.expr.grad(pred.get_param_vector(0), True)
        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/4, 0, -np.pi/2, -np.pi/2, -np.pi, 0, np.pi/2]]).T

        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0.65781389, -0.18729289, 1.100675 - pred.dsafe]]).T # moved can to the center of gripper again
        self.assertTrue(pred.test(0))



        """
            Uncomment the following to see the robot
        """
        pred._param_to_body[self.rPose].set_transparency(0.7)
        pred._param_to_body[self.can].set_transparency(0.7)
        self.test_env.SetViewer("qtcoin")
        import ipdb; ipdb.set_trace()




    def test_expr_is_pdp(self):

        # IsPDP, Robot, RobotPose, Can, Location

        self.setup()
        test_env = Environment()
        pred = pr2_predicates.IsPDP("testpdp", [self.robot, self.rPose, self.can, self.location], ["Robot", "RobotPose", "Can", "Location"], self.test_env)
        self.assertEqual(pred.get_type(), "IsPDP")

        # Since the pose of can is not defined, predicate is not concrete for the test
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0],[0],[0]])
        # Can's position overlap with robot's, test should fail
        self.location.value = np.array([[0],[0],[0]])
        # self.assertFalse(pred.test(0))
        # Set pose of can to be at the center of robot's gripper -> np.array([[0.951],[-0.188],[1.100675]])
        self.location.value = np.array([[0.951],[-0.188],[1.100675 - pred.dsafe]])
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        self.rPose.rArmPose = np.array([[0,0,0,0,0,0,1.57]]).T # turn the wrist to the side, test should fail
        self.assertFalse(pred.test(0))
        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/4, 0, -np.pi/2, -np.pi/2, -np.pi, 0, np.pi/2]]).T

        # self.assertFalse(pred.test(0))
        self.location.value = np.array([[0.65781389, -0.18729289, 1.100675 - pred.dsafe]]).T # moved can to the center of gripper again
        self.assertTrue(pred.test(0))

        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.rPose].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()


    def test_expr_is_in_gripper(self):

        # InGripper, Robot, Can, Grasp

        self.setup()
        pred = pr2_predicates.InGripper("InGripper", [self.robot, self.can, self.grasp], ["Robot", "Can", "Grasp"])
        self.assertEqual(pred.get_type(), "InGripper")


    # TODO: test other predicates

if __name__ == "__main__":
    unittest.main()
