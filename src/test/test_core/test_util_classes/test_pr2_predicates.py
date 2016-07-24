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

    def setup_environment(self):
        self.test_env = Environment()

    def setup_robot(self):
        attrs = {"name": ["pr2"], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": ['../models/pr2/pr2.zae'], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        self.robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        self.robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        self.robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T

    def setup_robot_pose(self):
        attrs = {"name": ["funnyPose"], "value": [(0, 0, 0)], "_type": ["RobotPose"], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "value": matrix.Vector3d, "_type": str, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        self.rPose = parameter.Symbol(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        self.rPose.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        self.rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T

    def setup_can(self):
        attrs = {"name": ["can"], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        self.can = parameter.Object(attrs, attr_types)

    def setup_location(self):
        attrs = {"name": ["location"], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Location"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        self.location = parameter.Symbol(attrs, attr_types)

    def test_expr_at(self):

        # At, Can, Location

        self.setup_can()
        self.setup_location()
        pred = pr2_predicates.At("testpred", [self.can, self.location], ["Can", "Location"])
        self.assertEqual(pred.get_type(), "At")
        # location is a symbol and doesn't have a value yet
        self.assertFalse(pred.test(time=0))
        self.can.pose = np.array([[3, 3, 5, 6],
                                  [6, 6, 7, 8],
                                  [6, 6, 4, 2]])
        self.can.rotation = np.zeros((3, 4))
        self.location.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())
        # Test timesteps
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can location)'.")
        #
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = pr2_predicates.At("testpred", [self.can, sym], ["Can", "Location"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")
        # Test rotation
        self.can.rotation = np.array([[1,2,3,4],
                                      [2,3,4,5],
                                      [3,4,5,6]])

        self.location.rotation = np.array([[2],[3],[4]])

        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_robot_at(self):

        # RobotAt, Robot, RobotPose

        self.setup_robot()
        self.setup_robot_pose()
        pred = pr2_predicates.RobotAt("testRobotAt", [self.robot, self.rPose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "RobotAt")
        # Robot and RobotPose are initialized to the same pose
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
        self.robot.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        self.robot.lGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        self.rPose.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        self.robot.backHeight = np.matrix([0.2, 0.29, 0.18, 0.2])
        self.robot.rArmPose = np.array([[0,0,0,0,0,0,0],
                                       [1,2,3,4,5,6,7],
                                       [7,6,5,4,3,2,1],
                                       [0,0,0,0,0,0,0]]).T
        self.robot.lArmPose = np.array([[0,0,0,0,0,0,0],
                                       [1,2,3,4,5,6,7],
                                       [7,6,5,4,3,2,1],
                                       [0,0,0,0,0,0,0]]).T
        self.rPose.rArmPose = np.array([[0,0,0,0,0,0,0]]).T
        self.rPose.lArmPose = np.array([[0,0,0,0,0,0,0]]).T
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

    def test_is_gp(self):

        # IsGP, Robot, RobotPose, Can

        self.setup_robot()
        self.setup_robot_pose()
        self.setup_can()
        self.setup_environment()
        pred = pr2_predicates.IsGP("testgp", [self.robot, self.rPose, self.can], ["Robot", "RobotPose", "Can"], self.test_env)
        self.assertEqual(pred.get_type(), "IsGP")
        # Since the pose of can is not defined, predicate is not concrete for the test
        self.assertFalse(pred.test(0))
        # Set Can's position to be the same as robot's base pose, test should fail
        self.can.pose = np.array([[0],[0],[0]])
        self.assertFalse(pred.test(0))
        # Uncomment to check gradient below
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)

        # Set pose of can to be at the center of robot's gripper -> np.array([[0.951],[-0.188],[1.100675]])
        self.can.pose = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        # Test gripper facing
        self.rPose.rArmPose = np.array([[0,0,0,0,0,0,1.57]]).T # turn the wrist to the side, test should fail
        self.assertFalse(pred.test(0))

        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.rPose.backHeight = np.matrix([0.29])
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0.39827922, -0.53027259,  0.92760163]]).T
        # moved can to the center of gripper and test again
        self.assertTrue(pred.test(0))
        # Uncomment to check gradient below
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.rPose].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()




    def test_is_pdp(self):

        # IsPDP, Robot, RobotPose, Can, Location
        self.setup_robot()
        self.setup_robot_pose()
        self.setup_can()
        self.setup_location()
        self.setup_environment()
        pred = pr2_predicates.IsPDP("testpdp", [self.robot, self.rPose, self.can, self.location], ["Robot", "RobotPose", "Can", "Location"], self.test_env)
        self.assertEqual(pred.get_type(), "IsPDP")
        # Since the pose of can is not defined, predicate is not concrete
        self.assertFalse(pred.test(0))
        # Set Can's position to be the same as robot's base pose, test should fail
        self.can.pose = np.array([[0],[0],[0]])
        self.location.value = np.array([[0],[0],[0]])
        self.assertFalse(pred.test(0))
        # Check gradient for the initial pose
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Set pose of can to be at the center of robot's gripper
        self.location.value = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        # Test gripper facing
        # turn the wrist to the side, test should fail
        self.rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi]]).T
        self.assertFalse(pred.test(0))
        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.rPose.backHeight = np.matrix([0.29])
        self.assertFalse(pred.test(0))
        self.location.value = np.array([[0.39827922, -0.53027259,  0.92760163]]).T
        # moved can to the center of gripper and test again
        self.assertTrue(pred.test(0))
        # Test gradient on the new pose
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.rPose].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()


    def test_in_gripper(self):

        # InGripper, Robot, Can
        self.setup_robot()
        self.setup_can()
        self.setup_environment()
        pred = pr2_predicates.InGripper("InGripper", [self.robot, self.can], ["Robot", "Can"], self.test_env)
        self.assertEqual(pred.get_type(), "InGripper")
        # Since the pose of the can and grasp is not defined, predicate is not concrete
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0],[0],[0]])
        self.can.rotation = np.array([[0],[0],[0]])
        # Can's initial position is not right
        self.assertFalse(pred.test(0))
        # check the gradient of the implementations
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Set can's pose on robot's gripper
        self.can.pose = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        self.can.rotation = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # Change rotation of the can (pose is right, but rotation is wrong)
        self.can.rotation = np.array([[np.pi/8],[np.pi/4],[np.pi/16]])
        self.assertFalse(pred.test(0))
        # Turn robot's wrist
        self.robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        # Now turn the can to the same direction as robot gripper
        self.can.rotation = np.array([[1.17810, 0, -0.52360]]).T
        self.assertTrue(pred.test(0))
        # Now randomly set a new pose
        self.robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        # Tune the can's rotation, now position is still wrong so test should fail
        self.can.rotation = np.array([[0.02484, -0.59793, -0.68047]]).T
        self.assertFalse(pred.test(0))
        # Setting pose back to robot's gripper, Test should work
        self.can.pose = np.array([[0.59152062, -0.71105108,  1.05144139]]).T
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations
        # pred.exprs[0].expr.grad(pred.get_param_vector(0), True, 1e-2)
        # pred.exprs[1].expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.robot].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_stationary(self):
        self.setup_can()
        pred = pr2_predicates.Stationary("test_stay", [self.can], ["Can"])
        self.assertEqual(pred.get_type(), "Stationary")
        # Since pose of can is undefined, predicate is not concrete
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0], [0], [0]])
        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")
        self.can.rotation = np.array([[1, 1, 1, 4, 4],
                                      [2, 2, 2, 5, 5],
                                      [3, 3, 3, 6, 6]])
        self.can.pose = np.array([[1, 2],
                                  [4, 4],
                                  [5, 7]])
        self.assertFalse(pred.test(time = 0))
        self.can.pose = np.array([[1, 1, 2],
                                  [2, 2, 2],
                                  [3, 3, 7]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")
        self.can.pose = np.array([[1, 4, 5, 5, 5],
                                  [2, 5, 6, 6, 6],
                                  [3, 6, 7, 7, 7]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))


    # TODO: test other predicates

if __name__ == "__main__":
    unittest.main()
