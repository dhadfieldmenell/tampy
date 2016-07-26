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

    # These functions sets up the variable necessary for each test
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

    def setup_target(self):
        # This is the target parameter
        attrs = {"name": ["target"], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Target"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        self.target = parameter.Symbol(attrs, attr_types)

    def setup_ee_pose(self):
        attrs = {"name": ["ee_pose"], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["EEPose"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        self.ee_pose = parameter.Symbol(attrs, attr_types)

    # Begin of the test
    def test_expr_at(self):

        # At, Can, Target

        self.setup_can()
        self.setup_target()
        pred = pr2_predicates.At("testpred", [self.can, self.target], ["Can", "Target"])
        self.assertEqual(pred.get_type(), "At")
        # target is a symbol and doesn't have a value yet
        self.assertFalse(pred.test(time=0))
        self.can.pose = np.array([[3, 3, 5, 6],
                                  [6, 6, 7, 8],
                                  [6, 6, 4, 2]])
        self.can.rotation = np.zeros((3, 4))
        self.target.value = np.array([[3, 4, 5, 7], [6, 5, 8, 7], [6, 3, 4, 2]])
        self.assertTrue(pred.is_concrete())
        # Test timesteps
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testpred: (At can target)'.")
        #
        self.assertTrue(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

        attrs = {"name": ["sym"], "value": ["undefined"], "_type": ["Sym"]}
        attr_types = {"name": str, "value": str, "_type": str}
        sym = parameter.Symbol(attrs, attr_types)
        with self.assertRaises(ParamValidationException) as cm:
            pred = pr2_predicates.At("testpred", [self.can, sym], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")
        # Test rotation
        self.can.rotation = np.array([[1,2,3,4],
                                      [2,3,4,5],
                                      [3,4,5,6]])

        self.target.rotation = np.array([[2],[3],[4]])

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

    def test_is_mp(self):
        self.setup_robot()
        pred = pr2_predicates.IsMP("test_isMP", [self.robot], ["Robot"])
        self.assertEqual(pred.get_type(), "IsMP")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (IsMP pr2)' at the timestep.")
        b_m = pr2_predicates.BASE_MOVE
        j_m = pr2_predicates.JOINT_MOVE
        # Base pose is valid in the timestep: 1,2,3,4,5
        self.robot.pose = np.array([[1,2,3,4,5,6,7],
                                    [0,2,3,4,5,6,7],
                                    [1,2,3,4,5,6,7]])
        # Arm pose is valid in the timestep: 0,1,2,3
        self.robot.rArmPose = np.array([[0,     0,     0,     0,     0,     0,     0],
                                        [j_m,   j_m,   j_m,   j_m,   j_m,   j_m,   j_m],
                                        [2*j_m, 2*j_m, 2*j_m, 2*j_m, 2*j_m, 2*j_m, 2*j_m],
                                        [3*j_m, 3*j_m, 3*j_m, 3*j_m, 3*j_m, 3*j_m, 3*j_m],
                                        [4*j_m, 4*j_m, 4*j_m, 4*j_m, 4*j_m, 4*j_m, 4*j_m],
                                        [7*j_m, 2*j_m, 3*j_m, 7*j_m, 3*j_m, 6*j_m, 9*j_m],
                                        [2*j_m, 5*j_m, 0*j_m, 1*j_m, 8*j_m, 5*j_m, 3*j_m]]).T
        self.robot.lArmPose = self.robot.rArmPose.copy()
        # Gripper pose is valid in the timestep: 0,1,3,4,5
        self.robot.rGripper = np.matrix([0, j_m, 0, 4*j_m, 3*j_m, 4*j_m, 5*j_m])
        self.robot.lGripper = self.robot.rGripper.copy()
        # Back height pose is always valid
        self.robot.backHeight = np.matrix([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # Thus only timestep 1 and 3 are valid
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertFalse(pred.test(4))
        self.assertFalse(pred.test(5))
        with self.assertRaises(PredicateException) as cm:
            pred.test(6)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (IsMP pr2)' at the timestep.")

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
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)

        # Set pose of can to be at the center of robot's gripper -> np.array([[0.951],[-0.188],[1.100675]])
        self.can.pose = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.rPose.backHeight = np.matrix([0.29])
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0.39827922, -0.53027259,  0.92760163]]).T
        # moved can to the center of gripper and test again
        self.assertTrue(pred.test(0))
        # Uncomment to check gradient below
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)

        pred2 = pr2_predicates.IsGPRot("test_gp_rot", [self.robot, self.rPose, self.can], ["Robot", "RobotPose", "Can"], self.test_env)
        self.assertEqual(pred2.get_type(), "IsGPRot")
        # Test gripper facing
        self.rPose.rArmPose = np.array([[0,0,0,0,0,0,1.57]]).T # turn the wrist to the side, test should fail
        self.assertFalse(pred.test(0))
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.assertTrue(pred.test(0))

        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.rPose].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_is_pdp(self):

        # IsPDP, Robot, RobotPose, Can, Target
        self.setup_robot()
        self.setup_robot_pose()
        self.setup_can()
        self.setup_target()
        self.setup_environment()
        pred = pr2_predicates.IsPDP("testpdp", [self.robot, self.rPose, self.can, self.target], ["Robot", "RobotPose", "Can", "Target"], self.test_env)
        self.assertEqual(pred.get_type(), "IsPDP")
        # Since the pose of can is not defined, predicate is not concrete
        self.assertFalse(pred.test(0))
        # Set Can's position to be the same as robot's base pose, test should fail
        self.can.pose = np.array([[0],[0],[0]])
        self.target.value = np.array([[0],[0],[0]])
        self.assertFalse(pred.test(0))
        # Check gradient for the initial pose
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Set pose of can to be at the center of robot's gripper
        self.target.value = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        # By default setting, gripper is facing up, test should pass
        self.assertTrue(pred.test(0))
        #change arm pose again
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.rPose.backHeight = np.matrix([0.29])
        # self.assertFalse(pred.test(0))
        self.target.value = np.array([[0.39827922, -0.53027259,  0.92760163]]).T
        # moved can to the center of gripper and test again
        self.assertTrue(pred.test(0))
        # Test gradient on the new pose
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)

        pred2 = pr2_predicates.IsPDPRot("test_pdp_rot", [self.robot, self.rPose, self.can, self.target], ["Robot", "RobotPose", "Can", "Target"], self.test_env)
        self.assertEqual(pred2.get_type(), "IsPDPRot")
        # Test gripper facing
        self.rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi]]).T
        self.assertFalse(pred.test(0))
        self.rPose.rArmPose = np.array([[-np.pi/2, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/4, np.pi/2]]).T
        self.assertTrue(pred.test(0))

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
        self.can.pose = np.array([[0],[0],[0]])
        self.can.rotation = np.array([[0],[0],[0]])
        # Can's initial position is not right
        self.assertFalse(pred.test(0))
        # check the gradient of the implementations
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Set can's pose on robot's gripper
        self.can.pose = np.array([[0.57788757, -0.12674368,  0.83760163]]).T
        self.can.rotation = np.array([[0],[0],[0]])
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
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.robot].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_in_gripper_rot(self):
        # Test In GripperRot
        self.setup_robot()
        self.setup_can()
        self.setup_environment()
        pred = pr2_predicates.InGripperRot("InGripper_rot", [self.robot, self.can], ["Robot", "Can"], self.test_env)
        self.assertEqual(pred.get_type(), "InGripperRot")
        # Since pose of can is not defined
        self.assertFalse(pred.test(0))
        # can is initialized with right default rotation axis
        self.can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # Turn robot's wrist to the side
        self.robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        # set the right rotation
        self.can.rotation = np.array([[1.17809725e+00,  -2.49800181e-16,  -5.23598776e-01]]).T
        self.assertTrue(pred.test(0))
        # New robot arm pose
        self.robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[ 0.59152062, -0.71105108,  1.05144139]]).T
        self.can.rotation = np.array([[0.02484449, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[self.robot].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        #
        # import ipdb; ipdb.set_trace()

    def test_grasp_valid(self):

        # GraspValid EEPose Target

        self.setup_ee_pose()
        self.setup_target() # Target is the target
        pred = pr2_predicates.GraspValid("test_grasp_valid", [self.ee_pose, self.target], ["EEPose", "Target"])
        self.assertTrue(pred.get_type(), "GraspValid")
        # Since EEPose and Target are both undefined
        self.assertFalse(pred.test(0))
        self.ee_pose.value = np.array([[1,2,3],
                                       [2,3,4],
                                       [3,4,5]])
        self.target.value = np.array([[1,2,3],
                                      [2,9,4],
                                      [3,4,5]])
        self.ee_pose.rotation = np.array([[1,2,3],
                                          [2,3,3],
                                          [3,4,5]])
        self.target.rotation = np.array([[1,2,3],
                                         [2,3,4],
                                         [3,4,5]])
        # Since target and eepose are both symbol, and their first timestep value are the same, test should all pass
        self.assertTrue(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertTrue(pred.test(2))
        # set rotation of target to be wrong
        self.target.rotation = np.array([[0],[1],[3]])
        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertFalse(pred.test(2))

    def test_ee_reachable(self):

        # EEUnreachable Robot, StartPose, EEPose

        self.setup_robot()
        self.setup_robot_pose()
        self.setup_ee_pose()
        self.setup_environment()
        pred = pr2_predicates.EEReachable("test_ee_reachable", [self.robot, self.rPose, self.ee_pose], ["Robot", "RobotPose", "EEPose"], self.test_env)
        self.assertTrue(pred.get_type(), "EEReachable")
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))
        self.ee_pose.value = np.array([[0,0,0]]).T
        self.rPose.value = np.array([[0,0,0]]).T
        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        # check the gradient of the implementations
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Now set can's pose and rotation to be the right things
        self.ee_pose.value = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # A new robot arm pose
        self.robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        self.ee_pose.value = np.array([[0.59152062, -0.71105108,  1.05144139]]).T
        self.assertTrue(pred.test(0))
        # now rotate robot basepose
        self.robot.pose = np.array([[0,0,np.pi/3]]).T
        self.assertFalse(pred.test(0))
        self.ee_pose.value = np.array([[0.91154861,  0.15674634,  1.05144139]]).T
        self.assertTrue(pred.test(0))

    def test_ee_reachable_rot(self):

        # EEUnreachable Robot, StartPose, EEPose

        self.setup_robot()
        self.setup_robot_pose()
        self.setup_ee_pose()
        self.setup_environment()
        pred = pr2_predicates.EEReachableRot("test_ee_reachable_rot", [self.robot, self.rPose, self.ee_pose], ["Robot", "RobotPose", "EEPose"], self.test_env)
        self.assertTrue(pred.get_type(), "EEReachable")
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))
        self.ee_pose.value = np.array([[0,0,0]]).T
        self.rPose.value = np.array([[0,0,0]]).T
        # initialized facing is right
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Now set a new robot arm pose
        self.robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        self.ee_pose.rotation = np.array([[1.17809725e+00,  -2.49800181e-16,  -5.23598776e-01]]).T
        self.assertTrue(pred.test(0))
        # now rotate robot basepose
        self.robot.pose = np.array([[0,0,np.pi/3]]).T
        self.assertFalse(pred.test(0))
        self.ee_pose.rotation= np.array([[2.22529480e+00,   3.33066907e-16,  -5.23598776e-01]]).T
        self.assertTrue(pred.test(0))

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

    def test_obstructs(self):

        # Obstructs, Robot, RobotPose, Can

        self.setup_robot()
        self.setup_robot_pose()
        self.setup_can()
        self.setup_environment()
        pred = pr2_predicates.Obstructs("test_obstructs", [self.robot, self.rPose, self.can], ["Robot", "RobotPose", "Can"], self.test_env)
        self.assertEqual(pred.get_type(), "Obstructs")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        self.can.pose = np.array([[0],[0],[0]])
        # import ipdb; ipdb.set_trace()
        # self.assertTrue(pred.test(0))
        #
        # """
        #     Uncomment the following to see the robot
        # """
        # pred._param_to_body[self.rPose].set_transparency(0.7)
        # pred._param_to_body[self.can].set_transparency(0.7)
        # self.test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    unittest.main()
