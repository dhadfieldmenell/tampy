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
        return Environment()

    def setup_robot(self, name = "pr2"):
        attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
        return robot

    def setup_robot_pose(self, name = "robot_Pose"):
        attrs = {"name": [name], "value": [(0, 0, 0)], "_type": ["RobotPose"], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "value": matrix.Vector3d, "_type": str, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        rPose = parameter.Symbol(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        rPose.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        rPose.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T
        return rPose

    def setup_can(self, name = "can"):
        attrs = {"name": [name], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)
        return can

    def setup_target(self, name = "target"):
        # This is the target parameter
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["Target"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        target = parameter.Symbol(attrs, attr_types)
        return target

    def setup_ee_pose(self, name = "ee_pose"):
        attrs = {"name": [name], "value": ["undefined"], "rotation": [(0,0,0)], "_type": ["EEPose"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        ee_pose = parameter.Symbol(attrs, attr_types)
        return ee_pose

    # Begin of the test
    def test_expr_at(self):

        # At, Can, Target

        can = self.setup_can()
        target = self.setup_target()
        pred = pr2_predicates.At("testpred", [can, target], ["Can", "Target"])
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
            pred = pr2_predicates.At("testpred", [can, sym], ["Can", "Target"])
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'testpred: (At can sym)'.")
        # Test rotation
        can.rotation = np.array([[1,2,3,4],
                                      [2,3,4,5],
                                      [3,4,5,6]])

        target.rotation = np.array([[2],[3],[4]])

        self.assertFalse(pred.test(time=0))
        self.assertTrue(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertFalse(pred.test(time=3))

    def test_robot_at(self):

        # RobotAt, Robot, RobotPose

        robot = self.setup_robot()
        rPose = self.setup_robot_pose()
        pred = pr2_predicates.RobotAt("testRobotAt", [robot, rPose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "RobotAt")
        # Robot and RobotPose are initialized to the same pose
        self.assertTrue(pred.test(0))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 robot_Pose)'.")
        robot.pose = np.array([[3, 4, 5, 3],
                               [6, 5, 7, 6],
                               [6, 3, 4, 6]])
        rPose.value = np.array([[3, 4, 5, 6],
                                [6, 5, 7, 1],
                                [6, 3, 9, 2]])
        self.assertTrue(pred.is_concrete())
        robot.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        robot.lGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        rPose.rGripper = np.matrix([0.5, 0.4, 0.6, 0.5])
        robot.backHeight = np.matrix([0.2, 0.29, 0.18, 0.2])
        robot.rArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        robot.lArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        rPose.rArmPose = np.array([[0,0,0,0,0,0,0]]).T
        rPose.lArmPose = np.array([[0,0,0,0,0,0,0]]).T
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 robot_Pose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (RobotAt pr2 robot_Pose)'.")

        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_is_mp(self):
        robot = self.setup_robot()
        test_env = self.setup_environment()
        pred = pr2_predicates.IsMP("test_isMP", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "IsMP")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (IsMP pr2)' at the timestep.")
        # Getting lowerbound and movement step
        lbH_l, bH_m = pred.lower_limit[0], pred.joint_step[0]
        llA_l, lA_m = pred.lower_limit[1:8], pred.joint_step[1:8]
        lrA_l, rA_m = pred.lower_limit[9:16], pred.joint_step[9:16]
        llG_l, lG_m = pred.lower_limit[8], pred.joint_step[8]
        lrG_l, rG_m = pred.lower_limit[16], pred.joint_step[16]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[1,2,3,4,5,6,7],
                               [0,2,3,4,5,6,7],
                               [1,2,3,4,5,6,7]])

        # Arm pose is valid in the timestep: 0,1,2,3
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))

        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))

        # Gripper pose is valid in the timestep: 0,1,3,4,5
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Back height pose is always valid
        robot.backHeight = np.matrix([bH_m, bH_m, bH_m, bH_m, bH_m, bH_m, bH_m]).reshape((1,7))
        # Thus only timestep 1 and 3 are valid
        # import ipdb; ipdb.set_trace()
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertFalse(pred.test(4))
        self.assertFalse(pred.test(5))
        with self.assertRaises(PredicateException) as cm:
            pred.test(6)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (IsMP pr2)' at the timestep.")

    def test_within_joint_limit(self):
        robot = self.setup_robot()
        test_env = self.setup_environment()
        pred = pr2_predicates.WithinJointLimit("test_joint_limit", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "WithinJointLimit")
        # Getting lowerbound and movement step
        lbH_l, bH_m = pred.lower_limit[0], pred.joint_step[0]
        llA_l, lA_m = pred.lower_limit[1:8], pred.joint_step[1:8]
        lrA_l, rA_m = pred.lower_limit[9:16], pred.joint_step[9:16]
        llG_l, lG_m = pred.lower_limit[8], pred.joint_step[8]
        lrG_l, rG_m = pred.lower_limit[16], pred.joint_step[16]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[1,2,3,4,5,6,7],
                               [0,2,3,4,5,6,7],
                               [1,2,3,4,5,6,7]])

        # timestep 6 should fail
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # timestep 3 shold fail
        robot.backHeight = np.matrix([bH_m, bH_m, bH_m, -bH_m, bH_m, bH_m, bH_m]).reshape((1,7))
        # Thus timestep 1, 3, 6 should fail
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertFalse(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))

    def test_in_gripper(self):

        # InGripper, Robot, Can
        robot = self.setup_robot()
        can = self.setup_can()
        test_env = self.setup_environment()
        pred = pr2_predicates.InGripper("InGripper", [robot, can], ["Robot", "Can"], test_env)
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0,0,0]]).T
        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        # check the gradient of the implementations (correct)
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Now set can's pose and rotation to be the right things
        can.pose = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # A new robot arm pose
        robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        # Only the pos is correct, rotation is not yet right
        can.pose = np.array([[0.59152062, -0.71105108,  1.05144139]]).T
        self.assertFalse(pred.test(0))
        can.rotation = np.array([[0.02484449, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        # now rotate robot basepose
        robot.pose = np.array([[0,0,np.pi/3]]).T
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0.91154861,  0.15674634,  1.05144139]]).T
        self.assertFalse(pred.test(0))
        can.rotation = np.array([[1.07204204, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        can.rotation = np.array([[2.22529480e+00,   3.33066907e-16,  -5.23598776e-01]]).T
        self.assertFalse(pred.test(0))
        can.pose = np.array([[3.98707028e-01,   4.37093473e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations (correct)
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[robot].set_transparency(0.7)
        # pred._param_to_body[can].set_transparency(0.7)
        # test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_grasp_valid(self):

        # GraspValid EEPose Target

        ee_pose = self.setup_ee_pose()
        target = self.setup_target() # Target is the target
        pred = pr2_predicates.GraspValid("test_grasp_valid", [ee_pose, target], ["EEPose", "Target"])
        self.assertTrue(pred.get_type(), "GraspValid")
        # Since EEPose and Target are both undefined
        self.assertFalse(pred.test(0))
        ee_pose.value = np.array([[1,2,3],
                                       [2,3,4],
                                       [3,4,5]])
        target.value = np.array([[1,2,3],
                                      [2,9,4],
                                      [3,4,5]])
        ee_pose.rotation = np.array([[1,2,3],
                                          [2,3,3],
                                          [3,4,5]])
        target.rotation = np.array([[1,2,3],
                                         [2,3,4],
                                         [3,4,5]])
        # Since target and eepose are both symbol, and their first timestep value are the same, test should all pass
        self.assertTrue(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertTrue(pred.test(2))
        # set rotation of target to be wrong
        target.rotation = np.array([[0],[1],[3]])
        self.assertFalse(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertFalse(pred.test(2))

    def test_in_contact(self):
        # InContact robot EEPose target
        robot = self.setup_robot()
        ee_pose = self.setup_ee_pose()
        target = self.setup_target()
        test_env = self.setup_environment()
        test_can = self.setup_can()
        pred = pr2_predicates.InContact("test_in_contact", [robot, ee_pose, target], ["Robot", "EEPose", "Target"], test_env)
        self.assertTrue(pred.get_type(), "InContact")
        # Since EEPose and Target are both undefined
        self.assertFalse(pred.test(0))
        target.value = ee_pose.value = np.array([[0],[0],[0]])

        # By default, gripper fingers are not close enough to touch the can
        self.assertFalse(pred.test(0))
        robot.rGripper = np.matrix([0.46])
        self.assertTrue(pred.test(0))
        robot.rGripper = np.matrix([0.2])
        self.assertFalse(pred.test(0))

        # check the gradient of the implementations (correct) #TODO gradient not right
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)

        # ref_can_body = pred.lazy_spawn_or_body(test_can, test_can.name, test_can.geom)
        # test_can.pose = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        # pred._param_to_body[robot].set_transparency(0.7)
        # ref_can_body.set_transparency(0)
        # ref_can_body.set_pose(test_can.pose)
        # test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_ee_reachable(self):

        # EEUnreachable Robot, StartPose, EEPose

        robot = self.setup_robot()
        rPose = self.setup_robot_pose()
        ee_pose = self.setup_ee_pose()
        test_env = self.setup_environment()
        pred = pr2_predicates.EEReachable("test_ee_reachable", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        self.assertTrue(pred.get_type(), "EEReachable")
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))
        ee_pose.value = np.array([[0,0,0]]).T
        rPose.value = np.array([[0,0,0]]).T
        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        # check the gradient of the implementations (correct)
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Now set can's pose and rotation to be the right things
        ee_pose.value = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # A new robot arm pose
        robot.rArmPose = np.array([[-np.pi/3, np.pi/7, -np.pi/5, -np.pi/3, -np.pi/7, -np.pi/7, np.pi/5]]).T
        self.assertFalse(pred.test(0))
        # Only the pos is correct, rotation is not yet right
        ee_pose.value = np.array([[0.59152062, -0.71105108,  1.05144139]]).T
        self.assertFalse(pred.test(0))
        ee_pose.rotation = np.array([[0.02484449, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        # now rotate robot basepose
        robot.pose = np.array([[0,0,np.pi/3]]).T
        self.assertFalse(pred.test(0))
        ee_pose.value = np.array([[0.91154861,  0.15674634,  1.05144139]]).T
        self.assertFalse(pred.test(0))
        ee_pose.rotation = np.array([[1.07204204, -0.59793421, -0.68047349]]).T
        self.assertTrue(pred.test(0))
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/3]]).T
        self.assertFalse(pred.test(0))
        ee_pose.rotation = np.array([[2.22529480e+00,   3.33066907e-16,  -5.23598776e-01]]).T
        self.assertFalse(pred.test(0))
        ee_pose.value = np.array([[3.98707028e-01,   4.37093473e-01,   8.37601627e-01]]).T
        self.assertTrue(pred.test(0))
        # check the gradient of the implementations (correct)
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[robot].set_transparency(0.7)
        # pred._param_to_body[can].set_transparency(0.7)
        # test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_stationary(self):
        can = self.setup_can()
        pred = pr2_predicates.Stationary("test_stay", [can], ["Can"])
        self.assertEqual(pred.get_type(), "Stationary")
        # Since pose of can is undefined, predicate is not concrete
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0], [0], [0]])
        with self.assertRaises(PredicateException) as cm:
            pred.test(0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")
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
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_stay: (Stationary can)' at the timestep.")
        can.pose = np.array([[1, 4, 5, 5, 5],
                                  [2, 5, 6, 6, 6],
                                  [3, 6, 7, 7, 7]])
        self.assertFalse(pred.test(time = 0))
        self.assertFalse(pred.test(time = 1))
        self.assertFalse(pred.test(time = 2))
        self.assertTrue(pred.test(time = 3))

    def test_obstructs(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can

        robot = self.setup_robot()
        rPose = self.setup_robot_pose()
        can = self.setup_can()
        test_env = self.setup_environment()
        pred = pr2_predicates.Obstructs("test_obstructs", [robot, rPose, rPose, can], ["Robot", "RobotPose", "RobotPose", "Can"], test_env)
        self.assertEqual(pred.get_type(), "Obstructs")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Move can to the center of the gripper (touching -> should recognize as collision)
        can.pose = np.array([[.578,  -.127,   .838]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # Move can into the robot arm, should have collision
        can.pose = np.array([[.50,  -.3,   .838]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[robot].set_transparency(0.7)
        # pred._param_to_body[can].set_transparency(0.7)
        # pred._param_to_body[can].set_pose(can.pose, can.rotation)
        # test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_obstructs_holding(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can, Can

        robot = self.setup_robot()
        rPose = self.setup_robot_pose()
        can = self.setup_can("can1")
        can_held = self.setup_can("can2")
        test_env = self.setup_environment()
        pred = pr2_predicates.ObstructsHolding("test_obstructs", [robot, rPose, rPose, can, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env)
        self.assertEqual(pred.get_type(), "ObstructsHolding")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        rPose.value = can_held.pose = can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-2)
        # Move can to the center of the gripper (touching -> should recognize as collision)
        can.pose = np.array([[.578,  -.127,   .838]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # Move caheldn into the robot arm, should have collision
        can.pose = np.array([[.50,  -.3,   .838]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))

        pred2 = pr2_predicates.ObstructsHolding("test_obstructs_held", [robot, rPose, rPose, can_held, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env)
        rPose.value = can_held.pose = can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred2.test(0))
        can_held.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred2.test(0))
        # Move can to the center of the gripper (touching -> should allow touching)
        can_held.pose = np.array([[.578,  -.127,   .838]]).T
        self.assertFalse(pred2.test(0))
        self.assertTrue(pred2.test(0, negated = True))
        # Move can away from the gripper, no collision
        can_held.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred2.test(0))
        self.assertTrue(pred2.test(0, negated = True))
        # Move caheldn into the robot arm, should have collision
        can_held.pose = np.array([[.50,  -.3,   .838]]).T
        self.assertTrue(pred2.test(0))
        # self.assertFalse(pred.test(0, negated = True))
        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[robot].set_transparency(0.7)
        # pred._param_to_body[can_held].set_transparency(0.7)
        # pred._param_to_body[can_held].set_pose(can_held.pose, can_held.rotation)
        # test_env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_collides(self):
        pass
    def test_r_collides(self):
        pass

    #TODO test other stationary

if __name__ == "__main__":
    unittest.main()
