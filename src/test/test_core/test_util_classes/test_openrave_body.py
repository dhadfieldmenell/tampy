import numpy as np
import unittest
from openravepy import Environment, KinBody,RaveCreateKinBody
from core.internal_repr import parameter
from core.util_classes.robots import PR2
from core.util_classes.items import BlueCan, RedCan, Obstacle, GreenCircle, BlueCircle
from core.util_classes.matrix import Vector2d, Vector3d, Vector7d, Value
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import OpenRAVEException
from core.util_classes import viewer
from core.parsing import parse_domain_config, parse_problem_config
import main
import time

N = 10

class DummyGeom(object):
    pass

class TestOpenRAVEBody(unittest.TestCase):
    def test_2D_base_pose_mat_trans(self):
        for i in range(N):
            pose = np.random.rand(2)
            mat = OpenRAVEBody.base_pose_2D_to_mat(pose)
            pose2 = OpenRAVEBody.mat_to_base_pose_2D(mat)
            self.assertTrue(np.allclose(pose, pose2))

    def test_base_pose_mat_trans(self):
        for i in range(N):
            pose = np.random.rand(3)
            mat = OpenRAVEBody.base_pose_to_mat(pose)
            pose2 = OpenRAVEBody.mat_to_base_pose(mat)
            self.assertTrue(np.allclose(pose, pose2))

    def test_exception(self):
        env = Environment()
        attrs = {"geom": [], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": DummyGeom, "pose": Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)
        with self.assertRaises(OpenRAVEException) as cm:
            green_body = OpenRAVEBody(env, "can0", green_can.geom)
        self.assertTrue("Geometry not supported" in cm.exception.message)


    def test_add_circle(self):
        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": GreenCircle, "pose": Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": BlueCircle, "pose": Vector2d, "_type": str, "name": str}
        blue_can = parameter.Object(attrs, attr_types)

        env = Environment()
        """
        to ensure the _add_circle and create_cylinder is working, uncomment the
        line below to make sure cylinders are being created
        """
        # env.SetViewer('qtcoin')

        green_body = OpenRAVEBody(env, "can0", green_can.geom)
        blue_body = OpenRAVEBody(env, "can1", blue_can.geom)

        green_body.set_pose([2,0])
        arr = np.eye(4)
        arr[0,3] = 2
        self.assertTrue(np.allclose(green_body.env_body.GetTransform(), arr))
        blue_body.set_pose(np.array([0,-1]))
        arr = np.eye(4)
        arr[1,3] = -1
        self.assertTrue(np.allclose(blue_body.env_body.GetTransform(), arr))

    def test_add_obstacle(self):
        attrs = {"geom": [], "pose": [(3, 5)], "_type": ["Obstacle"], "name": ["obstacle"]}
        attr_types = {"geom": Obstacle, "pose": Vector2d, "_type": str, "name": str}
        obstacle = parameter.Object(attrs, attr_types)

        env = Environment()
        """
        to ensure the _add_obstacle is working, uncomment the line below to make
        sure the obstacle is being created
        """
        # env.SetViewer('qtcoin')

        obstacle_body = OpenRAVEBody(env, obstacle.name, obstacle.geom)
        obstacle_body.set_pose([2,0])
        arr = np.eye(4)
        arr[0,3] = 2
        self.assertTrue(np.allclose(obstacle_body.env_body.GetTransform(), arr))

    def test_pr2_table(self):
        #TODO fix the pr2 domain problem
        #domain_fname, problem_fname = '../domains/can_domain/pr2.init', '../domains/can_domain/pr2.prob'
        #d_c = main.parse_file_to_dict(domain_fname)
        #self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        #self.p_c = main.parse_file_to_dict(problem_fname)
        #problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        """
            Uncomment the following to see things in the viewer
        """
        # view = viewer.OpenRAVEViewer()
        # robot = problem.init_state.params['dude']
        # table = problem.init_state.params['rll_table']
        # view.draw([robot, table], 0, 0.7)
        # import ipdb; ipdb.set_trace()

    def setup_environment(self):
        return Environment()

    def setup_robot(self, name = "pr2"):
        attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": Vector3d, "_type": str, "geom": PR2, "backHeight": Value, "lArmPose": Vector7d, "rArmPose": Vector7d, "lGripper": Value, "rGripper": Value}
        robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        R_ARM_INIT = [-1.832, -0.332, -1.011, -1.437, -1.1, 0, -3.074]
        L_ARM_INIT = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09]
        robot.rArmPose = np.array(R_ARM_INIT).reshape((7,1))
        robot.lArmPose = np.array(L_ARM_INIT).reshape((7,1))
        return robot

    def setup_can(self, name = "can"):
        attrs = {"name": [name], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": Vector3d, "rotation": Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)
        return can

    def test_get_ik_arm_pose(self):
        robot = self.setup_robot()
        can = self.setup_can()
        test_env = self.setup_environment()

        robot_body = OpenRAVEBody(test_env, robot.name, robot.geom)

        # robot.pose = np.array([-.5, 0, 0]).reshape((3,1))
        robot.pose = np.array([-0., 0, 0]).reshape((3,1))
        robot_body.set_pose(robot.pose)
        dof_value_map = {"backHeight": robot.backHeight,
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": robot.rArmPose.flatten(),
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)
        # can.pose = np.array([-0.31622543, -0.38142561,  1.19321209]).reshape((3,1))
        # can.pose = np.array([0.5, -0.2,  .8]).reshape((3,1))
        can.pose = np.array([.5, .2, .8]).reshape((3,1))
        # can.rotation = np.array([ 0.04588155, -0.38504402,  0.19207589]).reshape((3,1))
        can.rotation = np.array([ -.0, -0.,  0.]).reshape((3,1))
        can_body = OpenRAVEBody(test_env, can.name, can.geom)
        can_body.set_pose(can.pose, can.rotation)

        # view = viewer.OpenRAVEViewer()
        # view.draw([robot, can], 0, 0.7)
        #
        # arm_poses = robot_body.get_ik_arm_pose(can.pose.flatten(), can.rotation.flatten())
        # print arm_poses
        # import ipdb; ipdb.set_trace()
        # for arm_pose in arm_poses:
        #     robot.rArmPose = arm_pose.reshape((7,1))
        #     view.draw([robot, can], 0, 0.7)
        #     time.sleep(.1)
        # import ipdb; ipdb.set_trace()

    def test_ik_arm_pose2(self):
        env = Environment()
        attrs = {"name": ["pr2"], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
        attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
        attr_types = {"name": str, "pose": Vector3d, "_type": str, "geom": PR2, "backHeight": Value, "lArmPose": Vector7d, "rArmPose": Vector7d, "lGripper": Value, "rGripper": Value}
        robot = parameter.Object(attrs, attr_types)
        # Set the initial arm pose so that pose is not close to joint limit
        robot.lArmPose = np.array([[np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2]]).T
        robot.rArmPose = np.array([[-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2]]).T

        attrs = {"name": ["can"], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": BlueCan, "pose": Vector3d, "rotation": Vector3d, "_type": str}
        can = parameter.Object(attrs, attr_types)
        can.pose = np.array([[5.77887566e-01,  -1.26743678e-01,   8.37601627e-01]]).T
        can.rotation = np.array([[np.pi/4, np.pi/4, np.pi/4]]).T
        # Create openRAVEBody for each parameter
        can_body = OpenRAVEBody(env, can.name, can.geom)
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        # Set the poses and dof values for each body
        can_body.set_pose(can.pose, can.rotation)
        robot_body.set_pose(robot.pose)
        dof_value_map = {"backHeight": robot.backHeight,
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": robot.rArmPose.flatten(),
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)
        # Solve the IK solution
        ik_arm = robot_body.get_ik_arm_pose(can.pose, can.rotation)[0]

        dof_value_map = {"backHeight": ik_arm[:1],
                         "lArmPose": robot.lArmPose.flatten(),
                         "lGripper": robot.lGripper,
                         "rArmPose": ik_arm[1:],
                         "rGripper": robot.rGripper}
        robot_body.set_dof(dof_value_map)
        # robot_body.set_dof(ik_arm[:1], robot.lArmPose, robot.lGripper, ik_arm[1:], robot.rGripper)
        robot_trans = robot_body.env_body.GetLink("r_gripper_tool_frame").GetTransform()
        robot_pos = OpenRAVEBody.obj_pose_from_transform(robot_trans)
        # resulted robot eepose should be exactly the same as that can pose
        self.assertTrue(np.allclose(can.pose.flatten(), robot_pos[:3], atol = 1e-4))
        self.assertTrue(np.allclose(can.rotation.flatten(), robot_pos[3:], atol = 1e-4))
        """
            Uncomment the following to see the robot arm pose
        """
        # robot_body.set_transparency(.7)
        # can_body.set_transparency(.7)
        # env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()

    def test_washer(self):
        from core.util_classes.param_setup import ParamSetup
        washer = ParamSetup.setup_washer()
        env = ParamSetup.setup_env()
        washer_body = OpenRAVEBody(env, washer.name, washer.geom)
        # env.SetViewer("qtcoin")
        # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
