import numpy as np
import unittest
from openravepy import Environment
from core.internal_repr import parameter
from core.util_classes import circle
from core.util_classes import matrix
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import OpenRAVEException

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
        attr_types = {"geom": DummyGeom, "pose": matrix.Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)
        with self.assertRaises(OpenRAVEException) as cm:
            green_body = OpenRAVEBody(env, "can0", green_can.geom)
        self.assertTrue("Geometry not supported" in cm.exception.message)


    def test_add_circle(self):
        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.GreenCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
        green_can = parameter.Object(attrs, attr_types)

        attrs = {"geom": [1], "pose": [(3, 5)], "_type": ["Can"], "name": ["can0"]}
        attr_types = {"geom": circle.BlueCircle, "pose": matrix.Vector2d, "_type": str, "name": str}
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
