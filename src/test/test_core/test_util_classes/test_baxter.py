import unittest
from core.util_classes import box, matrix
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from core.internal_repr import parameter
from openravepy import Environment
import numpy as np

class TestBaxter(unittest.TestCase):

    def test_baxter(self):

        env = Environment()
        attrs = {"name": ['baxter'], "pose": ['undefined'], "_type": ["Robot"], "geom": []}
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": Baxter}
        baxter = parameter.Object(attrs, attr_types)
        baxter.pose = np.zeros((1,1))
        baxter.rArmPose = np.array([[0,-0.785,0.785,1.57,-0.785,-0.785,0]]).T
        baxter_body = OpenRAVEBody(env, baxter.name, baxter.geom)
        baxter_body.set_transparency(0.5)
        body = baxter_body.env_body
        dof = body.GetActiveDOFValues()
        """
        To check whether baxter model works, uncomment the following
        """
        # env.SetViewer('qtosg')
        # import ipdb; ipdb.set_trace()
