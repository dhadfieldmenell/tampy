import unittest
from core.util_classes import robots
from openravepy import Environment
import numpy as np

class TestRobots(unittest.TestCase):

    def test_pr2(self):
        pr2_robot = robots.PR2()
        env = Environment()
        # env.SetViewer("qtcoin")
        robot = env.ReadRobotXMLFile(pr2_robot.shape)
        env.Add(robot)
        dof_val = robot.GetActiveDOFValues()
        init_dof = np.zeros((39,))
        self.assertTrue(np.allclose(dof_val, init_dof, 1e-6))
        # import ipdb;ipdb.set_trace()
