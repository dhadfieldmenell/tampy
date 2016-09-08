import unittest
from core.util_classes import robots
from openravepy import Environment
import numpy as np

class TestRobots(unittest.TestCase):

    def test_pr2(self):
        pr2_robot = robots.PR2()
        env = Environment()
        robot = env.ReadRobotXMLFile(pr2_robot.shape)
        env.Add(robot)

        """
        To check whether pr2 model works, uncomment the following
        """
        # env.SetViewer('qtosg')
        # import ipdb; ipdb.set_trace()
