import unittest
from core.util_classes import pr2
from errors_exceptions import ProblemConfigException, ParamValidationException
from openravepy import Environment
import numpy as np

class TestPR2(unittest.TestCase):

    def test_basic(self):
        pr2_robot = pr2.PR2()
        env = Environment()
        robot = env.ReadRobotXMLFile(pr2_robot.shape)
        env.Add(robot)

        """
        To check whether pr2 model works, uncomment the following
        """
        # env.SetViewer('qtosg')
        # import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    unittest.main()
