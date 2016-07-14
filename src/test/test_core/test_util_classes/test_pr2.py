import unittest
from core.util_classes import pr2
from errors_exceptions import ProblemConfigException, ParamValidationException
from openravepy import Environment
import numpy as np

class TestPR2(unittest.TestCase):

    def test_basic(self):
        pr2_robot = pr2.PR2('../models/pr2/pr2.zae')
        env = Environment()
        robot = env.ReadRobotXMLFile(pr2_robot.shape)
        env.Add(robot)
        dof_val = robot.GetActiveDOFValues()
        init_dof = np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   2.77555756e-17,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                            0.00000000e+00,   0.00000000e+00,   2.77555756e-17,
                            0.00000000e+00,   0.00000000e+00,   0.00000000e+00])

        self.assertTrue(np.allclose(dof_val, init_dof, 1e-6))



if __name__ == "__main__":
    unittest.main()
