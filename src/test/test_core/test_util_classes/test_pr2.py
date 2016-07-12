import unittest
from core.util_classes import pr2
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from errors_exceptions import ProblemConfigException, ParamValidationException
import main
from core.util_classes import viewer
from openravepy import Environment
import numpy as np

class TestPR2(unittest.TestCase):

    def test_basic(self):
        pr2_robot = pr2.PR2('../models/pr2/pr2.zae')
        env = Environment()
        robot = env.ReadRobotXMLFile(pr2_robot.geom)
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

    def test_pr2_in_viewer(self):
        domain_fname, problem_fname = '../domains/can_domain/pr2.init', '../domains/can_domain/pr2.prob'
        d_c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.p_c = main.parse_file_to_dict(problem_fname)
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        view = viewer.OpenRAVEViewer()
        view.draw([problem.init_state.params['dude']], 0)
        # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
