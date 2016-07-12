import unittest
from core.util_classes import pr2
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from errors_exceptions import ProblemConfigException, ParamValidationException
import main
from core.util_classes import viewer

class TestPR2(unittest.TestCase):

    def test_basic(self):
        r = pr2.PR2(5)
        self.assertEqual(r.geom, 5)

    def test_pr2_in_viewer(self):
        domain_fname, problem_fname = '../domains/can_domain/pr2.init', '../domains/can_domain/pr2.prob'
        d_c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.p_c = main.parse_file_to_dict(problem_fname)
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        # view = viewer.OpenRAVEViewer()
        # view.draw([problem.init_state.params['dude']], 0)
        # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
