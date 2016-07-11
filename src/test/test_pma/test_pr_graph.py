import unittest
from pma import pr_graph
import main

class TestPRGraph(unittest.TestCase):

    ## TODO

    def test_goal_test(self):
        pass
        # domain_fname, problem_fname = '../domains/namo_domain/namo.domain', '../domains/namo_domain/namo_probs/namo_1234_1.prob'

        # d_c = main.parse_file_to_dict(domain_fname)
        # p_c = main.parse_file_to_dict(problem_fname)

        # s_c = {'LLSolver': 'NAMOSolver', 'HLSolver': 'FFSolver'}
        # plan, msg = pr_graph.p_mod_abs(d_c, p_c, s_c)
        # self.assertFalse(plan)
        # self.assertEqual(msg, "Goal is already satisfied. No planning done.")

if __name__ == '__main__':
    unittest.main()
