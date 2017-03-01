import unittest
from pma import pr_graph, hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
import main

class TestPRGraph(unittest.TestCase):

    def test_goal_test(self):

        domain_fname = '../domains/namo_domain/namo.domain'
        problem_fname = '../domains/namo_domain/namo_probs/putaway2.prob'
        d_c = main.parse_file_to_dict(domain_fname)
        p_c = main.parse_file_to_dict(problem_fname)
        s_c = {'LLSolver': 'NAMOSolver', 'HLSolver': 'FFSolver'}
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        def get_plan_str(p_fname, plan_str=None):
            if plan_str is not None:
                return plan_str
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            return hls._run_planner(hls.abs_domain, abs_problem)

        plan_str = get_plan_str('../domains/namo_domain/namo_probs/putaway2.prob')
        print plan_str
        plan, msg = pr_graph.p_mod_abs(d_c, p_c, s_c)
        # self.assertFalse(plan)
        # self.assertEqual(msg, "Goal is already satisfied. No planning done.")


if __name__ == '__main__':
    unittest.main()
