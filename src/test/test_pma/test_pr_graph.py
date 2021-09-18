import unittest
from pma import pr_graph, hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
import main

class TestPRGraph(unittest.TestCase):
    def test_putaway2(self):
        prob_file = '../domains/namo_domain/namo_probs/putaway2.prob'
        helper_test_prg(self, prob_file)
    def test_move_grasp(self):
        prob_file = '../domains/namo_domain/namo_probs/move_grasp.prob'
        helper_test_prg(self, prob_file)
    def test_move_no_obs(self):
        prob_file = '../domains/namo_domain/namo_probs/move_no_obs.prob'
        helper_test_prg(self, prob_file)
    def test_move_w_obs(self):
        prob_file = '../domains/namo_domain/namo_probs/move_w_obs.prob'
        helper_test_prg(self, prob_file)
    def test_moveholding(self):
        prob_file = '../domains/namo_domain/namo_probs/moveholding.prob'
        helper_test_prg(self, prob_file)
    def test_place(self):
        prob_file = '../domains/namo_domain/namo_probs/place.prob'
        helper_test_prg(self, prob_file)

def helper_test_prg(self, prob_file):

    domain_fname = '../domains/namo_domain/namo.domain'
    problem_fname = prob_file
    d_c = main.parse_file_to_dict(domain_fname)
    p_c = main.parse_file_to_dict(problem_fname)
    s_c = {'LLSolver': 'NAMOSolver', 'HLSolver': 'FFSolver'}
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    # """
    # Initialize Suggester
    # """
    # import ipdb; ipdb.set_trace()
    # suggester = PostLearner({}, "prg_testing", space = "CONFIG")
    # if not suggester.trained:
    #     feature_function = None
    #     suggester.train(domain, problem, feature_function)
    # """
    # End of Suggester
    # """
    # TODO fix the inconsistency when running prg graph
    # plan, msg = pr_graph.p_mod_abs(d_c, p_c, s_c,suggester = None, debug=True)
    # self.assertEqual(len(plan.get_failed_preds()), 0)

