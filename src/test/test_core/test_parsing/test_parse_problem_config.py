import unittest
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.util_classes import matrix
from errors_exceptions import ProblemConfigException, ParamValidationException
import main

class TestParseProblemConfig(unittest.TestCase):
    def setUp(self):
        domain_fname, problem_fname = '../domains/namo_domain/namo.domain', '../domains/namo_domain/namo_probs/namo_1234_1.prob'

        d_c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)


        self.p_c = main.parse_file_to_dict(problem_fname)

    def test_init_state(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        #The number of params and preds keeps on changing
        self.assertEqual(len(problem.init_state.params), 13)
        self.assertEqual(len(problem.init_state.preds), 17)
        self.assertEqual(sum(1 for k, p in list(problem.init_state.params.items()) if p.get_type() == "Can"), 2)
        self.assertEqual(sum(1 for k, p in list(problem.init_state.params.items()) if p.get_type() == "Target"), 3)
        self.assertEqual(sum(1 for k, p in list(problem.init_state.params.items()) if not p.is_symbol()), 4)
        self.assertEqual(sum(1 for k, p in list(problem.init_state.params.items()) if p.name.startswith("pdp")), 3)

    def test_goal_test(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        ## initially the goal is false
        self.assertFalse(problem.goal_test())
        ## want can0 at target1 and can1 at target0
        ## target1 is at 7, 6; target0 is at 2, 3
        for k, p in list(problem.init_state.params.items()):
            if k == "can0":
                p.pose = matrix.Vector2d((7, 6))
            if k == "can1":
                p.pose = matrix.Vector2d((2, 3))
        self.assertTrue(problem.goal_test())

    def test_missing_object(self):
        p2 = self.p_c.copy()
        p2["Objects"] = "Robot (name pr2); Target (name target0); Target (name target1); RobotPose (name gp_can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "'pdp_target0' is not an object in problem file.")
        p2["Objects"] = ""
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file needs objects.")
        del p2["Objects"]
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file needs objects.")

    def test_missing_prim_preds(self):
        p2 = self.p_c.copy()
        p2["Init"] = ";(At can0 target0), (InContact pr2 gp_can0 can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file has no primitive predicates for object 'grasp0'.")


    def test_missing_derived_preds(self):
        # should work fine even with no derived predicates
        p2 = self.p_c.copy()
        p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (value target0 [3, 5]), (geom target1 1), (value target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value pdp_target0 undefined), (value grasp0 undefined), (value robot_init_pose [1, 2]), (pose can1 [2, 3]), (value target2 [5, 5]), (value pdp_target2 undefined), (value pdp_target1 undefined), (pose obs0 [0, 0]), (geom obs0 closet), (value robot_end_pose [0, 0]);"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(problem.init_state.preds, set())

    # def test_failures(self):
    #     p2 = self.p_c.copy()
    #     p2["Objects"] += "; "
    #     p2["Init"] = ""
    #     with self.assertRaises(ProblemConfigException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Problem file needs init.")
    #     del p2["Init"]
    #     with self.assertRaises(ProblemConfigException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Problem file needs init.")

    #     p2 = self.p_c.copy()
    #     p2["Objects"] += "; Test (name testname)"
    #     with self.assertRaises(AssertionError) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)

    #     p2 = self.p_c.copy()
    #     p2["Objects"] += "; Test (name testname. value (3, 5))"
    #     with self.assertRaises(ProblemConfigException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Parameter 'testname' not defined in domain file.")

    #     p2 = self.p_c.copy()
    #     p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (pose target0 [3, 5]), (geom target1 1), (pose target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value pdp_target0 undefined); (At target0 can0), (InContact pr2 pdp_target0 target0)"
    #     with self.assertRaises(ParamValidationException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'initpred0: (At target0 can0)'.")

    #     p2 = self.p_c.copy()
    #     p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (pose target0 [3, 5]), (geom target1 1), (pose target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value gp_can0 undefined); (At can0 target2), (InContact pr2 gp_can0 target0)"
    #     with self.assertRaises(ProblemConfigException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Parameter 'target2' for predicate type 'At' not defined in domain file.")

    #     p2 = self.p_c.copy()
    #     p2["Goal"] = "(At can0 target3)"
    #     with self.assertRaises(ProblemConfigException) as cm:
    #         problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
    #     self.assertEqual(cm.exception.message, "Parameter 'target3' for predicate type 'At' not defined in domain file.")

if __name__ == "__main__":
    unittest.main()
