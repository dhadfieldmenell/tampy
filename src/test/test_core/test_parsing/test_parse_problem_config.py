import unittest
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.util_classes import matrix
from errors_exceptions import ProblemConfigException, ParamValidationException

class TestParseProblemConfig(unittest.TestCase):
    def setUp(self):
        d_c = {'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19', 'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19', 'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', 'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer', 'Predicates': 'At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Can; IsGP, RobotPose, Can; IsPDP, RobotPose, Target; Obstructs, Robot, RobotPose, Can', 'Types': 'Can (name str. geom RedCircle. pose Vector2d); Target (name str. geom BlueCircle. pose Vector2d); RobotPose (name str. value Vector2d); Robot (name str. geom GreenCircle. pose Vector2d); Workspace (name str. pose Vector2d. w int. h int. size int)'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.p_c = {'Init': '(At can0 target0), (IsGP gp_can0 can0)', 'Objects': 'Target (name target0. geom 1. pose (3, 5)); Target (name target1. geom 1. pose (4, 6)); Can (name can0. geom 1. pose (3, 5)); RobotPose (name gp_can0. value undefined)', 'Goal': '(At can0 target1)'}

    def test_init_state(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        self.assertEqual(len(problem.init_state.params), 4)
        self.assertEqual(len(problem.init_state.preds), 2)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.get_type() == "Can"), 1)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.get_type() == "Target"), 2)
        self.assertEqual(sum(1 for p in problem.init_state.params if not p.is_symbol()), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.name.startswith("gp")), 1)
        for p in problem.init_state.params:
            if p.is_symbol():
                break
        self.assertEqual(p.name, "gp_can0")
        self.assertTrue(p.is_symbol())
        self.assertFalse(p.is_defined())

    def test_goal_test(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        self.assertFalse(problem.goal_test())
        for p in problem.init_state.params:
            if p.name == "target1":
                break
        p.pose = matrix.Vector2d((3, 6))
        self.assertFalse(problem.goal_test())
        p.pose = matrix.Vector2d((3, 5))
        self.assertTrue(problem.goal_test())

    def test_failures(self):
        p2 = self.p_c.copy()
        p2["Objects"] += "; Workspace (name ws. pose (0, 0). w 8. h 9. size 1)"
        p2["Init"] = ""
        # should work fine even with no initial predicates
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(problem.init_state.preds, set())
        p2["Objects"] += "; Workspace (name ws. pose (0, 0). w 8. h nine. size 1)"
        # type of h is wrong
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Some attribute type in parameter 'ws' is incorrect.")

        p2 = self.p_c.copy()
        p2["Objects"] += "; Test (name testname)"
        with self.assertRaises(AssertionError) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)

        p2 = self.p_c.copy()
        p2["Objects"] += "; Test (name testname. value (3, 5))"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter 'testname' not defined in domain file.")

        p2 = self.p_c.copy()
        p2["Init"] = "(At target0 can0), (IsGP gp_can0 can0)"
        with self.assertRaises(ParamValidationException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'initpred0: (At target0 can0)'.")

        p2 = self.p_c.copy()
        p2["Init"] = "(At can0 target2), (IsGP gp_can0 can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter 'target2' for predicate type 'At' not defined in domain file.")

        p2 = self.p_c.copy()
        p2["Goal"] = "(At can0 target3)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter 'target3' for predicate type 'At' not defined in domain file.")
