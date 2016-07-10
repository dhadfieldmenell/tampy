import unittest
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.util_classes import matrix
from errors_exceptions import ProblemConfigException, ParamValidationException

class TestParseProblemConfig(unittest.TestCase):
    def setUp(self):
        d_c = {
            'Action moveto 20':
                '(?robot - Robot ?start - RobotPose ?end - RobotPose) \
                (and \
                    (RobotAt ?robot ?start) \
                    (forall (?obj - Can) \
                    (not (Obstructs ?robot ?start ?obj))\
                    )\
                ) \
                (and \
                    (not (RobotAt ?robot ?start)) \
                    (RobotAt ?robot ?end)\
                ) 0:0 0:19 19:19 19:19',
            'Action putdown 20':
                '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) \
                (and \
                    (RobotAt ?robot ?pdp) \
                    (IsPDP ?robot ?pdp ?target) \
                    (InGripper ?can) \
                    (forall (?obj - Can) \
                        (not (At ?obj ?target))\
                    ) \
                    (forall (?obj - Can) \
                        (not (Obstructs ?robot ?pdp ?obj))\
                    )\
                ) \
                (and \
                    (At ?can ?target) \
                    (not (InGripper ?can))\
                ) 0:0 0:0 0:0 0:0 0:19 19:19 19:19',
            'Derived Predicates':
                'At, Can, Target; \
                RobotAt, Robot, RobotPose; \
                InGripper, Can; \
                IsGP, Robot, RobotPose, Can; \
                IsPDP, Robot, RobotPose, Target; \
                Obstructs, Robot, RobotPose, Can',
            'Attribute Import Paths':
                'RedCircle core.util_classes.circle, \
                BlueCircle core.util_classes.circle, \
                GreenCircle core.util_classes.circle, \
                Vector2d core.util_classes.matrix, \
                GridWorldViewer core.util_classes.viewer',
            'Predicates Import Path':
                'core.util_classes.common_predicates',
            'Primitive Predicates':
                'geom, Can, RedCircle; \
                pose, Can, Vector2d; \
                geom, Target, BlueCircle; \
                pose, Target, Vector2d; \
                value, RobotPose, Vector2d; \
                geom, Robot, GreenCircle; \
                pose, Robot, Vector2d; \
                pose, Workspace, Vector2d; \
                w, Workspace, int; \
                h, Workspace, int; \
                size, Workspace, int; \
                viewer, Workspace, GridWorldViewer',
            'Action grasp 20':
                '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) \
                (and \
                    (At ?can ?target) \
                    (RobotAt ?robot ?gp) \
                    (IsGP ?robot ?gp ?can) \
                    (forall (?obj - Can) \
                        (not (InGripper ?obj))\
                    ) \
                    (forall (?obj - Can) \
                        (not (Obstructs ?robot ?gp ?obj))\
                    )\
                ) \
                (and \
                    (not (At ?can ?target)) \
                    (InGripper ?can) \
                    (forall (?sym - RobotPose) \
                        (not (Obstructs ?robot ?sym ?can))\
                    )\
                ) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19',
            'Types':
                'Can, Target, RobotPose, Robot, Workspace'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.p_c = {
            'Objects':
                'Robot (name pr2); \
                Target (name target0); \
                Target (name target1); \
                Can (name can0); \
                RobotPose (name gp_can0)',
            'Init': '(geom target0 1), \
                (pose target0 [3, 5]), \
                (geom target1 1), \
                (pose pr2 [1, 2]), \
                (geom pr2 1), \
                (pose target1 [4,6]), \
                (geom can0 1), \
                (pose can0 [3, 5]), \
                (value gp_can0 undefined); \
                (At can0 target0), \
                (IsGP pr2 gp_can0 can0)',
            'Goal': '(At can0 target1)'}

    def test_init_state(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        self.assertEqual(len(problem.init_state.params), 5)
        self.assertEqual(len(problem.init_state.preds), 2)
        self.assertEqual(sum(1 for k, p in problem.init_state.params.items() if p.get_type() == "Can"), 1)
        self.assertEqual(sum(1 for k, p in problem.init_state.params.items() if p.get_type() == "Target"), 2)
        self.assertEqual(sum(1 for k, p in problem.init_state.params.items() if not p.is_symbol()), 4)
        self.assertEqual(sum(1 for k, p in problem.init_state.params.items() if p.name.startswith("gp")), 1)
        for k, p in problem.init_state.params.items():
            if p.is_symbol():
                break
        self.assertEqual(p.name, "gp_can0")
        self.assertTrue(p.is_symbol())
        self.assertFalse(p.is_defined())

    def test_goal_test(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        self.assertFalse(problem.goal_test())
        for k, p in problem.init_state.params.items():
            if k == "target1":
                break
        p.pose = matrix.Vector2d((3, 6))
        self.assertFalse(problem.goal_test())
        p.pose = matrix.Vector2d((3, 5))
        self.assertTrue(problem.goal_test())

    def test_missing_object(self):
        p2 = self.p_c.copy()
        p2["Objects"] = "Robot (name pr2); Target (name target0); Target (name target1); RobotPose (name gp_can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "'can0' is not an object in problem file.")
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
        p2["Init"] = ";(At can0 target0), (IsGP pr2 gp_can0 can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file has no primitive predicates for object 'target1'.")


    def test_missing_derived_preds(self):
        # should work fine even with no derived predicates
        p2 = self.p_c.copy()
        p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (pose target0 [3, 5]), (geom target1 1), (pose target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value gp_can0 undefined);"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(problem.init_state.preds, set())

    def test_failures(self):
        p2 = self.p_c.copy()
        p2["Objects"] += "; Workspace (name ws. pose (0, 0). w 8. h 9. size 1)"
        p2["Init"] = ""
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file needs init.")
        del p2["Init"]
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Problem file needs init.")
        p2 = self.p_c.copy()
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
        p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (pose target0 [3, 5]), (geom target1 1), (pose target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value gp_can0 undefined); (At target0 can0), (IsGP pr2 gp_can0 can0)"
        with self.assertRaises(ParamValidationException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter type validation failed for predicate 'initpred0: (At target0 can0)'.")

        p2 = self.p_c.copy()
        p2["Init"] = "(pose pr2 [1, 2]), (geom pr2 1), (geom target0 1), (pose target0 [3, 5]), (geom target1 1), (pose target1 [4,6]), (geom can0 1), (pose can0 [3, 5]), (value gp_can0 undefined); (At can0 target2), (IsGP pr2 gp_can0 can0)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter 'target2' for predicate type 'At' not defined in domain file.")

        p2 = self.p_c.copy()
        p2["Goal"] = "(At can0 target3)"
        with self.assertRaises(ProblemConfigException) as cm:
            problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        self.assertEqual(cm.exception.message, "Parameter 'target3' for predicate type 'At' not defined in domain file.")

if __name__ == "__main__":
	unittest.main()
