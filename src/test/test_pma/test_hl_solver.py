import unittest
from pma import hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.internal_repr.plan import Plan

class TestHLSolver(unittest.TestCase):
    # TODO: add more tests
    def setUp(self):
        d_c = {'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19', 'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19', 'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', 'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer', 'Predicates': 'At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Can; IsGP, RobotPose, Can; IsPDP, RobotPose, Target; Obstructs, Robot, RobotPose, Can', 'Types': 'Can (name str. geom RedCircle. pose Vector2d); Target (name str. geom BlueCircle. pose Vector2d); RobotPose (name str. value Vector2d); Robot (name str. geom GreenCircle. pose Vector2d); Workspace (name str. pose Vector2d. w int. h int. size int. viewer GridWorldViewer)'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.p_c = {'Init': '(At can0 target0), (IsGP gp_can0 can0), (At can1 target1), (IsGP gp_can1 can1), (IsPDP pdp_target0 target0), (IsPDP pdp_target1 target1), (IsPDP pdp_target2 target2), (RobotAt pr2 robot_init_pose)', 'Objects': 'Target (name target0. geom 1. pose (3, 5)); RobotPose (name pdp_target0. value undefined); Can (name can0. geom 1. pose (3, 5)); RobotPose (name gp_can0. value undefined); Target (name target1. geom 1. pose (3, 6)); RobotPose (name pdp_target1. value undefined); Can (name can1. geom 1. pose (3, 6)); RobotPose (name gp_can1. value undefined); Target (name target2. geom 1. pose (5, 3)); RobotPose (name pdp_target2. value undefined); Robot (name pr2. geom 1. pose (0, 7)); RobotPose (name robot_init_pose. value (0, 7)); Workspace (name ws. pose (0, 0). w 8. h 9. size 1. viewer test)', 'Goal': '(At can0 target1)'}
        self.hls = hl_solver.FFSolver(d_c)

    def test_basic(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(repr(plan.actions), '[0: moveto (0, 19) pr2 robot_init_pose gp_can1, 1: grasp (20, 39) pr2 can1 target1 gp_can1, 2: moveto (40, 59) pr2 gp_can1 pdp_target2, 3: putdown (60, 79) pr2 can1 target2 pdp_target2, 4: moveto (80, 99) pr2 pdp_target2 gp_can0, 5: grasp (100, 119) pr2 can0 target0 gp_can0, 6: moveto (120, 139) pr2 gp_can0 pdp_target1, 7: putdown (140, 159) pr2 can0 target1 pdp_target1]')

    def test_obstr(self):
        p2 = self.p_c.copy()
        p2["Init"] += ", (Obstructs pr2 gp_can1 can0)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(repr(plan.actions[0:2]), '[0: moveto (0, 19) pr2 robot_init_pose gp_can0, 1: grasp (20, 39) pr2 can0 target0 gp_can0]')

    def test_impossible_obstr(self):
        p2 = self.p_c.copy()
        p2["Init"] += ", (Obstructs pr2 gp_can0 can1), (Obstructs pr2 gp_can1 can0)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(plan, Plan.IMPOSSIBLE)

    def test_impossible_goal(self):
        p2 = self.p_c.copy()
        p2["Goal"] += ", (At can1 target1)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(plan, Plan.IMPOSSIBLE)
