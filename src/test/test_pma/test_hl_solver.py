from IPython import embed as shell
import unittest
from pma import hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.internal_repr.plan import Plan
import numpy as np

class TestHLSolver(unittest.TestCase):
    def setUp(self):
        self.d_c = {'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19', 'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19', 'Derived Predicates': 'At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Can; IsGP, RobotPose, Can; IsPDP, RobotPose, Target; Obstructs, Robot, RobotPose, Can', 'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer', 'Primitive Predicates': 'geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; pose, Target, Vector2d; value, RobotPose, Vector2d; geom, Robot, GreenCircle; pose, Robot, Vector2d; pose, Workspace, Vector2d; w, Workspace, int; h, Workspace, int; size, Workspace, int; viewer, Workspace, GridWorldViewer', 'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', 'Types': 'Can, Target, RobotPose, Robot, Workspace'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.d_c)
        self.p_c = {'Init': '(geom target0 1), (pose target0 [3,5]), (value pdp_target0 undefined), (geom target1 1), (pose target1 [3,6]), (value pdp_target1 undefined), (geom target2 1), (pose target2 [5,3]), (value pdp_target2 undefined), (geom can0 1), (pose can0 [3,5]), (value gp_can0 undefined), (geom can1 1), (pose can1 [3,6]), (value gp_can1 undefined), (geom pr2 1), (pose pr2 [0,7]), (value robot_init_pose [0,7]), (pose ws [0,0]), (w ws 8), (h ws 9), (size ws 1), (viewer ws test); (At can0 target0), (IsGP gp_can0 can0), (At can1 target1), (IsGP gp_can1 can1), (IsPDP pdp_target0 target0), (IsPDP pdp_target1 target1), (IsPDP pdp_target2 target2), (RobotAt pr2 robot_init_pose)', 'Objects': 'Target (name target0); RobotPose (name pdp_target0); Can (name can0); RobotPose (name gp_can0); Target (name target1); RobotPose (name pdp_target1); Can (name can1); RobotPose (name gp_can1); Target (name target2); RobotPose (name pdp_target2); Robot (name pr2); RobotPose (name robot_init_pose); Workspace (name ws)', 'Goal': '(At can0 target1)'}
        self.hls = hl_solver.FFSolver(self.d_c)

    def test_basic(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        # test plan itself
        self.assertEqual(plan.horizon, 160)
        self.assertEqual(repr(plan.actions), '[0: moveto (0, 19) pr2 robot_init_pose gp_can1, 1: grasp (20, 39) pr2 can1 target1 gp_can1, 2: moveto (40, 59) pr2 gp_can1 pdp_target2, 3: putdown (60, 79) pr2 can1 target2 pdp_target2, 4: moveto (80, 99) pr2 pdp_target2 gp_can0, 5: grasp (100, 119) pr2 can0 target0 gp_can0, 6: moveto (120, 139) pr2 gp_can0 pdp_target1, 7: putdown (140, 159) pr2 can0 target1 pdp_target1]')
        # test plan params
        self.assertEqual(len(plan.params), 13)
        can0 = plan.params["can0"]
        arr = np.zeros((2, plan.horizon))
        arr[0, 0] = 3
        arr[1, 0] = 5
        self.assertTrue(np.array_equal(can0.pose, arr))
        self.assertEqual(can0.get_type(), "Can")
        self.assertEqual(plan.params["gp_can0"].value, "undefined")
        # test action preds
        a = plan.actions[5]
        self.assertEqual(repr(a), "5: grasp (100, 119) pr2 can0 target0 gp_can0")
        obstrs = filter(lambda x: "Obstructs" in repr(x["pred"]), a.preds)
        self.assertEqual([o["negated"] for o in obstrs], [True, True, True, True, True, True, True, True])
        self.assertEqual([o["active_timesteps"] for o in obstrs], [(100, 119), (100, 119), (119, 119),
                                                                   (119, 119), (119, 119), (119, 119),
                                                                   (119, 119), (119, 119)])
        reprs = [repr(o["pred"]) for o in obstrs]
        self.assertEqual(reprs, ['placeholder: (Obstructs pr2 gp_can0 can1)',
                                 'placeholder: (Obstructs pr2 gp_can0 can0)',
                                 'placeholder: (Obstructs pr2 gp_can1 can0)',
                                 'placeholder: (Obstructs pr2 gp_can0 can0)',
                                 'placeholder: (Obstructs pr2 pdp_target1 can0)',
                                 'placeholder: (Obstructs pr2 robot_init_pose can0)',
                                 'placeholder: (Obstructs pr2 pdp_target2 can0)',
                                 'placeholder: (Obstructs pr2 pdp_target0 can0)'])

    def test_nested_forall(self):
        d2 = self.d_c.copy()
        d2['Action grasp 20'] = '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (forall (?obj - Can) (forall (?r - Robot) (not (Obstructs ?r ?sym ?obj)))))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19'
        domain = parse_domain_config.ParseDomainConfig.parse(d2)
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), domain, problem)
        a = plan.actions[1]
        obstrs = filter(lambda x: "Obstructs" in repr(x["pred"]) and x["active_timesteps"] == (39, 39), a.preds)
        reprs = [repr(o["pred"]) for o in obstrs]
        self.assertEqual(reprs, ['placeholder: (Obstructs pr2 gp_can1 can1)',
                                 'placeholder: (Obstructs pr2 gp_can0 can1)',
                                 'placeholder: (Obstructs pr2 pdp_target1 can1)',
                                 'placeholder: (Obstructs pr2 robot_init_pose can1)',
                                 'placeholder: (Obstructs pr2 pdp_target2 can1)',
                                 'placeholder: (Obstructs pr2 pdp_target0 can1)',
                                 'placeholder: (Obstructs pr2 gp_can1 can0)',
                                 'placeholder: (Obstructs pr2 gp_can0 can0)',
                                 'placeholder: (Obstructs pr2 pdp_target1 can0)',
                                 'placeholder: (Obstructs pr2 robot_init_pose can0)',
                                 'placeholder: (Obstructs pr2 pdp_target2 can0)',
                                 'placeholder: (Obstructs pr2 pdp_target0 can0)'])

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
