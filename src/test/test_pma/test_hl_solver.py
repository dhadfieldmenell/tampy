from IPython import embed as shell
import unittest
from pma import hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.internal_repr.plan import Plan
import numpy as np

class TestHLSolver(unittest.TestCase):
    def setUp(self):
        self.d_c = {
            'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) \
                (and \
                    (RobotAt ?robot ?start) \
                    (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj))) \
                ) \
                (and \
                    (not (RobotAt ?robot ?start)) \
                    (RobotAt ?robot ?end) \
                ) \
                0:0 0:19 19:19 19:19',
            'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) \
                (and \
                    (RobotAt ?robot ?pdp) \
                    (IsPDP ?robot ?pdp ?can ?target) \
                    (InGripper ?can) \
                    (forall (?obj - Can) (not (At ?obj ?target))) \
                    (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))\
                ) \
                (and \
                    (At ?can ?target) \
                    (not (InGripper ?can)) \
                ) \
                0:0 0:0 0:0 0:0 0:19 19:19 19:19',
            'Derived Predicates': \
                'At, Can, Target; \
                RobotAt, Robot, RobotPose; \
                InGripper, Can; \
                IsGP, Robot, RobotPose, Can; \
                IsPDP, Robot, RobotPose, Can, Target; \
                Obstructs, Robot, RobotPose, Can', \
            'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer',
            'Primitive Predicates': \
                'geom, Can, RedCircle; pose, Can, Vector2d; \
                geom, Target, BlueCircle; pose, Target, Vector2d; \
                value, RobotPose, Vector2d; \
                geom, Robot, GreenCircle; pose, Robot, Vector2d; \
                pose, Workspace, Vector2d; w, Workspace, int; h, Workspace, int; size, Workspace, int; viewer, Workspace, GridWorldViewer', \
            'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) \
                (and \
                    (At ?can ?target) \
                    (RobotAt ?robot ?gp) \
                    (IsGP ?robot ?gp ?can) \
                    (forall (?obj - Can) (not (InGripper ?obj))) \
                    (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj))) \
                ) \
                (and \
                    (not (At ?can ?target)) \
                    (InGripper ?can) \
                    (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?can)))\
                ) \
                0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', \
            'Types': 'Can, Target, RobotPose, Robot, Workspace'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.d_c)
        self.p_c = {
            'Init': \
                '(geom target0 1), (pose target0 [3,5]), \
                (value pdp_target0 undefined), \
                (geom target1 1), (pose target1 [3,6]), \
                (value pdp_target1 undefined), \
                (geom target2 1), (pose target2 [5,3]), \
                (value pdp_target2 undefined), \
                (geom can0 1), (pose can0 [3,5]), \
                (value gp_can0 undefined), \
                (geom can1 1), (pose can1 [3,6]), \
                (value gp_can1 undefined), \
                (geom pr2 1), (pose pr2 [0,7]), \
                (value robot_init_pose [0,7]), \
                (pose ws [0,0]), (w ws 8), (h ws 9), (size ws 1), (viewer ws); \
                (At can0 target0), \
                (IsGP pr2 gp_can0 can0), \
                (At can1 target1), \
                (IsGP pr2 gp_can1 can1), \
                (IsPDP pr2 pdp_target0 can0 target0), \
                (IsPDP pr2 pdp_target1 can1 target1), \
                (IsPDP pr2 pdp_target2 can2 target2), \
                (RobotAt pr2 robot_init_pose)',
            'Objects': \
                'Target (name target0); \
                RobotPose (name pdp_target0); \
                Can (name can0); \
                RobotPose (name gp_can0); \
                Target (name target1); \
                RobotPose (name pdp_target1); \
                Can (name can1); \
                RobotPose (name gp_can1); \
                Target (name target2); \
                RobotPose (name pdp_target2); \
                Robot (name pr2); \
                RobotPose (name robot_init_pose); \
                Workspace (name ws)', \
            'Goal': '(At can0 target1)'}

        self.hls = hl_solver.FFSolver(self.d_c)

    def test_hl_state(self):
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        preds = problem.init_state.preds
        hl_state = hl_solver.HLState(preds)
        for pred in preds:
            self.assertTrue(pred in hl_state._pred_dict.values())
        moveto_preds = plan.actions[0].preds

        # testing preconditions, they shouldn't effect the HLState
        pre_robotat_init = moveto_preds[0]
        self.assertTrue(hl_state.get_rep(pre_robotat_init['pred'])
                        in hl_state._pred_dict)
        hl_state.add_pred_from_dict(pre_robotat_init)
        self.assertTrue(pre_robotat_init['pred'] not in hl_state.get_preds())

        pre_obstructs_can1 = moveto_preds[1]
        hl_state.add_pred_from_dict(pre_obstructs_can1)
        self.assertTrue(pre_obstructs_can1['pred'] not in hl_state.get_preds())

        # testing effects, these will effect the HLState
        # since the robot isn't at robot_init_pose anymore, this pred should be
        # removed from the state
        post_robotat_init = moveto_preds[3]
        self.assertTrue(hl_state.get_rep(post_robotat_init['pred'])
                        in hl_state._pred_dict)
        hl_state.add_pred_from_dict(post_robotat_init)
        self.assertTrue(hl_state.get_rep(post_robotat_init['pred'])
                        not in hl_state._pred_dict)

        # since the robot is now at gp_can1, this pred should be added to the
        # state
        post_robotat_gpcan1 = moveto_preds[4]
        self.assertTrue(hl_state.get_rep(post_robotat_gpcan1['pred'])
                        not in hl_state._pred_dict)
        hl_state.add_pred_from_dict(post_robotat_gpcan1)
        self.assertTrue(hl_state.get_rep(post_robotat_gpcan1['pred'])
                        in hl_state._pred_dict)

    def test_preds_creation_in_spawn_action(self):
        p_c = self.p_c.copy()
        p_c['Goal'] = '(InGripper can0), (RobotAt pr2 robot_init_pose)'
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        HLState = hl_solver.HLState
        init_pred_rep_list = [HLState.get_rep(pred) for pred in problem.init_state.preds]

        def extract_pred_reps_from_pred_dicts(pred_dicts):
            pred_rep_list = []
            for pred_dict in pred_dicts:
                pred = pred_dict['pred']
                pred_rep_list.append(HLState.get_rep(pred))
            return pred_rep_list

        def test_hl_info(pred_dicts, pred_rep, hl_info):
            for pred_dict in pred_dicts:
                pred = pred_dict['pred']
                if HLState.get_rep(pred) == pred_rep:
                    if pred_dict["hl_info"] == hl_info:
                        return True
            return False

        moveto = plan.actions[0]
        moveto_pred_rep_list = extract_pred_reps_from_pred_dicts(moveto.preds)
        self.assertTrue('(IsGP gp_can1 can1)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(IsGP gp_can1 can1)', "hl_state"))
        self.assertTrue('(RobotAt pr2 robot_init_pose)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(RobotAt pr2 robot_init_pose)', "pre"))
        self.assertTrue('(RobotAt pr2 gp_can0)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(RobotAt pr2 gp_can0)', "eff"))

        grasp = plan.actions[1]
        grasp_pred_rep_list = extract_pred_reps_from_pred_dicts(grasp.preds)
        self.assertTrue('(RobotAt pr2 robot_init_pose)' not in grasp_pred_rep_list)
        self.assertTrue('(IsGP gp_can0 can0)' in grasp_pred_rep_list)
        self.assertTrue(test_hl_info(grasp.preds, '(IsGP gp_can0 can0)', "pre"))
        self.assertTrue('(IsGP gp_can1 can1)' in grasp_pred_rep_list)
        self.assertTrue(test_hl_info(grasp.preds, '(IsGP gp_can1 can1)', "hl_state"))
        self.assertTrue('(InGripper can0)' in grasp_pred_rep_list)
        self.assertTrue(test_hl_info(grasp.preds, '(InGripper can0)', "eff"))

        moveto2 = plan.actions[2]
        moveto2_pred_rep_list = extract_pred_reps_from_pred_dicts(moveto2.preds)
        self.assertTrue('(RobotAt pr2 gp_can0)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(RobotAt pr2 gp_can0)', "pre"))
        self.assertTrue('(RobotAt pr2 robot_init_pose)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(RobotAt pr2 robot_init_pose)', "eff"))
        self.assertTrue('(InGripper can0)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(InGripper can0)', "hl_state"))
        self.assertTrue('(IsGP gp_can0 can0)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(IsGP gp_can0 can0)', "hl_state"))
        self.assertTrue('(IsGP gp_can1 can1)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(IsGP gp_can1 can1)', "hl_state"))

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
        arr[:] = np.NaN
        arr[0, 0] = 3
        arr[1, 0] = 5
        self.assertTrue(np.allclose(can0.pose, arr, equal_nan=True))
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
        d2['Action grasp 20'] = '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?robot ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (forall (?obj - Can) (forall (?r - Robot) (not (Obstructs ?r ?sym ?obj)))))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19'
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
