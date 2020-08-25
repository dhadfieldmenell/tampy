from IPython import embed as shell
import unittest
from pma import hl_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.internal_repr.plan import Plan
import numpy as np
import main

class TestHLSolver(unittest.TestCase):
    def setUp(self):
        domain_fname, problem_fname = '../domains/namo_domain/namo.domain', '../domains/namo_domain/namo_probs/namo_1234_1.prob'
        self.d_c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.d_c)

        self.p_c = main.parse_file_to_dict(problem_fname)
        self.hls = hl_solver.FFSolver(self.d_c)


    # Not Needed Anymore
    # TODO: remove HL_State tracking
    # def test_hl_state(self):
    #     problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
    #     plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
    #     preds = problem.init_state.preds
    #     hl_state = hl_solver.HLState(preds)
    #     for pred in preds:
    #         self.assertTrue(pred in hl_state._pred_dict.values())
    #     moveto_preds = plan.actions[0].preds

    #     # testing preconditions, they shouldn't effect the HLState
    #     pre_robotat_init = moveto_preds[0]
    #     import pdb; pdb.set_trace()
    #     self.assertTrue(hl_state.get_rep(pre_robotat_init['pred'])
    #                     in hl_state._pred_dict)
    #     hl_state.add_pred_from_dict(pre_robotat_init)
    #     self.assertTrue(pre_robotat_init['pred'] not in hl_state.get_preds())

    #     pre_obstructs_can1 = moveto_preds[1]
    #     hl_state.add_pred_from_dict(pre_obstructs_can1)
    #     self.assertTrue(pre_obstructs_can1['pred'] not in hl_state.get_preds())

    #     # testing effects, these will effect the HLState
    #     # since the robot isn't at robot_init_pose anymore, this pred should be
    #     # removed from the state
    #     post_robotat_init = moveto_preds[3]
    #     self.assertTrue(hl_state.get_rep(post_robotat_init['pred'])
    #                     in hl_state._pred_dict)
    #     hl_state.add_pred_from_dict(post_robotat_init)
    #     self.assertTrue(hl_state.get_rep(post_robotat_init['pred'])
    #                     not in hl_state._pred_dict)

    #     # since the robot is now at gp_can1, this pred should be added to the
    #     # state
    #     post_robotat_gpcan1 = moveto_preds[4]
    #     self.assertTrue(hl_state.get_rep(post_robotat_gpcan1['pred'])
    #                     not in hl_state._pred_dict)
    #     hl_state.add_pred_from_dict(post_robotat_gpcan1)
    #     self.assertTrue(hl_state.get_rep(post_robotat_gpcan1['pred'])
    #                     in hl_state._pred_dict)

    def test_preds_creation_in_spawn_action(self):
        p_c = self.p_c.copy()
        p_c['Goal'] = '(InGripper pr2 can0 grasp0), (RobotAt pr2 robot_init_pose)'
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

        moveto = plan.actions[1]
        moveto_pred_rep_list = extract_pred_reps_from_pred_dicts(moveto.preds)

        self.assertTrue('(InContact pr2 pdp_target1 target1)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(InContact pr2 pdp_target1 target1)', "hl_state"))
        self.assertTrue('(RobotAt pr2 robot_init_pose)' in moveto_pred_rep_list)
        # self.assertTrue(test_hl_info(moveto.preds, '(RobotAt pr2 robot_init_pose)', "pre")) original test
        self.assertTrue(test_hl_info(moveto.preds, '(RobotAt pr2 robot_init_pose)', "eff"))
        self.assertTrue('(RobotAt pr2 pdp_target0)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(RobotAt pr2 pdp_target0)', "eff"))

        grasp = plan.actions[0]
        grasp_pred_rep_list = extract_pred_reps_from_pred_dicts(grasp.preds)
        # self.assertTrue('(RobotAt pr2 robot_init_pose)' not in grasp_pred_rep_list)
        self.assertTrue('(InContact pr2 pdp_target0 target0)' in grasp_pred_rep_list)
        self.assertTrue(test_hl_info(grasp.preds, '(InContact pr2 pdp_target0 target0)', "pre"))
        self.assertTrue('(InContact pr2 pdp_target1 target1)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(InContact pr2 pdp_target1 target1)', "hl_state"))
        self.assertTrue('(InGripper pr2 can0 grasp0)' in grasp_pred_rep_list)
        self.assertTrue(test_hl_info(grasp.preds, '(InGripper pr2 can0 grasp0)', "eff"))

        moveto2 = plan.actions[1]
        moveto2_pred_rep_list = extract_pred_reps_from_pred_dicts(moveto2.preds)
        self.assertTrue('(RobotAt pr2 pdp_target0)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(RobotAt pr2 pdp_target0)', "pre"))
        self.assertTrue('(RobotAt pr2 robot_init_pose)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(RobotAt pr2 robot_init_pose)', "eff"))
        self.assertTrue('(InGripper pr2 can0 grasp0)' in moveto2_pred_rep_list)
        self.assertTrue(test_hl_info(moveto2.preds, '(InGripper pr2 can0 grasp0)', "pre"))
        self.assertTrue('(InContact pr2 pdp_target0 target0)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(InContact pr2 pdp_target0 target0)', "hl_state"))
        self.assertTrue('(InContact pr2 pdp_target1 target1)' in moveto_pred_rep_list)
        self.assertTrue(test_hl_info(moveto.preds, '(InContact pr2 pdp_target1 target1)', "hl_state"))

    def test_basic(self):
        p_c = self.p_c.copy()
        p_c['Init'] += ', (Obstructs pr2 robot_init_pose pdp_target0 can1)'
        problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, self.domain)
        abs_prob = self.hls.translate_problem(problem)
        plan = self.hls.solve(abs_prob, self.domain, problem)
        # test plan itself
        self.assertEqual(plan.horizon, 115)
        # self.assertEqual(repr(plan.actions), '[0: moveto (0, 19) pr2 robot_init_pose pdp_target1, 1: grasp (20, 21) pr2 can1 target1 pdp_target1 grasp0, 2: movetoholding (22, 41) pr2 pdp_target1 pdp_target2 can1 grasp0, 3: putdown (42, 43) pr2 can1 target2 pdp_target2 grasp0, 4: moveto (44, 63) pr2 pdp_target2 pdp_target0, 5: grasp (64, 65) pr2 can0 target0 pdp_target0 grasp0, 6: movetoholding (66, 85) pr2 pdp_target0 pdp_target1 can0 grasp0, 7: putdown (86, 87) pr2 can0 target1 pdp_target1 grasp0, 8: moveto (88, 107) pr2 pdp_target1 pdp_target2, 9: grasp (108, 109) pr2 can1 target2 pdp_target2 grasp0, 10: movetoholding (110, 129) pr2 pdp_target2 pdp_target0 can1 grasp0, 11: putdown (130, 131) pr2 can1 target0 pdp_target0 grasp0]')
        # test plan params
        self.assertEqual(len(plan.params), 13)
        can0 = plan.params["can0"]
        arr = np.zeros((2, plan.horizon))
        arr[:] = np.NaN
        arr[0, 0] = 2
        arr[1, 0] = 3
        self.assertTrue(np.allclose(can0.pose, arr, equal_nan=True))
        self.assertEqual(can0.get_type(), "Can")
        arr = np.empty((2, 1))
        arr[:] = np.NaN
        self.assertTrue(np.allclose(plan.params["pdp_target0"].value, arr, equal_nan=True))
        # test action preds
        a = plan.actions[5]
        # testing for repr(a) is commented out because
        # result of repr(a) changes when the same test is executed multiple times, It's not reliable
        # self.assertEqual(repr(a), "5: grasp (59, 60) pr2 can1 target1 pdp_target1 grasp1")
        obstrs = [x for x in a.preds if "Obstructs" in repr(x["pred"])]
        self.assertEqual([o["negated"] for o in obstrs], [True, True, True, True])
        self.assertEqual([o["active_timesteps"] for o in obstrs], [(95, 114), (95, 114), (114, 114), (114, 114)])
        reprs = [repr(o["pred"]) for o in obstrs]
        # In the following test of reprs, sometimes can0 obstructs pr2, sometimes can1, not reliable for test
        # expected_vals = ['placeholder: (Obstructs pr2 robot_init_pose can0)',
        #                  'placeholder: (Obstructs pr2 pdp_target1 can0)',
        #                  'placeholder: (Obstructs pr2 pdp_target2 can0)',
        #                  'placeholder: (Obstructs pr2 pdp_target0 can0)']
        #
        #
        # self.assertEqual(sorted(reprs), sorted(expected_vals))

    # Note: Will take care of this later, somehow parameter passed in caused error
    # TODO: During execution of hl_solver, exception is raised on 'placeholder: (InGripper pr2 can1 pdp_target1)'
    # def test_nested_forall(self):
    #     d2 = self.d_c.copy()
    #
    #     d2['Action grasp 20'] = '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose ?g - Grasp) \
    #                             (and (At ?can ?target) \
    #                                 (RobotAt ?robot ?gp) \
    #                                 (InContact ?robot ?gp ?target) \
    #                                 (forall (?obj - Can) \
    #                                     (not (InGripper ?robot ?obj ?g))\
    #                                 ) \
    #                                 (forall (?obj - Can) \
    #                                     (not (Obstructs ?robot ?gp ?obj))\
    #                                 )\
    #                             ) \
    #                             (and (not (At ?can ?target)) \
    #                                 (InGripper ?robot ?can ?g) \
    #                                 (forall (?sym - RobotPose) \
    #                                     (forall (?obj - Can) \
    #                                         (forall (?r - Robot) \
    #                                             (not (Obstructs ?r ?sym ?obj))\
    #                                         )\
    #                                     )\
    #                                 )\
    #                             ) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19'
    #     domain = parse_domain_config.ParseDomainConfig.parse(d2)
    #     problem = parse_problem_config.ParseProblemConfig.parse(self.p_c, domain)
    #     plan = self.hls.solve(self.hls.translate_problem(problem), domain, problem)
    #     a = plan.actions[1]
    #     obstrs = filter(lambda x: "Obstructs" in repr(x["pred"]) and x["active_timesteps"] == (38, 38), a.preds)
    #     reprs = [repr(o["pred"]) for o in obstrs]
    #     expected_vals = ['placeholder: (Obstructs pr2 robot_init_pose can0)',
    #                      'placeholder: (Obstructs pr2 pdp_target1 can0)',
    #                      'placeholder: (Obstructs pr2 pdp_target2 can0)',
    #                      'placeholder: (Obstructs pr2 pdp_target0 can0)',
    #                      'placeholder: (Obstructs pr2 robot_init_pose can1)',
    #                      'placeholder: (Obstructs pr2 pdp_target1 can1)',
    #                      'placeholder: (Obstructs pr2 pdp_target2 can1)',
    #                      'placeholder: (Obstructs pr2 pdp_target0 can1)']
    #     # ['placeholder: (Obstructs pr2 pdp_target1 can1)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target0 can1)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target1 can1)',
    #     #                          'placeholder: (Obstructs pr2 robot_init_pose can1)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target2 can1)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target0 can1)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target1 can0)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target0 can0)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target1 can0)',
    #     #                          'placeholder: (Obstructs pr2 robot_init_pose can0)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target2 can0)',
    #     #                          'placeholder: (Obstructs pr2 pdp_target0 can0)']
    #     self.assertEqual(sorted(reprs), sorted(expected_vals))

    def test_obstr(self):
        p2 = self.p_c.copy()
        p2["Init"] += ", (Obstructs pr2 robot_init_pose pdp_target1 can0)"
        p2["Goal"] = "(InGripper pr2 can0 grasp0)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(repr(plan.actions[0:2]), '[0: grasp (0, 19) pr2 can0 target0 robot_init_pose pdp_target0 grasp0]')
        # self.assertEqual(repr(plan.actions[0:2]), '[0: moveto (0, 19) pr2 robot_init_pose pdp_target0, 1: grasp (19, 20) pr2 can0 target0 pdp_target0 grasp0]')

    def test_impossible_obstr(self):
        p2 = self.p_c.copy()
        p2["Init"] += ", (Obstructs pr2 robot_init_pose pdp_target0 can1), (Obstructs pr2 pdp_target1 pdp_target0 can1), (Obstructs pr2 pdp_target2 pdp_target0 can1), (Obstructs pr2 robot_end_pose pdp_target0 can1), (Obstructs pr2 robot_init_pose pdp_target0 can0), (Obstructs pr2 pdp_target1 pdp_target0 can0), (Obstructs pr2 pdp_target2 pdp_target0 can0), (Obstructs pr2 robot_end_pose pdp_target0 can0)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(plan, Plan.IMPOSSIBLE)

    def test_impossible_goal(self):
        p2 = self.p_c.copy()
        p2["Goal"] += ", (At can1 target1)"
        problem = parse_problem_config.ParseProblemConfig.parse(p2, self.domain)
        plan = self.hls.solve(self.hls.translate_problem(problem), self.domain, problem)
        self.assertEqual(plan, Plan.IMPOSSIBLE)


    def test_hl_plan(self):
        domain_fname, problem_fname = '../domains/laundry_domain/laundry_hl.domain', '../domains/laundry_domain/laundry_probs/laundry_hl.prob'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)

        p_c = main.parse_file_to_dict(problem_fname)
        hls = hl_solver.FFSolver(d_c)
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan = hls.solve(hls.translate_problem(problem), domain, problem)
        print("\n\n" + str(plan.plan_str) + "\n\n")
        self.assertFalse(plan == Plan.IMPOSSIBLE)

if __name__ == '__main__':
    unittest.main()
