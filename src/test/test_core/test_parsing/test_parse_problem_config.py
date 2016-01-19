import unittest
from core.parsing import parse_problem_config

class TestParseProblemConfig(unittest.TestCase):
    # TODO, error msg bad goal preds
    def setUp(self):
        # can domain config
        self.config = {'LLSolver': 'CanSolver', 'Objects': 'can1 - Can, can2 - Can, can3 - Can, robot_init_pose - Symbol, larm_gp_can1 - Symbol, larm_gp_can2 - Symbol, larm_gp_can3 - Symbol, rarm_gp_can1 - Symbol, rarm_gp_can2 - Symbol, rarm_gp_can3 - Symbol, larm_pdp_can1 - Symbol, larm_pdp_can2 - Symbol, larm_pdp_can3 - Symbol, rarm_pdp_can1 - Symbol, rarm_pdp_can2 - Symbol, rarm_pdp_can3 - Symbol, tableloc - Symbol, pr2 - Robot, lgripper - Manip, rgripper - Manip', 'Viewer': 'OpenRAVEViewer', 'Action grasp': '(?robot - Robot ?can - Can ?gp - Symbol ?manip - Manip) (and (RobotAt ?robot ?gp) (IsGP ?gp ?can ?manip) (forall (?obj - Can) (not (InGripper ?obj ?manip))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (InGripper ?can ?manip) (forall (?sym - Symbol) (not (Obstructs ?robot ?sym ?can))))', 'Environment Initializer': 'InitCanEnv', 'HLSolver': 'FFSolver', 'Action moveto': '(?robot - Robot ?start - Symbol ?end - Symbol) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end))', 'Init': '(RobotAt pr2 robot_init_pose), (IsGP larm_gp_can1 can1 lgripper), (IsGP larm_gp_can2 can2 lgripper), (IsGP larm_gp_can3 can3 lgripper), (IsGP rarm_gp_can1 can1 rgripper), (IsGP rarm_gp_can2 can2 rgripper), (IsGP rarm_gp_can3 can3 rgripper), (IsPDP larm_pdp_can1 tableloc can1 lgripper), (IsPDP larm_pdp_can2 tableloc can2 lgripper), (IsPDP larm_pdp_can3 tableloc can3 lgripper), (IsPDP rarm_pdp_can1 tableloc can1 rgripper), (IsPDP rarm_pdp_can2 tableloc can2 rgripper), (IsPDP rarm_pdp_can3 tableloc can3 rgripper)', 'Action putdown': '(?robot - Robot ?can - Can ?pdp - Symbol ?targetloc - Symbol ?manip - Manip) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?targetloc ?can ?manip) (InGripper ?can ?manip) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (not (InGripper ?can ?manip)))', 'Predicates': 'RobotAt, Robot, Symbol; InGripper, Can, Manip; IsGP, Symbol, Can, Manip; IsPDP, Symbol, Symbol, Can, Manip; Obstructs, Robot, Symbol, Can', 'Goal': '(InGripper can1 lgripper)', 'Types': 'Can, Symbol, Robot, Manip', 'Environment File': 'can_basic.can'}

    def test_basic(self):
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        self.assertEqual(len(problem.init_state.params), 20)
        self.assertEqual(len(problem.init_state.preds), 13)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.get_type() == "Can"), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.name.startswith("larm_gp")), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.name.startswith("rarm_gp")), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.name.startswith("larm_pdp")), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if p.name.startswith("rarm_pdp")), 3)
        self.assertEqual(sum(1 for p in problem.init_state.params if not p.is_symbol()), 6)
        self.assertFalse(problem.goal_test())

    def test_goal_met(self):
        self.config["Goal"] = "(RobotAt pr2 robot_init_pose)"
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        self.assertTrue(problem.goal_test())
        self.config["Goal"] = "(InGripper can1 lgripper)"
