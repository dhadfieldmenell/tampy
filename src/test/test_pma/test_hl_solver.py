import unittest
from pma import hl_solver
from core import parse_config_to_problem

class TestHLSolver(unittest.TestCase):
    def setUp(self):
        # can domain config
        self.config = {'LLSolver': 'CanSolver', 'Objects': 'can1 - Can, can2 - Can, can3 - Can, robot_init_pose - Symbol, larm_gp_can1 - Symbol, larm_gp_can2 - Symbol, larm_gp_can3 - Symbol, rarm_gp_can1 - Symbol, rarm_gp_can2 - Symbol, rarm_gp_can3 - Symbol, larm_pdp_can1 - Symbol, larm_pdp_can2 - Symbol, larm_pdp_can3 - Symbol, rarm_pdp_can1 - Symbol, rarm_pdp_can2 - Symbol, rarm_pdp_can3 - Symbol, tableloc - Symbol, pr2 - Robot, lgripper - Manip, rgripper - Manip', 'Viewer': 'OpenRAVEViewer', 'Action grasp': '(?robot - Robot ?can - Can ?gp - Symbol ?manip - Manip) (and (RobotAt ?robot ?gp) (IsGP ?gp ?can ?manip) (forall (?obj - Can) (not (InGripper ?obj ?manip))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (InGripper ?can ?manip) (forall (?sym - Symbol) (not (Obstructs ?robot ?sym ?can))))', 'Environment Initializer': 'InitCanEnv', 'HLSolver': 'FFSolver', 'Action moveto': '(?robot - Robot ?start - Symbol ?end - Symbol) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end))', 'Init': '(RobotAt pr2 robot_init_pose), (IsGP larm_gp_can1 can1 lgripper), (IsGP larm_gp_can2 can2 lgripper), (IsGP larm_gp_can3 can3 lgripper), (IsGP rarm_gp_can1 can1 rgripper), (IsGP rarm_gp_can2 can2 rgripper), (IsGP rarm_gp_can3 can3 rgripper), (IsPDP larm_pdp_can1 tableloc can1 lgripper), (IsPDP larm_pdp_can2 tableloc can2 lgripper), (IsPDP larm_pdp_can3 tableloc can3 lgripper), (IsPDP rarm_pdp_can1 tableloc can1 rgripper), (IsPDP rarm_pdp_can2 tableloc can2 rgripper), (IsPDP rarm_pdp_can3 tableloc can3 rgripper)', 'Action putdown': '(?robot - Robot ?can - Can ?pdp - Symbol ?targetloc - Symbol ?manip - Manip) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?targetloc ?can ?manip) (InGripper ?can ?manip) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (not (InGripper ?can ?manip)))', 'Predicates': 'RobotAt, Robot, Symbol; InGripper, Can, Manip; IsGP, Symbol, Can, Manip; IsPDP, Symbol, Symbol, Can, Manip; Obstructs, Robot, Symbol, Can', 'Goal': '(InGripper can1 lgripper)', 'Types': 'Can, Symbol, Robot, Manip', 'Environment File': 'can_basic.can'}

    def test_basic(self):
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        hls = hl_solver.FFSolver()
        plan = hls.solve(hls.translate(problem, self.config), problem)
        self.assertEqual(plan, ['0: MOVETO PR2 ROBOT_INIT_POSE LARM_GP_CAN1', '1: GRASP PR2 CAN1 LARM_GP_CAN1 LGRIPPER'])

    def test_obstr(self):
        orig = self.config["Init"]
        self.config["Init"] += ", (Obstructs pr2 larm_gp_can1 can2), (Obstructs pr2 rarm_gp_can2 can1)"
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        hls = hl_solver.FFSolver()
        plan = hls.solve(hls.translate(problem, self.config), problem)
        self.assertEqual(plan, ['0: MOVETO PR2 ROBOT_INIT_POSE LARM_GP_CAN2', '1: GRASP PR2 CAN2 LARM_GP_CAN2 LGRIPPER', '2: MOVETO PR2 LARM_GP_CAN2 LARM_PDP_CAN2', '3: PUTDOWN PR2 CAN2 LARM_PDP_CAN2 TABLELOC LGRIPPER', '4: MOVETO PR2 LARM_PDP_CAN2 LARM_GP_CAN1', '5: GRASP PR2 CAN1 LARM_GP_CAN1 LGRIPPER'])

        self.config["Init"] = orig
        self.config["Init"] += ", (Obstructs pr2 larm_gp_can1 can2), (Obstructs pr2 larm_gp_can2 can1)"
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        hls = hl_solver.FFSolver()
        plan = hls.solve(hls.translate(problem, self.config), problem)
        self.assertEqual(plan, ['0: MOVETO PR2 ROBOT_INIT_POSE RARM_GP_CAN2', '1: GRASP PR2 CAN2 RARM_GP_CAN2 RGRIPPER', '2: MOVETO PR2 RARM_GP_CAN2 LARM_GP_CAN1', '3: GRASP PR2 CAN1 LARM_GP_CAN1 LGRIPPER'])
        self.config["Init"] = orig

    def test_impossible(self):
        orig = self.config["Init"]
        self.config["Init"] += ", (Obstructs pr2 larm_gp_can1 can2), (Obstructs pr2 rarm_gp_can1 can2), (Obstructs pr2 larm_gp_can2 can1), (Obstructs pr2 rarm_gp_can2 can1)"
        problem = parse_config_to_problem.ParseConfigToProblem(self.config).parse()
        hls = hl_solver.FFSolver()
        plan = hls.solve(hls.translate(problem, self.config), problem)
        self.assertEqual(plan, "impossible")
