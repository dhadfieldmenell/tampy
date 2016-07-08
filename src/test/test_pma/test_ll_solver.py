import unittest
from pma import hl_solver
from pma import ll_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
import gurobipy as grb
import numpy as np
from sco.prob import Prob
from sco.solver import Solver
from sco.variable import Variable
from sco import expr
from core.util_classes.viewer import OpenRAVEViewer
import time

d_c = {'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) \
            (and (RobotAt ?robot ?start)\
                (forall (?obj - Obstacle) (not (Obstructs ?robot ?start ?obj)))\
            ) \
            (and (not (RobotAt ?robot ?start)) \
                (RobotAt ?robot ?end)\
            ) 0:0 0:19 19:19 19:19',
    'Derived Predicates': 'RobotAt, Robot, RobotPose; Obstructs, Robot, RobotPose, Obstacle',
    'Attribute Import Paths': 'GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer, Obstacle core.util_classes.obstacle',
    'Primitive Predicates': 'value, RobotPose, Vector2d; \
        geom, Robot, GreenCircle; pose, Robot, Vector2d; \
        geom, Obstacle, Obstacle; pose, Obstacle, Vector2d; \
        pose, Workspace, Vector2d; w, Workspace, int; h, Workspace, int; size, Workspace, int; viewer, Workspace, GridWorldViewer',
    'Types': 'RobotPose, Robot, Obstacle, Workspace'}

class TestLLSolver(unittest.TestCase):
    def setUp(self):
        """
        self.d_c = {'Action moveto 20': '(?robot - Robot ?start - RobotPose ?end - RobotPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19',
        'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19',
        'Derived Predicates': 'At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Can; IsGP, Robot, RobotPose, Can; IsPDP, RobotPose, Target; Obstructs, Robot, RobotPose, Can',
        'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer',
        'Primitive Predicates': 'geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; pose, Target, Vector2d; value, RobotPose, Vector2d; geom, Robot, GreenCircle; pose, Robot, Vector2d; pose, Workspace, Vector2d; w, Workspace, int; h, Workspace, int; size, Workspace, int; viewer, Workspace, GridWorldViewer',
        'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?robot ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19',
        'Types': 'Can, Target, RobotPose, Robot, Workspace'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.d_c)
        """

        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        p_c = {'Init': '(geom pr2 1), (pose pr2 [0,7]), (value robot_init_pose [0,7]), (value target [0,0]), (pose ws [0,0]), (w ws 8), (h ws 9), (size ws 1), (viewer ws); (RobotAt pr2 robot_init_pose)',
        'Objects': 'RobotPose (name target); Robot (name pr2); RobotPose (name robot_init_pose); RobotPose (name target); Workspace (name ws)',
        'Goal': '(RobotAt pr2 target)'}

        p_c = {'Init': '(geom pr2 1), (pose pr2 [0,7]),\
                (pose obstacle [10,10]),\
                (value robot_init_pose [0,7]),\
                (value target [0,0]),\
                (pose ws [0,0]), (w ws 8), (h ws 9), (size ws 1), (viewer ws);\
                (RobotAt pr2 robot_init_pose)',
                'Objects': 'RobotPose (name target); Robot (name pr2); Obstacle (name obstacle); RobotPose (name robot_init_pose); RobotPose (name target); Workspace (name ws)',
                'Goal': '(RobotAt pr2 target)'}

        hls = hl_solver.FFSolver(d_c)
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        self.move_no_obs = hls.solve(hls.translate_problem(problem), domain, problem)

        p_c = {'Init': '(geom pr2 1), (pose pr2 [-2,0]),\
                (pose obstacle [0,0]),\
                (value robot_init_pose [-2,0]),\
                (value target [2,0]),\
                (pose ws [0,0]), (w ws 8), (h ws 9), (size ws 1), (viewer ws);\
                (RobotAt pr2 robot_init_pose)',
                'Objects': 'RobotPose (name target); Robot (name pr2); Obstacle (name obstacle); RobotPose (name robot_init_pose); RobotPose (name target); Workspace (name ws)',
                'Goal': '(RobotAt pr2 target)'}
        hls = hl_solver.FFSolver(d_c)
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        self.move_w_obs = hls.solve(hls.translate_problem(problem), domain, problem)
        # import ipdb; ipdb.set_trace()

    def test_llparam(self):
        # TODO: tests for undefined, partially defined and fully defined params
        plan = self.move_no_obs
        horizon = plan.horizon
        move = plan.actions[0]
        pr2 = move.params[0]
        robot_init_pose = move.params[1]
        start = move.params[1]
        end = move.params[2]

        model = grb.Model()
        model.params.OutputFlag = 0 # silences Gurobi output

        # pr2 is an Object parameter
        pr2_ll = ll_solver.LLParam(model, pr2, horizon)
        pr2_ll.create_grb_vars()
        self.assertTrue(pr2_ll.pose.shape == (2, horizon))
        with self.assertRaises(AttributeError):
            pr2_ll._type
        with self.assertRaises(AttributeError):
            pr2_ll.geom
        model.update()
        obj = grb.QuadExpr()
        obj += pr2_ll.pose[0,0]*pr2_ll.pose[0,0] + \
                pr2_ll.pose[1,0]*pr2_ll.pose[1,0]
        model.setObjective(obj)
        model.optimize()
        self.assertTrue(np.allclose(pr2_ll.pose[0,0].X, 0.))
        self.assertTrue(np.allclose(pr2_ll.pose[1,0].X, 0.))

        pr2_ll.batch_add_cnts()
        model.optimize()
        self.assertTrue(np.allclose(pr2_ll.pose[0,0].X, 0.))
        self.assertTrue(np.allclose(pr2_ll.pose[1,0].X, 7.))
        # x1^2 + x2^2 - 2x
        obj = grb.QuadExpr()
        obj += pr2_ll.pose[0,1]*pr2_ll.pose[0,1] + \
                pr2_ll.pose[1,1]*pr2_ll.pose[1,1]- 2*pr2_ll.pose[1,1]
        model.setObjective(obj)
        model.optimize()
        self.assertTrue(np.allclose(pr2_ll.pose[0,0].X, 0.))
        self.assertTrue(np.allclose(pr2_ll.pose[1,0].X, 7.))

        self.assertTrue(np.allclose(pr2_ll.pose[0,1].X, 0.))
        self.assertTrue(np.allclose(pr2_ll.pose[1,1].X, 1.))
        pr2_ll.update_param()
        self.assertTrue(np.allclose(pr2.pose[0,1], 0.))
        self.assertTrue(np.allclose(pr2.pose[1,1], 1.))

        # robot_init_pose is a Symbol parameter
        model = grb.Model()
        model.params.OutputFlag = 0 # silences Gurobi output
        robot_init_pose_ll = ll_solver.LLParam(model, robot_init_pose, horizon)
        robot_init_pose_ll.create_grb_vars()
        self.assertTrue(robot_init_pose_ll.value.shape == (2,1))
        with self.assertRaises(AttributeError):
            pr2_ll._type
        with self.assertRaises(AttributeError):
            pr2_ll.geom

    def test_namo_solver_one_move_plan(self):
        plan = self.move_no_obs
        move = plan.actions[0]
        pr2 = move.params[0]
        start = move.params[1]
        end = move.params[2]
        model = grb.Model()
        model.params.OutputFlag = 0 # silences Gurobi output
        namo_solver = ll_solver.NAMOSolver()
        namo_solver._prob = Prob(model)

        namo_solver._spawn_parameter_to_ll_mapping(model, plan)
        model.update()
        namo_solver._add_actions_to_sco_prob(plan)

        pr2_ll = namo_solver._param_to_ll[pr2]
        start_ll = namo_solver._param_to_ll[start]
        end_ll = namo_solver._param_to_ll[end]
        # optimize without trajopt objective
        model.optimize()
        namo_solver._update_ll_params()
        arr = np.zeros((2, plan.horizon))
        arr[:,0] = [0., 7.]
        self.assertTrue(np.allclose(pr2.pose, arr))
        self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
        self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))


        # setting up trajopt objective
        T = plan.horizon
        K = 2
        KT = K*T
        v = -1 * np.ones((KT - K, 1))
        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
        # [:,0] allows numpy to see v and d as one-dimensional so
        # that numpy will create a diagonal matrix with v and d as a diagonal
        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
        Q = np.dot(np.transpose(P), P)

        quad_expr = expr.QuadExpr(Q, np.zeros((1,KT)), np.zeros((1,1)))
        pr2_ll_grb_vars = pr2_ll.pose.reshape((pr2_ll.pose.size, 1), order='F')
        bexpr = expr.BoundExpr(quad_expr, Variable(pr2_ll_grb_vars))
        namo_solver._prob.add_obj_expr(bexpr)
        sco_solver = Solver()
        sco_solver.solve(namo_solver._prob, method='penalty_sqp')
        namo_solver._update_ll_params()

        arr1 = np.zeros(plan.horizon)
        arr2 = np.linspace(7,0, num=20)
        arr = np.c_[arr1, arr2].T
        self.assertTrue(np.allclose(pr2.pose, arr))
        self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
        self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))

    def test_namo_solver_one_move_plan_solve(self):
        plan = self.move_no_obs
        move = plan.actions[0]
        pr2 = move.params[0]
        start = move.params[1]
        end = move.params[2]

        namo_solver = ll_solver.NAMOSolver()
        namo_solver.solve(plan)

        arr1 = np.zeros(plan.horizon)
        arr2 = np.linspace(7,0, num=20)
        arr = np.c_[arr1, arr2].T
        self.assertTrue(np.allclose(pr2.pose, arr))
        self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
        self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))

        """
        Uncomment following three lines to view trajectory
        """
        # viewer = OpenRAVEViewer()
        # viewer.draw_traj([pr2], range(20))
        # time.sleep(3)


    def test(self):
        plan = self.one_move_plan
        horizon = plan.horizon
        self.assertEqual(True, True)
