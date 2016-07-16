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
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.internal_repr import parameter
import time, main

class TestLLSolver(unittest.TestCase):
    def setUp(self):

        domain_fname = '../domains/namo_domain/namo.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        def get_plan(p_fname):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            return hls.solve(abs_problem, domain, problem)
            
        self.move_no_obs = get_plan('../domains/namo_domain/namo_probs/ll_solver_one_move.prob')
        self.move_grasp = get_plan('../domains/namo_domain/namo_probs/move_grasp.prob')
        self.move_grasp_moveholding = get_plan('../domains/namo_domain/namo_probs/moveholding.prob')
        self.place = get_plan('../domains/namo_domain/namo_probs/place.prob')

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
        self.assertTrue(np.allclose(pr2_ll.pose[0,0].X, robot_init_pose.value[0,0]))
        self.assertTrue(np.allclose(pr2_ll.pose[1,0].X, robot_init_pose.value[1,0]))
        # x1^2 + x2^2 - 2x
        obj = grb.QuadExpr()
        obj += pr2_ll.pose[0,1]*pr2_ll.pose[0,1] + \
                pr2_ll.pose[1,1]*pr2_ll.pose[1,1]- 2*pr2_ll.pose[1,1]
        model.setObjective(obj)
        model.optimize()
        self.assertTrue(np.allclose(pr2_ll.pose[0,0].X, robot_init_pose.value[0,0]))
        self.assertTrue(np.allclose(pr2_ll.pose[1,0].X, robot_init_pose.value[1,0]))

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

    # def test_namo_solver_init_params(self):
    #     class dummy_plan(object):
    #         def __init__(self, params):
    #             self.params = params
    #
    #     radius = 1.0
    #
    #     attrs = {"geom": [radius], "pose": [(0, 0)], "_type": ["Can"], "name": ["can0"]}
    #     attr_types = {"geom": circle.GreenCircle, "pose": Vector2d, "_type": str, "name": str}
    #     green_can = parameter.Object(attrs, attr_types)
    #
    #     attrs = {"name": ["robot_pose"], "value": [(0,0)], "_type": ["RobotPose"]}
    #     attr_types = {"name": str, "value": Vector2d, "_type": str}
    #     rp = parameter.Symbol(attrs, attr_types)
    #     # test random seeds
    #     pass
    #
    # def test_solver(self):
    #     # check that parameter values don't change
    #     pass
    #
    # def test_namo_solver_one_move_plan(self):
    #     plan = self.move_no_obs
    #     move = plan.actions[0]
    #     pr2 = move.params[0]
    #     start = move.params[1]
    #     end = move.params[2]
    #     model = grb.Model()
    #     model.params.OutputFlag = 0 # silences Gurobi output
    #     namo_solver = ll_solver.NAMOSolver()
    #     namo_solver._prob = Prob(model)
    #
    #     namo_solver._spawn_parameter_to_ll_mapping(model, plan)
    #     model.update()
    #     namo_solver._add_actions_to_sco_prob(plan)
    #
    #     pr2_ll = namo_solver._param_to_ll[pr2]
    #     start_ll = namo_solver._param_to_ll[start]
    #     end_ll = namo_solver._param_to_ll[end]
    #     # optimize without trajopt objective
    #     model.optimize()
    #     namo_solver._update_ll_params()
    #     arr = np.zeros((2, plan.horizon))
    #     arr[:,0] = [0., 7.]
    #     self.assertTrue(np.allclose(pr2.pose, arr))
    #     self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
    #     self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))
    #
    #
    #     # setting up trajopt objective
    #     T = plan.horizon
    #     K = 2
    #     KT = K*T
    #     v = -1 * np.ones((KT - K, 1))
    #     d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
    #     # [:,0] allows numpy to see v and d as one-dimensional so
    #     # that numpy will create a diagonal matrix with v and d as a diagonal
    #     P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
    #     Q = np.dot(np.transpose(P), P)
    #
    #     quad_expr = expr.QuadExpr(Q, np.zeros((1,KT)), np.zeros((1,1)))
    #     pr2_ll_grb_vars = pr2_ll.pose.reshape((pr2_ll.pose.size, 1), order='F')
    #     bexpr = expr.BoundExpr(quad_expr, Variable(pr2_ll_grb_vars))
    #     namo_solver._prob.add_obj_expr(bexpr)
    #     sco_solver = Solver()
    #     sco_solver.solve(namo_solver._prob, method='penalty_sqp')
    #     namo_solver._update_ll_params()
    #
    #     arr1 = np.zeros(plan.horizon)
    #     arr2 = np.linspace(7,0, num=20)
    #     arr = np.c_[arr1, arr2].T
    #     self.assertTrue(np.allclose(pr2.pose, arr))
    #     self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
    #     self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))
    #

    def test_namo_solver_one_move_plan_solve_init(self):
        # return
        plan = self.move_no_obs
        # import ipdb; ipdb.set_trace()
        move = plan.actions[0]
        pr2 = move.params[0]
        start = move.params[1]
        end = move.params[2]

        plan_params = plan.params.values()
        for action in plan.actions:
            for p in action.params:
                self.assertTrue(p in plan_params)
            for pred_dict in action.preds:
                pred = pred_dict['pred']
                for p in pred.params:
                    if p not in plan_params:
                        if pred_dict['hl_info'] != 'hl_state':
                            print pred
                            break
                    # self.assertTrue(p in plan_params)

        callback = None
        """
        Uncomment out lines below to see optimization.
        """
        # viewer = OpenRAVEViewer()
        # def callback():
        #     namo_solver._update_ll_params()
        #     viewer.draw_plan(plan)
        #     time.sleep(0.3)
        """
        """
        namo_solver = ll_solver.NAMOSolver()
        namo_solver._solve_opt_prob(plan, priority=-1, callback=callback)
        namo_solver._update_ll_params()

        # arr1 = np.zeros(plan.horizon)
        # arr2 = np.linspace(7,0, num=20)
        # arr = np.c_[arr1, arr2].T
        # self.assertTrue(np.allclose(pr2.pose, arr))
        # self.assertTrue(np.allclose(start.value, np.array([[0.],[7.]])))
        # self.assertTrue(np.allclose(end.value, np.array([[0.],[0.]])))

        # """
        # Uncomment following three lines to view trajectory
        # """
        # # viewer.draw_traj([pr2], range(20))
        # viewer.draw_plan(plan)
        # import ipdb; ipdb.set_trace()
        # time.sleep(3)

    def test_namo_solver_one_move_plan_solve(self):
        _test_plan(self, self.move_no_obs)

    def test_move_grasp(self):
        _test_plan(self, self.move_grasp)

    def test_moveholding(self):
        _test_plan(self, self.move_grasp_moveholding)

    def test_place(self):
        _test_plan(self, self.place)

    def test_initialize_params(self):
        plan = self.move_no_obs

        namo_solver = ll_solver.NAMOSolver()
        namo_solver._initialize_params(plan)

        for p in plan.params.itervalues():
            self.assertTrue(p in namo_solver._init_values)


def _test_plan(test_obj, plan):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    viewer = OpenRAVEViewer.create_viewer()
    # def callback():
    #     namo_solver._update_ll_params()
    #     viewer.draw_plan(plan)
    #     time.sleep(0.03)
    """
    """
    namo_solver = ll_solver.NAMOSolver()
    namo_solver.solve(plan, callback=callback)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    
    if viewer != None:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)
        
    test_obj.assertTrue(plan.satisfied())
    

if __name__ == "__main__":
    unittest.main()
