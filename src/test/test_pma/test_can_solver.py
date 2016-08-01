import unittest
from pma import hl_solver
from pma import can_solver
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

class TestCanSolver(unittest.TestCase):
    def setUp(self):
        domain_fname = '../domains/can_domain/can.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        def get_plan(p_fname, plan_str=None):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            if plan_str is not None:
                return hls.get_plan(plan_str, domain, problem)
            # view = OpenRAVEViewer()
            # robot = problem.init_state.params['pr2']
            # table = problem.init_state.params['table']
            # cans = []
            # for param_name, param in problem.init_state.params.iteritems():
            #     if "can" in param_name:
            #         cans.append(param)
            # objs = [robot, table]
            # objs.extend(cans)
            # view.draw(objs, 0, 0.7)
            return hls.solve(abs_problem, domain, problem)

        # self.move_no_obs = get_plan('../domains/can_domain/can_probs/move.prob')
        # self.move_no_obs = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.grasp = get_plan('../domains/can_domain/can_probs/grasp.prob')
        self.grasp = get_plan('../domains/can_domain/can_probs/grasp_rot.prob')
        # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob', ['0: MOVETOHOLDING PR2 ROBOT_INIT_POSE ROBOT_END_POSE CAN0'])
        # # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.gen_plan = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')

    def test_move(self):
        pass
        # _test_plan(self, self.move_no_obs)

    def test_move_obs(self):
        pass
        # _test_plan(self, self.move_obs)

    def test_grasp(self):
        pass
        # _test_plan(self, self.grasp)

    def test_grasp_resampling(self):
        _test_plan(self, self.grasp, n_resamples=3)

    def test_moveholding(self):
        pass
        # _test_plan(self, self.moveholding)

    def test_gen_plan(self):
        pass
        # _test_plan(self, self.gen_plan)

def _test_plan(test_obj, plan, n_resamples=0):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    viewer = OpenRAVEViewer.create_viewer()
    # def callback():
    #     solver._update_ll_params()
    # #     obj_list = viewer._get_plan_obj_list(plan)
    # #     # viewer.draw_traj(obj_list, [0,9,19,20,29,38])
    # #     # viewer.draw_traj(obj_list, range(19,30))
    # #     # viewer.draw_traj(obj_list, range(29,39))
    # #     # viewer.draw_traj(obj_list, [38])
    # #     # viewer.draw_traj(obj_list, range(19,39))
    # #     # viewer.draw_plan_range(plan, [0,19,38])
    #     viewer.draw_plan(plan)
    #     time.sleep(0.03)
    """
    """
    solver = can_solver.CanSolver()
    solver.solve(plan, callback=callback, n_resamples=n_resamples, verbose=True)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    test_obj.assertTrue(plan.satisfied())


if __name__ == "__main__":
    unittest.main()
