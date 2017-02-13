import unittest
from pma import hl_solver
from pma import robot_ll_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
import gurobipy as grb
import numpy as np
from sco.prob import Prob
from sco.solver import Solver
from sco.variable import Variable
from sco import expr
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import circle
from core.util_classes.matrix import Vector2d
from core.internal_repr import parameter, plan
from core.util_classes import plan_hdf5_serialization
import time, main

VIEWER = False

class TestRobotLLSolver(unittest.TestCase):
    def setUp(self):
        """
            This function sets up the planner domain and problem class for optimization.
        """
        domain_fname = '../domains/baxter_domain/baxter.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        def get_plan(p_fname, plan_str=None):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            if plan_str is not None:
                return hls.get_plan(plan_str, domain, problem)
            return hls.solve(abs_problem, domain, problem)
        self.get_plan = get_plan
        # Successful Problem
        # self.move_arm_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_move_arm.prob')
        # self.grab_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])
        # self.move_hold_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_move_holding.prob', ['0: MOVETOHOLDING BAXTER ROBOT_INIT_POSE ROBOT_END_POSE CAN0'])
        # self.complex_grab_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_complex_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])

        # Problem for testing
        # self.putdown_prob = get_plan('../domains/baxter_domain/baxter_probs/putdown_1234_0.prob', ['0: PUTDOWN BAXTER CAN0 TARGET2 ROBOT_INIT_POSE EE_TARGET2 ROBOT_END_POSE'])

        # Problem for test_free_attrs test
        # self.test_free_attrs_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_complex_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])

    # Helper function used for debug purposes


    # Successful plan
    # def test_move_prob(self):
    #     _test_plan(self, self.move_arm_prob)
    # def test_grab_prob(self):
    #     _test_plan(self, self.grab_prob)
    # def test_move_holding(self):
    #     _test_plan(self, self.move_hold_prob)

    # Need to be tested plan
    def test_complex_grab_prob(self):
        complete_time = []
        for i in range(50):
            self.complex_grab_prob = self.get_plan('../domains/baxter_domain/baxter_probs/baxter_complex_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])
            complete_time.append(_test_plan(self, self.complex_grab_prob))
        print complete_time
        import ipdb; ipdb.set_trace()
    # def test_putdown_prob(self):
    #     _test_plan(self, self.putdown_prob)
    # def test_from_saved_plan(self):
    #     _test_resampling(self)

    # def test_saved_plan(self):
    #     # plan = self.complex_grab_prob
    #     plan_reader = plan_hdf5_serialization.PlanDeserializer()
    #     plan = plan_reader.read_from_hdf5("saved_plan2.hdf5")
    #     import ipdb; ipdb.set_trace()
    #     callback, verbose = None, False
    #     viewer = OpenRAVEViewer(plan.env)
    #     animate = get_animate_fn(viewer, plan)
    #     draw_ts = get_draw_ts_fn(viewer, plan)
    #     draw_cols_ts = get_draw_cols_ts_fn(viewer, plan)
    #     solver = robot_ll_solver.RobotLLSolver()
    #     # Initializing to sensible values
    #     active_ts = (0, plan.horizon-1)
    #     #
    #     # solver._solve_opt_prob(plan, priority=-2, callback=callback, active_ts=active_ts, verbose=verbose)
    #     # solver._solve_opt_prob(plan, priority=-1, callback=callback, active_ts=active_ts, verbose=verbose)
    #     # serializer = plan_hdf5_serialization.PlanSerializer()
    #     # serializer.write_plan_to_hdf5("saved_plan1", plan)
    #
    #
    #     import ipdb; ipdb.set_trace()


"""
    Helper functions that enables test on motion planning
"""

def get_animate_fn(viewer, plan):
    def animate(delay = 0.5):
        viewer.animate_plan(plan, delay)
    return animate

def get_draw_ts_fn(viewer, plan):
    def draw_ts(ts):
        viewer.draw_plan_ts(plan, ts)
    return draw_ts

def get_draw_cols_ts_fn(viewer, plan):
    def draw_cols_ts(ts):
        viewer.draw_cols_ts(plan, ts)
    return draw_cols_ts

def _test_resampling(test_obj, n_resamples=0):
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    plan_reader = plan_hdf5_serialization.PlanDeserializer()
    plan = plan_reader.read_from_hdf5("initialized_plan.hdf5")

    viewer = OpenRAVEViewer(plan.env)
    animate = get_animate_fn(viewer, plan)
    draw_ts = get_draw_ts_fn(viewer, plan)
    draw_cols_ts = get_draw_cols_ts_fn(viewer, plan)
    def callback(set_trace=False):
        if set_trace:
            import ipdb; ipdb.set_trace()
        return viewer
        # solver._update_ll_params()
    """
    """
    solver = robot_ll_solver.RobotLLSolver()
    # Initializing to sensible values
    active_ts = (0, plan.horizon-1)
    verbose = False
    fp = plan.get_failed_preds()
    import ipdb; ipdb.set_trace()
    if len(fp) > 0:
        solver._solve_opt_prob(plan, priority=0, callback=calloback, active_ts=active_ts, verbose=verbose)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()

    print(plan.satisfied())


def _test_plan(test_obj, plan, n_resamples=10):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    viewer = OpenRAVEViewer.create_viewer(plan.env)

    animate = get_animate_fn(viewer, plan)
    draw_ts = get_draw_ts_fn(viewer, plan)
    draw_cols_ts = get_draw_cols_ts_fn(viewer, plan)
    def callback(set_trace=False):
        if set_trace:
            animate()
        return viewer
    """
    """
    solver = robot_ll_solver.RobotLLSolver()
    timesteps = solver.solve(plan, callback=callback, n_resamples=50, verbose=False)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    print(plan.get_failed_preds())
    # import ipdb; ipdb.set_trace()
    return timesteps
    # assert plan.satisfied()


if __name__ == "__main__":
    unittest.main()
