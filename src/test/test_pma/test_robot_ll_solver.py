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
        # Successful Problem
        # self.move_arm_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_move_arm.prob')
        # self.grab_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])
        # self.move_hold_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_move_holding.prob', ['0: MOVETOHOLDING BAXTER ROBOT_INIT_POSE ROBOT_END_POSE CAN0'])

        # Problem for testing
        self.complex_grab_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_complex_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])

        # self.putdown_prob = get_plan('../domains/baxter_domain/baxter_probs/putdown_1234_0.prob', ['0: PUTDOWN BAXTER CAN0 TARGET2 ROBOT_INIT_POSE EE_TARGET2 ROBOT_END_POSE'])

        # Problem for test_free_attrs test
        self.test_free_attrs_prob = get_plan('../domains/baxter_domain/baxter_probs/baxter_complex_grasp.prob', ['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'])

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
        _test_plan(self, self.complex_grab_prob)
    # def test_putdown_prob(self):
    #     _test_plan(self, self.putdown_prob)

    
    # def test_free_attrs(self):
    #     plan = self.test_free_attrs_prob
    #     viewer = OpenRAVEViewer.create_viewer(plan.env)
    #     def callback(set_trace=False):
    #         if set_trace:
    #             animate()
    #         return viewer
    #     solver = robot_ll_solver.RobotLLSolver()
    #     # solver.solve(plan, callback=callback, n_resamples=10, verbose=False)
    #     success = False
    #     solver._solve_opt_prob(plan, priority=-2, callback=None, active_ts=None, verbose=False)
    #     solver._solve_opt_prob(plan, priority=-1, callback=None, active_ts=None, verbose=False)
    #     success = solver._solve_helper(plan, callback=None, active_ts=None, verbose=False)
    #
    #     robot = plan.params['baxter']
    #     body = plan.env.GetRobot("baxter")
    #     def draw(t):
    #         viewer.draw_plan_ts(plan, t)
    #     ## Backup the free_attrs value
    #     plan.save_free_attrs()
    #
    #     model = grb.Model()
    #     model.params.OutputFlag = 0
    #     solver._prob = Prob(model, callback=callback)
    #     solver._spawn_parameter_to_ll_mapping(model, plan, (0, plan.horizon-1))
    #     model.update()
    #     solver._bexpr_to_pred = {}
    #
    #     failed_preds = plan.get_failed_preds()
    #     if len(failed_preds) <= 0:
    #         success = True
    #         import ipdb; ipdb.set_trace()
    #     print "{} predicates fails, resampling process begin...\n Checking {}".format(len(failed_preds), failed_preds[0])
    #     ## this is an objective that places
    #     ## a high value on matching the resampled values
    #     obj_bexprs, resample_timestep = [], []
    #     obj_bexprs.extend(solver._resample(plan, failed_preds, resample_timestep))
    #     ## solve an optimization movement primitive to
    #     ## transfer current trajectories
    #     obj_bexprs.extend(solver._get_trajopt_obj(plan, (0, plan.horizon-1)))
    #
    #     solver._add_obj_bexprs(obj_bexprs)
    #     solver._add_all_timesteps_of_actions(
    #         plan, priority=1, add_nonlin=False, active_ts=(0, resample_timestep[0]-1),  verbose=False)
    #     solver._add_all_timesteps_of_actions(
    #         plan, priority=2, add_nonlin=True, active_ts= (resample_timestep[0], resample_timestep[-1]),  verbose=False)
    #     solver._add_all_timesteps_of_actions(
    #         plan, priority=1, add_nonlin=False, active_ts=(resample_timestep[-1]+1, plan.horizon-1),  verbose=False)
    #     tol = 1e-3
    #
    #     solv = Solver()
    #     solv.initial_trust_region_size = solver.initial_trust_region_size
    #     solv.initial_penalty_coeff = solver.init_penalty_coeff
    #     solv.max_merit_coeff_increases = solver.max_merit_coeff_increases
    #
    #     sampled_pose = robot.rArmPose[:, 30]
    #
    #     self.assertTrue(np.all(robot._free_attrs["rArmPose"][:, 30] ==  np.zeros((7,))))
    #     success = solv.solve(solver._prob, method='penalty_sqp', tol=tol, verbose=True)
    #     resulted_pose = robot.rArmPose[:, 30]
    #     self.assertTrue(np.all(sampled_pose == resulted_pose))
    #     solver._update_ll_params()
    #     plan.restore_free_attrs()
    #     self.assertTrue(np.all(robot._free_attrs["rArmPose"][:, 30] ==  np.ones((7,))))
    #     import ipdb; ipdb.set_trace()


"""
    Helper functions that enables test on motion planning
"""

def get_animate_fn(viewer, plan):
    def animate():
        viewer.animate_plan(plan, delay = 0.5)
    return animate

def get_draw_ts_fn(viewer, plan):
    def draw_ts(ts):
        viewer.draw_plan_ts(plan, ts)
    return draw_ts

def get_draw_cols_ts_fn(viewer, plan):
    def draw_cols_ts(ts):
        viewer.draw_cols_ts(plan, ts)
    return draw_cols_ts

def _test_resampling(test_obj, plan, n_resamples=0):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    viewer = test_obj.viewer
    animate = get_animate_fn(viewer, plan)
    draw_ts = get_draw_ts_fn(viewer, plan)
    draw_cols_ts = get_draw_cols_ts_fn(viewer, plan)
    def callback(set_trace=False):
        solver._update_ll_params()
    """
    """
    solver = robot_ll_solver.RobotLLSolver()

    # Initializing to sensible values
    active_ts = (0, plan.horizon-1)
    verbose = False
    solver._solve_opt_prob(plan, priority=-2, callback=callback, active_ts=active_ts, verbose=verbose)
    solver._solve_opt_prob(plan, priority=-1, callback=callback, active_ts=active_ts, verbose=verbose)


    for _ in range(n_resamples):
        ## refinement loop
        ## priority 0 resamples the first failed predicate in the plan
        ## and then solves a transfer optimization that only includes linear constraints
        solver._solve_opt_prob(plan, priority=0, callback=callback, active_ts=active_ts, verbose=verbose)
        fp = plan.get_failed_preds()
        # import ipdb; ipdb.set_trace()

        solver._solve_opt_prob(plan, priority=1, callback=calloback, active_ts=active_ts, verbose=verbose)
        fp = plan.get_failed_preds()
        # import ipdb; ipdb.set_trace()

        success = solver._solve_opt_prob(plan, priority=2, callback=callback, active_ts=active_ts, verbose=verbose)
        fp = plan.get_failed_preds()
        # import ipdb; ipdb.set_trace()
        if len(fp) == 0:
            break

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)
    import ipdb;ipdb.set_trace()
    print(plan.satisfied())


def _test_plan(test_obj, plan, n_resamples=10):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    """
    Uncomment out lines below to see optimization.
    """
    viewer = OpenRAVEViewer(plan.env)

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
    solver.solve(plan, callback=callback, n_resamples=10, verbose=False)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    print(plan.get_failed_preds())
    import ipdb; ipdb.set_trace()
    # assert plan.satisfied()


if __name__ == "__main__":
    unittest.main()
