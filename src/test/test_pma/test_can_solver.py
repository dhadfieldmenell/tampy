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
from core.internal_repr import parameter, plan
import time, main

VIEWER = True
FAKE_TOL = 1e-2 # Not used......

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
            # for param_name, param in problem.init_state.params.items():
            #     if "can" in param_name:
            #         cans.append(param)
            # objs = [robot, table]
            # objs.extend(cans)
            # view.draw(objs, 0, 0.7)
            return hls.solve(abs_problem, domain, problem)

        self.bmove = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')

        # self.move_obs = get_plan('../domains/can_domain/can_probs/move_obs.prob')

        # self.move_no_obs = get_plan('../domains/can_domain/can_probs/move.prob')
        # self.move_no_obs = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.grasp = get_plan('../domains/can_domain/can_probs/grasp.prob')
        # self.grasp = get_plan('../domains/can_domain/can_probs/grasp_rot.prob')
        # self.grasp_gen = get_plan('../domains/can_domain/can_probs/can_grasp_1234_0.prob')
        # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob', ['0: MOVETOHOLDING PR2 ROBOT_INIT_POSE ROBOT_END_POSE CAN0'])
        # # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.gen_plan = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.grasp_obstructs1 = get_plan('../domains/can_domain/can_probs/can_grasp_1234_1.prob', ['0: GRASP PR2 CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 PDP_TARGET0'])
        # self.grasp_obstructs0 = get_plan('../domains/can_domain/can_probs/can_grasp_1234_0.prob', ['0: GRASP PR2 CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 PDP_TARGET0'])

        # self.grasp_obstructs = get_plan('../domains/can_domain/can_probs/can_grasp_1234_4.prob', ['0: GRASP PR2 CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 PDP_TARGET0'])

        if VIEWER:
            self.viewer = OpenRAVEViewer.create_viewer()
        else:
            self.viewer = None

    # def test_backtrack_move_grasp(self):
    #     _test_backtrack_plan(self, self.bmove_grasp)

    # def test_move(self):
    #     _test_plan(self, self.move_no_obs)
    #
    # def test_backtrack_move(self):
    #     _test_backtrack_plan(self, self.move_no_obs, method='Backtrack', plot = True)


    def test_move_obs(self):
        pass
        # _test_plan(self, self.bmove)

    # def test_grasp_gen(self):
    #     np.random.seed(1)
    #     _test_plan(self, self.grasp_gen, n_resamples=5)

    # def test_grasp(self):
    #     np.random.seed(1)
    #     _test_plan(self, self.grasp, n_resamples=0)

    # def test_grasp_resampling(self):
    #     # np.random.seed(4)
    #     np.random.seed(3) # demonstrates the need to use closest joint angles
    #     _test_resampling(self, self.grasp_obstructs0, n_resamples=3)

        # demonstate base moving from farther away
        # np.random.seed(2)
        # _test_resampling(self, self.grasp_obstructs0, n_resamples=3)

        # demonstrates base moving
        # np.random.seed(6) # forms right angle
        # _test_resampling(self, self.grasp_obstructs1, n_resamples=3)

    def test_grasp_obstructs(self):
        pass
        # _test_plan(self, self.grasp, n_resamples=3)

    def test_moveholding(self):
        pass
        # _test_plan(self, self.moveholding)

    def test_gen_plan(self):
        pass
        # _test_plan(self, self.gen_plan)

def get_animate_fn(viewer, plan):
    def animate():
        viewer.animate_plan(plan)
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
        # draw_ts(20)
        # viewer.draw_plan(plan)
        draw_ts(17)
        # viewer.draw_cols_ts(plan, 17)
        # time_range = (13,17)
        # viewer.draw_plan_range(plan, time_range)
        # viewer.draw_cols_range(plan, time_range)
        # time.sleep(0.03)
        # if set_trace:
        #     animate()
            # import ipdb; ipdb.set_trace()
    """
    """
    solver = can_solver.CanSolver()

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
    # if viewer != None:
    #     viewer = OpenRAVEViewer.create_viewer()
    #     viewer.animate_plan(plan)
    #     if t < plan.horizon:
    #         viewer.draw_plan_ts(plan, t)

    test_obj.assertTrue(plan.satisfied(FAKE_TOL))


def _test_plan(test_obj, plan, n_resamples=0):
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
    # #     obj_list = viewer._get_plan_obj_list(plan)
    # #     # viewer.draw_traj(obj_list, [0,9,19,20,29,38])
    # #     # viewer.draw_traj(obj_list, range(19,30))
    # #     # viewer.draw_traj(obj_list, range(29,39))
    # #     # viewer.draw_traj(obj_list, [38])
    # #     # viewer.draw_traj(obj_list, range(19,39))
    # #     # viewer.draw_plan_range(plan, [0,19,38])
        # draw_ts(50)
        if set_trace:
            animate()
            # import ipdb; ipdb.set_trace()

        # viewer.draw_plan(plan)
        # time.sleep(0.03)
    """
    """
    solver = can_solver.CanSolver()
    solver.solve(plan, callback=callback, n_resamples=n_resamples, verbose=False)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)
    # import ipdb; ipdb.set_trace()

    # test_obj.assertTrue(plan.satisfied(FAKE_TOL))

def _test_backtrack_plan(test_obj, plan, n_resamples=0):
    print "testing plan: {}".format(plan.actions)
    callback = None
    viewer = None
    solver = can_solver.CanSolver()
    """
    Uncomment out lines below to see optimization.
    """
    viewer = OpenRAVEViewer.create_viewer()
    animate = get_animate_fn(viewer, plan)
    def callback(a):
        solver._update_ll_params()
        # viewer.clear()
        viewer.draw_plan_range(plan, a.active_timesteps)
        # time.sleep(0.3)
    """
    """

    solver.backtrack_solve(plan, callback=None, anum = 0, verbose=True)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    # import ipdb; ipdb.set_trace()
    test_obj.assertTrue(plan.satisfied(0))

def _test_backtrack_plan(test_obj, plan, method='SQP', plot=False, animate=True, verbose=False,
               early_converge=False):
    print "testing plan: {}".format(plan.actions)
    if not plot:
        callback = None
        viewer = None
    else:
        viewer = OpenRAVEViewer.create_viewer()
        if method=='SQP':
            def callback():
                solver._update_ll_params()
                # viewer.draw_plan_range(plan, range(57, 77)) # displays putdown action
                # viewer.draw_plan_range(plan, range(38, 77)) # displays moveholding and putdown action
                viewer.draw_plan(plan)
                # viewer.draw_cols(plan)
                time.sleep(0.03)
        elif method == 'Backtrack':
            def callback(a):
                solver._update_ll_params()
                viewer.clear()
                viewer.draw_plan_range(plan, a.active_timesteps)
                time.sleep(0.3)
    solver = can_solver.CanSolver(early_converge=early_converge)
    start = time.time()
    if method == 'SQP':
        solver.solve(plan, callback=callback, verbose=verbose)
    elif method == 'Backtrack':
        solver.backtrack_solve(plan, callback=callback, verbose=verbose)
    print "Solve Took: {}".format(time.time() - start)
    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    if animate:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    # test_obj.assertTrue(plan.satisfied())

if __name__ == "__main__":
    unittest.main()
