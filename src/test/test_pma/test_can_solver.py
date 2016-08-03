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
        # self.move_obs = get_plan('../domains/can_domain/can_probs/move_obs.prob')
        # self.grasp = get_plan('../domains/can_domain/can_probs/grasp.prob')
        # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob', ['0: MOVETOHOLDING PR2 ROBOT_INIT_POSE ROBOT_END_POSE CAN0'])
        # self.moveholding = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')
        # self.gen_plan = get_plan('../domains/can_domain/can_probs/can_1234_0.prob')

    # def test_move(self):
    #     _test_plan(self, self.move_no_obs)
    #
    # def test_backtrack_move(self):
    #     _test_backtrack_plan(self, self.move_no_obs, method='Backtrack', plot = True)

    def test_move_obs(self):
        pass
        # _test_plan(self, self.move_obs)

    def test_grasp(self):
        pass
        # _test_plan(self, self.grasp)

    def test_moveholding(self):
        pass
        # _test_plan(self, self.moveholding)

    def test_gen_plan(self):
        pass
        # _test_plan(self, self.gen_plan)

    def test_sample_ee_from_target(self):
        from openravepy import Environment
        from core.util_classes.can import BlueCan, GreenCan
        from core.util_classes.openrave_body import OpenRAVEBody
        from core.util_classes import matrix
        from core.util_classes.pr2 import PR2
        solver = can_solver.CanSolver()
        env = Environment()
        # env.SetViewer('qtcoin')
        attrs = {"name": ['targ'], "value": [(0, 1, .8)], "rotation": [(0,0,0)], "_type": ["Target"]}
        attr_types = {"name": str, "value": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
        target = parameter.Symbol(attrs, attr_types)
        target.rotation = np.array([[1.1,.3,0]]).T
        dummy_targ_geom = BlueCan(0.04, 0.25)
        target_body = OpenRAVEBody(env, target.name, dummy_targ_geom)
        target_body.set_pose(target.value.flatten(), target.rotation.flatten())
        target_body.set_transparency(.7)
        dummy_ee_pose_geom = GreenCan(.03,.3)
        ee_list = list(enumerate(solver.sample_ee_from_target(target)))
        for ee_pose in ee_list:
            ee_pos, ee_rot = ee_pose[1]
            body = OpenRAVEBody(env, "dummy"+str(ee_pose[0]), dummy_ee_pose_geom)
            body.set_pose(ee_pos, ee_rot)
            body.set_transparency(.9)

        attrs = {"name": ['pr2'], "pose": [(-.45, 1.19,-.1)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
        attrs["lArmPose"] = [(np.pi/4, np.pi/8, np.pi/2, -np.pi/2, np.pi/8, -np.pi/8, np.pi/2)]
        attrs["rArmPose"] = [(-np.pi/4, np.pi/8, -np.pi/2, -np.pi/2, -np.pi/8, -np.pi/8, np.pi/2)]
        attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
        robot = parameter.Object(attrs, attr_types)
        robot_body = OpenRAVEBody(env, robot.name, robot.geom)
        robot_body.set_transparency(.7)
        robot_body.set_pose(robot.pose.flatten())
        robot_body.set_dof(robot.backHeight, robot.lArmPose.flatten(), robot.lGripper, robot.rArmPose.flatten(), robot.rGripper)
        def set_arm(n):
            pos, rot = ee_list[n][1][0], ee_list[n][1][1]
            iksol = robot_body.ik_arm_pose(pos, rot)
            for k in range(len(iksol)):
                robot_body.set_pose(robot.pose.flatten())
                robot_body.set_dof(iksol[k][0], robot.lArmPose.flatten(), robot.lGripper, iksol[k][1:], robot.rGripper)
                time.sleep(0.03)
        # plot all the possible gripping position
        # import ipdb; ipdb.set_trace()
        # for i in range(50):
        #     set_arm(i)
        #     time.sleep(0.03)
        # import ipdb; ipdb.set_trace()

def _test_plan(test_obj, plan):
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
    solver.solve(plan, callback=callback, n_resamples=0, verbose=True)

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    #
    if viewer != None:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    test_obj.assertTrue(plan.satisfied())

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
