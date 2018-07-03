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
from core.util_classes.items import RedCircle, BlueCircle, GreenCircle
from core.util_classes.matrix import Vector2d
from core.internal_repr import parameter
import time, main
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from Tkinter import *
import tkFileDialog
from PIL import ImageTk, Image

wall_endpoints = [[-1.0,-3.0],[-1.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[8.0,4.0],[8.0,-3.0],[-1.0,-3.0]]


class TestLLSolver(unittest.TestCase):
    def setUp(self):

        domain_fname = '../domains/namo_domain/namo.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        def get_plan(p_fname, plan_str=None):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            if plan_str is not None:
                print "*****************************************************"
                print "calling hls"
                print "*****************************************************"
                return hls.get_plan(plan_str, domain, problem)
            return hls.solve(abs_problem, domain, problem)

        self.move_no_obs = get_plan('../domains/namo_domain/namo_probs/move_no_obs.prob')
        self.move_w_obs = get_plan('../domains/namo_domain/namo_probs/move_w_obs.prob')
        self.move_grasp = get_plan('../domains/namo_domain/namo_probs/move_grasp.prob')
        self.move_grasp_moveholding = get_plan('../domains/namo_domain/namo_probs/moveholding.prob')
        self.place = get_plan('../domains/namo_domain/namo_probs/place.prob')
        self.putaway = get_plan('../domains/namo_domain/namo_probs/putaway.prob')
        self.putaway3 = get_plan('../domains/namo_domain/namo_probs/putaway3.prob')
        self.putaway2 = get_plan('../domains/namo_domain/namo_probs/putaway2.prob', ['0: MOVETO PR2 ROBOT_INIT_POSE PDP_TARGET2',
                                                                                     '1: GRASP PR2 CAN0 TARGET0 PDP_TARGET2 PDP_TARGET0 GRASP0',
                                                                                     '2: MOVETOHOLDING PR2 PDP_TARGET0 PDP_TARGET2 CAN0 GRASP0',
                                                                                     '3: PUTDOWN PR2 CAN0 TARGET2 PDP_TARGET2 ROBOT_END_POSE GRASP0'])
        self.move_two_cans = get_plan('../domains/namo_domain/namo_probs/namo_1234_1.prob')

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
        pr2_ll = ll_solver.LLParam(model, pr2, horizon, (0,horizon-1))
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
        robot_init_pose_ll = ll_solver.LLParam(model, robot_init_pose, horizon, (0, horizon-1))
        robot_init_pose_ll.create_grb_vars()
        self.assertTrue(robot_init_pose_ll.value.shape == (2,1))
        with self.assertRaises(AttributeError):
            pr2_ll._type
        with self.assertRaises(AttributeError):
            pr2_ll.geom

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
        def callback():
            namo_solver._update_ll_params()
            viewer.draw_plan(plan)
            time.sleep(0.1)
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
        viewer.draw_plan(plan)
        import ipdb; ipdb.set_trace()
        # time.sleep(3)

    def test_move_no_obs(self):
        _test_plan_with_learning(self, self.move_no_obs)

    def test_move_w_obs(self):
        _test_plan_with_learning(self, self.move_w_obs)

    def test_move_grasp(self):
        _test_plan_with_learning(self, self.move_grasp)

    def test_moveholding(self):
        _test_plan_with_learning(self, self.move_grasp_moveholding)

    def test_place(self):
        _test_plan_with_learning(self, self.place)

    def test_putaway(self):
        _test_plan_with_learning(self, self.putaway)

    def test_putaway3(self):
        _test_plan_with_learning(self, self.putaway3)

    def test_putaway2(self):
        # this is a plan where the robot needs to end up
        # behind the obstruction (this means that the
        # default initialization should fail
        _test_plan_with_learning(self, self.putaway2, animate=True)


    def test_early_converge(self):
        print "No Early Converge"
        _test_plan(self, self.putaway2, plot=False, animate=False)
        print "Early Converge"
        _test_plan(self, self.putaway2, plot=False, early_converge=True, animate=False)
    def test_backtrack_move(self):
        _test_plan(self, self.move_no_obs, method='Backtrack')

    def test_backtrack_move_grasp(self):
        _test_plan(self, self.move_grasp, method='Backtrack')

    def test_backtrack_moveholding(self):
        _test_plan(self, self.move_grasp_moveholding, method='Backtrack')

    def test_backtrack_putaway(self):
        _test_plan(self, self.putaway, method='Backtrack')

    def test_backtrack_putaway2(self):
        _test_plan(self, self.putaway2, method='Backtrack')

def closet_maker(thickness, wall_endpoints, ax):
    rects = []
    for i, (start, end) in enumerate(zip(wall_endpoints[0:-1], wall_endpoints[1:])):
        dim_x, dim_y = 0, 0
        et = thickness / 2.0
        if start[0] == end[0]: # vertical line
            if start[1] > end[1]: #downwards line
                x1 = (start[0] - et, start[1] + et)
                x2 = (end[0] + et, end[1] - et)
            else:
                x1 = (end[0] - et, end[1] + et)
                x2 = (start[0] + et, start[1] - et)
        elif start[1] == end[1]: # horizontal line
            if start[0] < end[0]: #left to right
                x1 = (start[0] - et, start[1] + et)
                x2 = (end[0] + et, end[1] - et)
            else:
                x1 = (end[0] - et, end[1] + et)
                x2 = (start[0] + et, start[1] - et)
        left_bottom = (x1[0], x2[1])
        width = x2[0] - x1[0]
        height = x1[1] - x2[1]
        rects.append([(left_bottom[0], left_bottom[1]), width, height])
    for rect in rects:
        p = patches.Rectangle(rect[0],rect[1],rect[2], lw=0, color="brown")
        ax.add_patch(p)
    return True

def _test_plan_with_learning(test_obj, plan, method='SQP', plot=True, animate=True, verbose=False,
               early_converge=False):
    print "testing plan: {}".format(plan.actions)
    if not plot:
        callback = None
        viewer = None
    else:
        viewer = OpenRAVEViewer.create_viewer()
        fig, ax = plt.subplots()
        objList = []
        for p in plan.params.itervalues():
            if not p.is_symbol():
                objList.append(p)
        center = 0
        radius = 0
        circColor = None
        for obj in objList:
            if (isinstance(obj.geom, BlueCircle)):
                circColor = 'blue'
            elif (isinstance(obj.geom, GreenCircle)):
                circColor = 'g'
            elif (isinstance(obj.geom, RedCircle)):
                circColor = 'r'
            else:
                print("not a circle; probably a wall")
                continue
            center = obj.pose[:,0]
            radius = obj.geom.radius
            ax.add_artist(plt.Circle((center[0], center[1]), radius, color=circColor))
        closet_maker(1, wall_endpoints, ax)
        ax.set_xlim(-3, 10)
        ax.set_ylim(-5, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis("off")
        fig.canvas.draw()
        root = Tk()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image)
        img = ImageTk.PhotoImage(image)
        '''
        Uncomment below lines for calibration:
        '''
        # path = "calibration.jpg"
        # img = ImageTk.PhotoImage(Image.open(path))
        panel = Label(root, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        def motion(event):
            origin = (241, 307)
            scaling = 25.0
            x, y = event.x, event.y
            X = (x - origin[0])/scaling
            Y =-(y - origin [1])/scaling
            print('{}, {}'.format(X, Y))
            plan.params['pr2'].pose[0][0] = int(X)
            plan.params['pr2'].pose[1][0] = int(Y)
            plan.params['robot_init_pose'].value[0][0] = int(X)
            plan.params['robot_init_pose'].value[1][0] = int(Y)
            root.destroy()
            time.sleep(0.1)
        root.bind('<ButtonRelease-1>', motion)
        root.mainloop()
        # offset_tolerance = [[0.000085], [-0.000085]]
        if method=='SQP':
            def callback():
                namo_solver._update_ll_params()
                # viewer.draw_plan_range(plan, range(57, 77)) # displays putdown action
                # viewer.draw_plan_range(plan, range(38, 77)) # displays moveholding and putdown action
                # viewer.draw_plan_range(plan, [0,19])
                viewer.draw_plan(plan)
                # viewer.draw_cols(plan)
                time.sleep(0.03)
        elif method == 'Backtrack':
            def callback(a):
                namo_solver._update_ll_params()
                viewer.clear()
                viewer.draw_plan_range(plan, a.active_timesteps)
                time.sleep(0.3)
    namo_solver = ll_solver.NAMOSolver(early_converge=early_converge)
    start = time.time()
    #plan to check preds
    #subtract actual position, check which timestep plan fails due to a COLLISION.
    #recall planner if this happens.
    if method == 'SQP':
        namo_solver.solve(plan, callback=callback, verbose=verbose)
    elif method == 'Backtrack':
        namo_solver.backtrack_solve(plan, callback=callback, verbose=verbose)
    print "Solve Took: {}".format(time.time() - start)
    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    if animate:
        viewer = OpenRAVEViewer.create_viewer()
        # import ipdb; ipdb.set_trace()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)
        # pseudocode
        # plan.params['pr2'].pose -= inaccuracy
        # plan.params['robot_init_pose'].value -= inaccuracy
        # for ts in range(plan.horizon):
        #     if plan.get_failed_pred()[ts] and isinstance(collisionPredicate):
        #         replan???
        import ipdb; ipdb.set_trace()

def _test_plan(test_obj, plan, method='SQP', plot=False, animate=False, verbose=False,
               early_converge=False):
    print "testing plan: {}".format(plan.actions)
    if not plot:
        callback = None
        viewer = None
    else:
        viewer = OpenRAVEViewer.create_viewer()
        if method=='SQP':
            def callback():
                namo_solver._update_ll_params()
                # viewer.draw_plan_range(plan, range(57, 77)) # displays putdown action
                # viewer.draw_plan_range(plan, range(38, 77)) # displays moveholding and putdown action
                viewer.draw_plan_range(plan, [0,19])
                # viewer.draw_plan(plan)
                # viewer.draw_cols(plan)
                time.sleep(0.03)
        elif method == 'Backtrack':
            def callback(a):
                namo_solver._update_ll_params()
                viewer.clear()
                viewer.draw_plan_range(plan, a.active_timesteps)
                time.sleep(0.3)
    namo_solver = ll_solver.NAMOSolver(early_converge=early_converge)
    start = time.time()
    if method == 'SQP':
        namo_solver.solve(plan, callback=callback, verbose=verbose)
    elif method == 'Backtrack':
        namo_solver.backtrack_solve(plan, callback=callback, verbose=verbose)
    print "Solve Took: {}".format(time.time() - start)
    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    if animate:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)  

if __name__ == "__main__":
    unittest.main()
