import unittest
from core.internal_repr import parameter
from core.util_classes import pr2_predicates, viewer, matrix
from core.util_classes.items import BlueCan, RedCan
from core.util_classes.robots import PR2
from core.util_classes.viewer import OpenRAVEViewer
from pma.can_solver import CanSolver
import time
from openravepy import Environment
from core.internal_repr.action import Action
from core.internal_repr.plan import Plan
import numpy as np

class TestPR2InGripper(unittest.TestCase):
    pass
#     def setup_environment(self):
#         return Environment()
#
#     def setup_robot(self, name = "pr2"):
#         attrs = {"name": [name], "pose": [(0, 0, 0)], "_type": ["Robot"], "geom": [], "backHeight": [0.2], "lGripper": [0.5], "rGripper": [0.5]}
#         attrs["lArmPose"] = [(0,0,0,0,0,0,0)]
#         attrs["rArmPose"] = [(0,0,0,0,0,0,0)]
#         attr_types = {"name": str, "pose": matrix.Vector3d, "_type": str, "geom": PR2, "backHeight": matrix.Value, "lArmPose": matrix.Vector7d, "rArmPose": matrix.Vector7d, "lGripper": matrix.Value, "rGripper": matrix.Value}
#         robot = parameter.Object(attrs, attr_types)
#         # Set the initial arm pose so that pose is not close to joint limit
#         R_ARM_INIT = [-1.832, -0.332, -1.011, -1.437, -1.1, 0, -3.074]
#         L_ARM_INIT = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09]
#         robot.rArmPose = np.array(R_ARM_INIT).reshape((7,1))
#         robot.lArmPose = np.array(L_ARM_INIT).reshape((7,1))
#         return robot
#
#     def setup_can(self, name = "can"):
#         attrs = {"name": [name], "geom": (0.04, 0.25), "pose": ["undefined"], "rotation": [(0, 0, 0)], "_type": ["Can"]}
#         attr_types = {"name": str, "geom": BlueCan, "pose": matrix.Vector3d, "rotation": matrix.Vector3d, "_type": str}
#         can = parameter.Object(attrs, attr_types)
#         return can
#
#     def test_in_gripper_optimization2(self):
#         robot = self.setup_robot()
#         can = self.setup_can()
#         test_env = self.setup_environment()
#         in_gripper = pr2_predicates.InGripper("InGripper", [robot, can], ["Robot", "Can"], test_env)
#         in_gripper_dict = {"negated": False, "active_timesteps": (0,0), "hl_info": "pre", "pred": in_gripper}
#         within_joint_limit = pr2_predicates.WithinJointLimit("WithinJointLimit", [robot], ["Robot"], test_env)
#         within_joint_limit_dict = {"negated": False, "active_timesteps": (0,0), "hl_info": "pre", "pred": within_joint_limit}
#         robot.copy(1)
#
#         robot.pose = np.array([np.NaN, np.NaN, np.NaN]).reshape((3,1))
#         robot.backHeight = np.array([ 0.00111811]).reshape((1,1))
#         # robot.backHeight = np.array([np.NaN]).reshape((1,1))
#         robot.rGripper = np.array([np.NaN]).reshape((1,1))
#         robot.rArmPose = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]).reshape((7,1))
#
#         # pose = np.empty((3,1))
#         # pose[:] = np.NaN
#         # can.pose = pose
#         # rotation = np.empty((3,1))
#         # rotation[:] = np.NaN
#         # can.rotation = rotation
#         can.pose = np.array([-0.31622543, -0.38142561,  1.19321209]).reshape((3,1))
#         can.rotation = np.array([ 0.04588155, -0.38504402,  0.19207589]).reshape((3,1))
#
#         params = [robot, can]
#         dummy_action = Action(1, 'dummy', (0,0), params, [in_gripper_dict, within_joint_limit_dict])
#         param_dict = {}
#         for param in params:
#             param_dict[param.name] = param
#         dummy_plan = Plan(param_dict, [dummy_action], 1, test_env)
#
#         robot.pose = np.array([-0.57236661,  0.34939761,  0.02360263]).reshape((3,1))
#         robot.rArmPose = np.array([-1.832, -0.332, -1.011, -1.437, -1.1  ,  0.   , -3.074]).reshape((7,1))
#         robot.rGripper = np.array([0.5]).reshape((1,1))
#
#         can.pose = np.array([-0.31622543, -0.38142561,  1.19321209]).reshape((3,1))
#         can.rotation = np.array([ 0.04588155, -0.38504402,  0.19207589]).reshape((3,1))
#
#         _test_plan(self, dummy_plan)
#
#     def test_in_gripper_optimization(self):
#         robot = self.setup_robot()
#         can = self.setup_can()
#         test_env = self.setup_environment()
#         in_gripper = pr2_predicates.InGripper("InGripper", [robot, can], ["Robot", "Can"], test_env)
#         in_gripper_dict = {"negated": False, "active_timesteps": (0,0), "hl_info": "pre", "pred": in_gripper}
#         within_joint_limit = pr2_predicates.WithinJointLimit("WithinJointLimit", [robot], ["Robot"], test_env)
#         within_joint_limit_dict = {"negated": False, "active_timesteps": (0,0), "hl_info": "pre", "pred": within_joint_limit}
#         robot.copy(1)
#         robot.pose = np.array([-.5, 0, 0]).reshape((3,1))
#         robot.backHeight = np.array([0.3]).reshape((1,1))
#         robot.rGripper = np.array([np.NaN]).reshape((1,1))
#         robot.rArmPose = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]).reshape((7,1))
#         # robot.lGripper = np.array([np.NaN]).reshape((1,1))
#         # robot.lArmPose = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]).reshape((7,1))
#         # pose = np.empty((3,1))
#         # pose[:] = np.NaN
#         # can.pose = pose
#         # can.rotation =
#         can.pose = np.array([-0., -0.08297436,  0.925]).reshape((3,1))
#         # can.pose = np.array([-0., -.15,  0.925]).reshape((3,1))
#         # can.pose = np.array([-0., -.25,  0.925]).reshape((3,1))
#         # can.pose = np.array([-0., -.5,  0.925]).reshape((3,1))
#         can.rotation = np.array([-1., -0., -0.]).reshape((3,1))
#
#         params = [robot, can]
#         dummy_action = Action(1, 'dummy', (0,0), params, [in_gripper_dict, within_joint_limit_dict])
#         param_dict = {}
#         for param in params:
#             param_dict[param.name] = param
#         dummy_plan = Plan(param_dict, [dummy_action], 1, test_env)
#         _test_plan(self, dummy_plan)
#
# def _test_plan(test_obj, plan):
#     print "testing plan: {}".format(plan.actions)
#     callback = None
#     viewer = None
#     """
#     Uncomment out lines below to see optimization.
#     """
#     viewer = OpenRAVEViewer.create_viewer()
#     def callback():
#         solver._update_ll_params()
#     #     obj_list = viewer._get_plan_obj_list(plan)
#     #     # viewer.draw_traj(obj_list, [0,9,19,20,29,38])
#     #     # viewer.draw_traj(obj_list, range(19,30))
#     #     # viewer.draw_traj(obj_list, range(29,39))
#     #     # viewer.draw_traj(obj_list, [38])
#     #     # viewer.draw_traj(obj_list, range(19,39))
#     #     # viewer.draw_plan_range(plan, [0,19,38])
#         viewer.draw_plan(plan)
#         time.sleep(0.03)
#     """
#     """
#     can_solver = CanSolver()
#     can_solver.solve(plan, callback=callback, n_resamples=0)
#
#     fp = plan.get_failed_preds()
#     _, _, t = plan.get_failed_pred()
#     #
#     if viewer != None:
#         viewer = OpenRAVEViewer.create_viewer()
#         viewer.animate_plan(plan)
#         if t < plan.horizon:
#             viewer.draw_plan_ts(plan, t)
#
#     import ipdb; ipdb.set_trace()
#     test_obj.assertTrue(plan.satisfied(1e-4))
