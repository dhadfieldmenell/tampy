import sys
import unittest
import time
import main
import numpy as np
from ros_interface import action_execution, environment_monitor
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from pma.robot_ll_solver import RobotLLSolver


class TestEnvMonitoring(unittest.TestCase):

    def test_env_monitoring(self): 
        pass

    def test_plan_updating(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('env_mon.hdf5')
        env_m = environment_monitor.EnvironmentMonitor()
        solver = RobotLLSolver()
        env_m.basket_pose = [-0.05, -0.3, 1.57]
        env_m.cloth_pose = [0.07, 0.15, 1.57]
        env_m.update_plan(plan, 0, ['cloth'], read_file=False)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) > 0)
        # solver._solve_opt_prob(plan, priority=-2, callback=None, active_ts=None, verbose=False, resample = False)
        # result = solver.traj_smoother(plan)
        result = solver.solve(plan, active_ts=(0,plan.horizon-1), callback=lambda: None)
        self.assertTrue(result)
        env_m.update_plan(plan, 0, ['basket'], read_file=False)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) > 0)
        solver._solve_opt_prob(plan, priority=-2, callback=None, active_ts=None, verbose=False, resample = False, smoothing = True)
        result = solver.traj_smoother(plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()
