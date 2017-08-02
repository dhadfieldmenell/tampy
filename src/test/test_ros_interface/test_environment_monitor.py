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
        env_m.basket_pose = [-0.05, -0.25, 1.57]
        env_m.cloth_pose = [0.07, 0.15, 1.57]
        env_m.update_plan(plan, 0, ['cloth'], read_file=False)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) > 0)
        result = solver._backtrack_solve(plan, callback=lambda a:None, anum=0, amax=3)
        result = solver.traj_smoother(plan, callback=None)
        self.assertTrue(result)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) == 0)
        env_m.update_plan(plan, 0, ['basket'], read_file=False)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) > 0)
        result = solver._backtrack_solve(plan, callback=lambda a:None, anum=0, amax=5)
        result = solver.traj_smoother(plan, callback=None)
        self.assertTrue(result)
        self.assertTrue(len(plan.get_failed_preds(tol=1e-3)) == 0)
        import ipdb; ipdb.set_trace()
