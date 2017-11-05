from ros_interface.controllers import EEController, TrajectoryController
from ros_interface.environment_monitor import EnvironmentMonitor

import rospy

import numpy as np


class PlanExecution(object):
    def __init__(self, plan):
        self.current_ts = 0
        self.plan = plan
        self.ee_control = EEController()
        self.traj_control = TrajectoryController()
        self.env_monitor = EnvironmentMonitor()

    def update_plan(self, plan, start_ts=0):
        self.current_ts = start_ts
        self.plan = plan

    def execute_plan(self):
        cur_action = self.plan.actions.filter(lambda a: a.active_timesteps[0] == self.current_ts)[0]
        active = True
        while (active)
            if cur_action.name == "center_over_basket":
                pass
            elif cur_action.name == "center_over_cloth":
                pass
            elif cur_action.name == "center_over_washer_handle":
                pass
            else:
                pass
