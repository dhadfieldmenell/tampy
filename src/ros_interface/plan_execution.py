from ros_interface.controllers import EEController, TrajectoryController

import rospy

import numpy as np


class PlanExecution(object):
    def __init__(self):
        self.current_ts = 0
        self.ee_control = EEController()
        self.traj_control = TrajectoryController()
        # TODO: Add wrist camera predictors here

    def update_plan(self, plan, start_ts=0):
        self.current_ts = start_ts
        self.plan = plan

    def execute_plan(self, plan):
        cur_action = self.plan.actions.filter(lambda a: a.active_timesteps[0] == self.current_ts)[0]
        active = True
        while (active)
            if cur_action.name == "center_over_basket":
                pass
            elif cur_action.name == "center_over_cloth":
                pass
            elif cur_action.name == "center_over_washer_handle":
                pass
            elif cur_action.name.startswith("rotate"):
                pass
            else:
                active_ts = cur_action.active_timesteps
                self.traj_control.execute_plan(plan, active_ts=active_ts)
