import sys
import unittest
import time
import main
import numpy as np
from ros_interface import action_execution
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from openravepy import Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
from core.util_classes import baxter_constants

class TestActionExecute(unittest.TestCase):

    def test_execute(self):
        # import ipdb; ipdb.set_trace()
        pass
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5("plan.hdf5")
        # baxter = plan.params['baxter']
        # natural_state = np.array([0., 1.42, 0., 0.02, 0., 0.22, -0.]).reshape((7, 1))
        # natural_traj = np.repeat(natural_state, 40, axis=1)
        # baxter.lArmPose = natural_traj
        # action = plan.actions[0]
        # action_execution.execute_action(action)
