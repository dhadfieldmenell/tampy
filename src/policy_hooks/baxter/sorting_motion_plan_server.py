import copy
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

import rospy
from std_msgs.msg import *

from policy_hooks.abstract_motion_plan_server import AbstractMotionPlanServer
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.baxter.fold_prob import *
from policy_hooks.baxter.baxter_policy_solver import BaxterPolicySolver

from tamp_ros.msg import *
from tamp_ros.srv import *


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class SortingMotionPlanServer(AbstractMotionPlanServer):
    def __init__(self, hyperparams):
        self.solver = BaxterPolicySolver(hyperparams)
        super(SortingMotionPlanServer, self).__init__(hyperparams)
