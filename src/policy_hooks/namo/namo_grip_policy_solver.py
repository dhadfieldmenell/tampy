import os

import numpy as np

import gurobipy as grb

from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.prob import Prob
from sco.solver import Solver
from sco.variable import Variable

# from gps.gps_main import GPSMain
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
# from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import *

from core.util_classes.namo_predicates import ATTRMAP
from pma.namo_grip_solver import NAMOSolver
# from policy_hooks.namo.multi_task_main import GPSMain
from policy_hooks.namo.vector_include import *
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.namo.namo_agent import NAMOSortingAgent
# import policy_hooks.namo.namo_hyperparams as namo_hyperparams
# import policy_hooks.namo.namo_optgps_hyperparams as namo_hyperparams
from policy_hooks.namo.namo_policy_predicates import NAMOPolicyPredicate
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.namo.sorting_prob_2 import *
from policy_hooks.task_net import tf_binary_network, tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

# N_RESAMPLES = 5
# MAX_PRIORITY = 3
# DEBUG=False

BASE_CLASS = get_base_solver(NAMOSolver)

class NAMOGripPolicySolver(BASE_CLASS):
    pass
