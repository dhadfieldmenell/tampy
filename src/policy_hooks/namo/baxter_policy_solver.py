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

from core.util_classes.baxter_predicates import ATTRMAP
from pma.robot_ll_solver import RobotLLSolver
# from policy_hooks.namo.multi_task_main import GPSMain
from policy_hooks.namo.vector_include import *
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
# import policy_hooks.namo.namo_hyperparams as namo_hyperparams
# import policy_hooks.namo.namo_optgps_hyperparams as namo_hyperparams
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.baxter.fold_prob import *
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

BASE_CLASS = get_base_solver(RobotLLSolver)

class BaxterPolicySolver(BASE_CLASS):
    def _fill_sample(self, xxx_todo_changeme, start_t, end_t, plan):
        (i, j) = xxx_todo_changeme
        T = end_t - start_t + 1
        self.agent.T = T
        sample = self.fill_sample((i, j, k), start_t, end_t)
        sample = Sample(self.agent)
        state = np.zeros((self.symbolic_bound, T))
        act = np.zeros((self.dU, T))
        for p_name, a_name in self.state_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            state[self.state_inds[p_name, a_name], :] = getattr(p, a_name)[:, start_t:end_t+1]
        for p_name, a_name in self.action_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            x1 = getattr(p, a_name)[:, start_t:end_t]
            x2 = getattr(p, a_name)[:, start_t+1:end_t+1]
            act[self.action_inds[p_name, a_name], :-1] = x2 - x1
        hist_len = self.agent.hist_len
        target_vec = np.zeros(self.agent.target_vecs[0].shape)
        for p_name, a_name in self.target_inds:
            param = plan.params[p_name]
            target_vec[self.target_inds[p_name, a_name]] = getattr(param, a_name).flatten()
        for t in range(start_t, end_t+1):
            sample.set(STATE_ENUM, state[:, t-start_t], t-start_t)
            task_vec = np.zeros((len(self.agent.task_list)))
            task_vec[i] = 1
            obj_vec = np.zeros((len(self.agent.obj_list)))
            obj_vec[j] = 1
            targ_vec = np.zeros((len(self.agent.targ_list)))
            targ_vec[k] = 1
            traj_hist = np.zeros((hist_len, self.dU))
            for sub_t in range(t-hist_len, t):
                if sub_t < start_t:
                    continue
                traj_hist[sub_t-t+hist_len, :] = act[:, sub_t-start_t]
            sample.set(TASK_ENUM, task_vec, t-start_t)
            sample.set(OBJ_ENUM, obj_vec, t-start_t)
            sample.set(TARG_ENUM, targ_vec, t-start_t)
            ee_pose = state[:, t-start_t][plan.state_inds[self.robot_name, 'pose']]
            obj_pose = state[:, t-start_t][plan.state_inds[self.agent.obj_list[j], 'pose']] - ee_pose
            targ_pose = plan.params[self.agent.targ_list[k]].value[:,0] - ee_pose
            sample.set(EE_ENUM, ee_pose, t-start_t)
            sample.set(OBJ_POSE_ENUM, obj_pose, t-start_t)
            sample.set(TARG_POSE_ENUM, targ_pose, t-start_t)
            sample.set(TRAJ_HIST_ENUM, traj_hist.flatten(), t-start_t)
            sample.set(TARGETS_ENUM, target_vec, t-start_t)
            sample.set(ACTION_ENUM, act[:, t-start_t], t-start_t)
            if LIDAR_ENUM in self.agent._hyperparams['obs_include']:
                lidar = self.agent.dist_obs(plan, t)
                sample.set(LIDAR_ENUM, lidar.flatten(), t-start_t)
        return sample
