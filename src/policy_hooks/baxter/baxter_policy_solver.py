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
from policy_hooks.baxter.multi_task_main import GPSMain
from policy_hooks.baxter.vector_include import *
from policy_hooks.utils.load_task_definitions import *
from policy_hooks.multi_head_policy_opt_tf import MultiHeadPolicyOptTf
from policy_hooks.baxter.baxter_agent import BaxterSortingAgent
import policy_hooks.baxter.baxter_hyperparams as baxter_hyperparams
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.baxter.pick_place_prob import *
from policy_hooks.task_net import tf_classification_network
from policy_hooks.mcts import MCTS
from policy_hooks.state_traj_cost import StateTrajCost
from policy_hooks.action_traj_cost import ActionTrajCost
from policy_hooks.traj_constr_cost import TrajConstrCost
from policy_hooks.cost_product import CostProduct
from policy_hooks.sample import Sample
from policy_hooks.policy_solver import get_base_solver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

N_RESAMPLES = 5
MAX_PRIORITY = 3
DEBUG=False


# Dynamically determine the original MP solver to put the policy code on top of
BASE_CLASS = get_base_solver(RobotLLSolver)

class BaxterPolicySolver(BASE_CLASS):
    def _fill_sample(self, (i, j), start_t, end_t, plan):
        T = end_t - start_t + 1
        self.agent.T = T
        sample = Sample(self.agent)
        state = np.zeros((self.symbolic_bound, T))
        act = np.zeros((self.dU, T))
        robot = plan.params[self.robot_name]
        robot_body  = robot.openrave_body

        for p_name, a_name in self.state_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            state[self.state_inds[p_name, a_name], :] = getattr(p, a_name)[:, start_t:end_t+1]

        for p_name, a_name in self.action_inds:
            p = plan.params[p_name]
            if p.is_symbol(): continue
            if a_name in ['lArmPose', 'lGripper', 'rArmPose', 'rGripper']:
                x1 = getattr(p, a_name)[:, start_t:end_t]
                x2 = getattr(p, a_name)[:, start_t+1:end_t+1]
                act[self.action_inds[p_name, a_name], :-1] = x2 - x1
            elif a_name == 'ee_pose':
                ee_data1 = robot_body.param_fwd_kinemtics(robot, ['right_gripper', 'left_gripper'], start_t)
                ee_pose1 = np.r_[ee_data1['right_gripper']['pos'],
                                 ee_data1['right_gripper']['quat'],
                                 ee_data1['left_gripper']['pos'],
                                 ee_data1['left_gripper']['quat']]
                for t in range(start_t+1, end_t+1)
                    ee_data2 = robot_body.param_fwd_kinemtics(robot, ['right_gripper', 'left_gripper'], t)
                    ee_pose2 = np.r_[ee_data2['right_gripper']['pos'],
                                     ee_data2['right_gripper']['quat'],
                                     ee_data2['left_gripper']['pos'],
                                     ee_data2['left_gripper']['quat']]

                    act[self.action_inds[p_name, a_name], t-1] = ee_pose2 - ee_pose1
                    ee_data1 = ee_data2
                    ee_pose1 = ee_pose2


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

            ee_data = robot_body.param_fwd_kinemtics(robot, ['right_gripper', 'left_gripper'], t)
            ee_pose = np.r_[ee_data['right_gripper']['pos'],
                            ee_data['right_gripper']['quat'],
                            ee_data['left_gripper']['pos'],
                            ee_data['left_gripper']['quat']]

            obj_right_pose = state[:, t-start_t][plan.state_inds[self.agent.obj_list[j], 'pose']] - ee_pose[:3]
            targ_right_pose = plan.params[self.agent.targ_list[k]].value[:,0] - ee_pose[:3]

            obj_left_pose = state[:, t-start_t][plan.state_inds[self.agent.obj_list[j], 'pose']] - ee_pose[7:10]
            targ_left_pose = plan.params[self.agent.targ_list[k]].value[:,0] - ee_pose[7:10]

            sample.set(EE_ENUM, ee_pose, t-start_t)
            sample.set(OBJ_POSE_ENUM, np.r_[obj_right_pose, obj_left_pose], t-start_t)
            sample.set(TARG_POSE_ENUM, np.r_[targ_left_pose, targ_right_pose], t-start_t)
            sample.set(TRAJ_HIST_ENUM, traj_hist.flatten(), t-start_t)
            sample.set(TARGETS_ENUM, target_vec, t-start_t)
            sample.set(ACTION_ENUM, act[:, t-start_t], t-start_t)

        return sample


    def convert_ee(self, mu, sig, manip_name, param, t, arm_joints, attr='pos', use_cov=False):
        '''
        Convert mu and sig for the end effector displacement to mu and sig for joint displacement
        Use jacobian as heuristic for conversion
        '''
        robot_pos = param.openrave_body.param_fwd_kinemtics(param, [manip_name], t+active_ts[0])[manip_name][attr]
        next_robot_pos = param.openrave_body.param_fwd_kinemtics(param, [manip_name], t+active_ts[0]+1)[manip_name][attr]
        jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        new_mu = (next_robot_pos - robot_pose - attr_mu[t]).dot(jac)

        if use_cov:
            new_sig = jac.T.dot(sig.dot(jac))
            tol = 1e-4
            if np.linalg.matrix_rank(new_sig, tol=tol) < len(mu):
                new_sig += tol * np.eye(len(mu))
        else:
            new_sig_diag = jac.dot(np.sqrt(np.diag(sig)))**2
            new_sig_diag[np.abs(new_sig_diag) < 1e-4] = 1e-4
            new_sig = np.diag(new_sig_diag)

        return new_mu.flatten(), new_sig


    def convert_attrs(self, attr_name, attr_mu, attr_sig, param, active_ts, sample):
        if attr_name == 'ee_left_pos':
            ee_pos = param.openrave_body.param_fwd_kinemtics(param, ['left_gripper'], active_ts[0])
            start_val = ee_pos['left_gripper']['pos']
            new_attr = 'lArmPose'

            new_mu = np.zeros((attr_mu.shape[0], 7))
            new_sig = np.zeros((attr_sig.shape[0], 7, 7))
            arm_joints = [param.openrave_body.GetJointFromDOFIndex(ind) for ind in range(2,9)]
            for t in range(active_ts[1]-active_ts[0]):
                new_mu[t] = param.lArmPose[:, t+active_ts[0]+1]
                new_mu[t], new_sig[t] += self.convert_ee(attr_mu[t], attr_sig[t], 'left_gripper', param, t, arm_joints, 'pos')

        elif attr_name == 'ee_right_pos':
            ee_pos = param.openrave_body.param_fwd_kinemtics(param, ['right_gripper'], active_ts[0])
            start_val = ee_pos['right_gripper']['pos']
            new_attr = 'rArmPose'
            abs_mu = attr_mu.copy()
            abs_mu[0] += start_val
            for t in range(len(attr_mu), 0, -1):
                attr_mu[t-1] = np.sum(attr_mu[:t-1], axis=0)

            new_mu = np.zeros((attr_mu.shape[0], 7))
            new_sig = np.zeros((attr_sig.shape[0], 7, 7))
            arm_joints = [param.openrave_body.GetJointFromDOFIndex(ind) for ind in range(2,9)]
            for t in range(active_ts[1]-active_ts[0]):
                new_mu[t] = param.rArmPose[:, t]
                new_mu[t], new_sig[t] += self.convert_ee(attr_mu[t], attr_sig[t], 'right_gripper', param, t, arm_joints, 'pos')
        else:
            raise NotImplementedError

        return new_mu, new_sig
