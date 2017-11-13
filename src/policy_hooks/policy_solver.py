import os

import numpy as np

from gps.gps_main import GPSMain
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network

from  pma.robot_ll_solver import RobotLLSolver
from policy_hooks.cloth_world_policy_utils import *
import policy_hooks.policy_hyperparams as baxter_hyperparams
import policy_hooks.policy_solver_utils as utils
from policy_hooks.tamp_agent import LaundryWorldMujocoAgent
from policy_hooks.tamp_cost import TAMPCost


BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

class BaxterPolicySolver(RobotLLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        self.config = None
        self.gps = None
        self.policy_transfer_coeff = 5e1
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)

    # TODO: Add hooks for online policy learning
    def train_policy(self, num_cloths, hyperparams=None):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must have the same state dimension and action diemensions as the others, and equivalent parameters in both (e..g same # of 
        cloths, same table dimensions, etc.)
        '''

        is_first_run = not self.config
        if is_first_run:
            self.config = baxter_hyperparams.config if not hyperparams else hyperparams

        if hyperparams and self.config:
            self.config.update(hyperparams)

        initial_plan = generate_cond(num_cloths)
        initial_plan.time = np.ones((initial_plan.horizon,))
        initial_plan.dX, initial_plan.state_inds, initial_plan.dU, initial_plan.action_inds, initial_plan.symbolic_bound = utils.get_plan_to_policy_mapping(initial_plan, u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
        x0s = []
        for c in range(self.config['num_conds']):
            x0s.append(get_randomized_initial_state(initial_plan))

        sensor_dims = {
            utils.STATE_ENUM: initial_plan.symbolic_bound,
            utils.ACTION_ENUM: initial_plan.dU,
            utils.OBS_ENUM: initial_plan.symbolic_bound
        }

        if is_first_run:
            self.config['agent'] = {
                'type': LaundryWorldMujocoAgent,
                'x0s': x0s,
                'x0': map(lambda x: x[0][:initial_plan.symbolic_bound], x0s),
                'plan': initial_plan,
                'sensor_dims': sensor_dims,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [utils.OBS_ENUM],
                'conditions': len(x0s),
                'dX': initial_plan.symbolic_bound,
                'dU': initial_plan.dU,
                'demonstrations': 5,
                'expert_ratio': 0.75,
                'solver': self,
                # 'T': initial_plan.horizon - 1
                'T': (initial_plan.horizon - 1) * 200
            }
            self.config['algorithm']['cost'] = []

        else:
            # TODO: Fill in this case
            self.config['agent']['conditions'] += len(x0s)
            self.config['agent']['x0'].extend(x0s)
        
        for cond in range(len(x0s)):
            self.config['algorithm']['cost'].append({
                'type': TAMPCost,
                'plan': initial_plan,
                'dX': initial_plan.symbolic_bound,
                'dU': initial_plan.dU,
                'x0': x0s[cond]
        })

        self.config['dQ'] = initial_plan.dU
        self.config['algorithm']['init_traj_distr']['dQ'] = initial_plan.dU
        self.config['algorithm']['init_traj_distr']['init_gains'] = np.ones((initial_plan.dU)) * 500
        self.config['algorithm']['init_traj_distr']['init_acc'] = np.zeros((sensor_dims[utils.ACTION_ENUM],))
        self.config['algorithm']['init_traj_distr']['dt'] = 0.005
        self.config['algorithm']['init_traj_distr']['T'] = self.config['agent']['T']

        self.config['algorithm']['policy_opt'] = {
            'type': PolicyOptTf,
            'network_params': {
                'obs_include': [utils.STATE_ENUM],
                'obs_vector_data': [utils.STATE_ENUM],
                'sensor_dims': sensor_dims,
            },
            'network_model': tf_network,
            'iterations': 2000,
            'weights_file_prefix': EXP_DIR + 'policy',
        }

        if not self.gps:
            self.gps = GPSMain(self.config)
        else:
            # TODO: Handle this case
            self._update_agent(x0s)
            self._update_algorithm(self.config['algorithm']['cost'][-len(x0s):])
        self.center_trajectories_around_demonstrations()
        self.gps.run()

    def center_trajectories_around_demonstrations(self):
        alg = self.gps.algorithm
        agent = self.gps.agent
        agent.initial_samples = True
        for m in range(alg.M):
            traj_distr = alg.cur[m].traj_distr
            traj_sample = agent.sample(traj_distr, m, on_policy=False)
            k = np.zeros((traj_distr.T, traj_distr.dU))
            for t in range(traj_distr.T):
                k[t] = traj_sample.get_U(t)
            traj_distr.k = k
        agent.initial_samples = False

    def center_trajectory_around_demonstration(self, alg, agent, condition):
        traj_distr = alg.cur[condition].traj_distr
        traj_sample = agent.sample(traj_distr, condition)
        k = np.zeros((traj_distr.T, traj_distr.dU))
        for t in range(traj_distr.T):
            k[t] = traj_sample.get_U(t)
        traj_distr.k = k

    # def _update_algorithm(self, plans, costs):
    #     if not self.gps: return
    #     alg = self.gps.algorithm
    #     alg.M += len(plans)
    #     alg._cond_idx = range(alg.M)
    #     alg._hyperparams['train_conditions'] = alg._cond_idx
    #     alg._hyperparams['test_conditions'] = alg._cond_idx

    #     # IterationData objects for each condition.
    #     alg.cur.extend([IterationData() for _ in range(len(plans))])
    #     alg.prev.extend([IterationData() for _ in range(len(plans))])

    #     init_traj_distr = alg._hyperparams['init_traj_distr']
    #     assert len(self.gps.agent.x0) == alg.M
    #     init_traj_distr['x0'] = self.gps.agent.x0
    #     init_traj_distr['dX'] = self.gps.agent.dX
    #     init_traj_distr['dU'] = self.gps.agent.dU

    #     for m in range(alg.M-len(plans), alg.M):
    #         alg.cur[m].traj_info = TrajectoryInfo()
    #         if alg._hyperparams['fit_dynamics']:
    #             alg.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
    #         alg = extract_condition(
    #             alg._hyperparams['init_traj_distr'], alg._cond_idx[m]
    #         )
    #         alg.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

    #     alg.cost.extend([
    #         costs[i]['type'](costs[i])
    #         for i in range(len(plans))
    #     ])

    # def _update_agent(self, plans, x0):
    #     if not self.gps: return
    #     agent = self.gps.agent
    #     agent._samples.extend([[] for _ in range(self._hyperparams['conditions'])])
    #     agent.x0.extend(x0)
    #     agent.conditions += len(plans)
    #     agent.plans.extend(plans)

    def _solve_opt_prob(self, plan, priority, callback=None, init=True, active_ts=None, verbose=False, resample=False, smoothing=False):
        if self.gps.agent.initial_samples:
            if priority == 3:
                policy = self.gps.algorithm.policy_opt
                pol_sample = self.agent.sample_joint_trajectory_policy(policy, self.gps.agent.current_condition, np.zeros((plan.horizon, plan.dU)))
                traj_state = np.zeros((plan.dX, plan.horizon))
                for t in range(plan.horizon-1):
                    traj_state[:, t] = pol_sample.get_X(t)
                traj_state[:,plan.horizon-1] = pol_sample.get_X(plan.horizon-1)
                obj_bexprs = self._traj_policy_opt(plan, traj_state)
                self._add_obj_bexprs(obj_bexprs)

        # if not self.gps.agent.initial_samples:
        #     if priority == 3:
        #         traj_distr = self.gps.algorithm.cur[self.gps.agent.current_condition].traj_distr
        #         obj_bexprs = self._traj_policy_opt(plan, traj_distr.k)
        #         self._add_obj_bexprs(obj_bexprs)

    def _traj_policy_opt(self, plan, traj_mean):
        transfer_objs = []
        for param_name, attr_name in plan.action_inds.keys():
            param = plan.params[param_name]
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            T = param_ll._horizon
            attr_val = traj_mean[:, plan.action_inds[(param_name, attr_name)]].T
            K = attr_type.dim

            # pose = param.pose
            if DEBUG: assert (K, T) == attr_val.shape
            KT = K*T
            v = -1 * np.ones((KT - K, 1))
            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
            # [:,0] allows numpy to see v and d as one-dimensional so
            # that numpy will create a diagonal matrix with v and d as a diagonal
            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
            # P = np.eye(KT)
            Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
            cur_val = attr_val.reshape((KT, 1), order='F')
            A = -2*cur_val.T.dot(Q)
            b = cur_val.T.dot(Q.dot(cur_val))
            policy_transfer_coeff = self.policy_transfer_coeff/float(plan.T)

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(2*transfer_coeff*Q,
                                 transfer_coeff*A, transfer_coeff*b)
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)
        return transfer_objs

if __name__ == '__main__':
    PS = BaxterPolicySolver()
    PS.train_policy(1)
