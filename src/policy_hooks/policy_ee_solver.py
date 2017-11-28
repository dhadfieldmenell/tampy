import os

import numpy as np

import gurobipy as grb

from sco.expr import BoundExpr, QuadExpr, AffExpr
from sco.prob import Prob
from sco.solver import Solver

from gps.gps_main import GPSMain
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network

from  pma.robot_ll_solver import RobotLLSolver
from policy_hooks.cloth_world_policy_utils import *
import policy_hooks.policy_hyperparams as baxter_hyperparams
import policy_hooks.policy_solver_utils as utils
from policy_hooks.tamp_ee_cloth_agent import LaundryWorldEEAgent
from policy_hooks.tamp_cost import TAMPCost


BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'


class BaxterPolicyEESolver(RobotLLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        self.config = None
        self.gps = None
        super(BaxterPolicyEESolver, self).__init__(early_converge, transfer_norm)

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

        # initial_plan.dX, initial_plan.state_inds, initial_plan.dU, \
        # initial_plan.action_inds, initial_plan.symbolic_bound = \
        # utils.get_plan_to_policy_mapping(initial_plan, 
        #                                  x_params=['baxter', 'cloth_0', 'cloth_1', 'cloth_2', 'cloth_3', 'basket'], 
        #                                  x_attrs=['pose'], 
        #                                  u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
        initial_plan.dX, initial_plan.state_inds, initial_plan.dU, \
        initial_plan.action_inds, initial_plan.symbolic_bound = \
        utils.get_plan_to_policy_mapping(initial_plan, 
                                         x_params=['baxter', 'cloth_0', 'cloth_1', 'basket'], 
                                         x_attrs=['pose', 'lArmPose', 'rArmPose', 'rGripper'], 
                                         u_attrs=set(['ee_left_pos', 'lGripper']))
        
        x0s = []
        # for c in range(self.config['num_conds']):
        #     x0s.append(get_randomized_initial_state(initial_plan))
        for c in range(0, self.config['num_conds'], num_cloths):
            x0s.extend(get_randomized_initial_state_multi_step(initial_plan, c/num_cloths))

        sensor_dims = {
            utils.STATE_ENUM: initial_plan.symbolic_bound,
            utils.ACTION_ENUM: initial_plan.dU,
            utils.OBS_ENUM: initial_plan.symbolic_bound
        }

        self.T = initial_plan.actions[x0s[0][1][1]].active_timesteps[1] - initial_plan.actions[x0s[0][1][0]].active_timesteps[0]

        if is_first_run:
            self.config['agent'] = {
                'type': LaundryWorldEEAgent,
                'x0s': x0s,
                'x0': map(lambda x: x[0][:initial_plan.symbolic_bound], x0s),
                'plan': initial_plan,
                'sensor_dims': sensor_dims,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [utils.OBS_ENUM],
                'conditions': self.config['num_conds'],
                'dX': initial_plan.symbolic_bound,
                'dU': initial_plan.dU,
                'demonstrations': 5,
                'expert_ratio': 0.75,
                'solver': self,
                'num_cloths': num_cloths,
                # 'T': initial_plan.horizon - 1
                'T': self.T * utils.MUJOCO_STEPS_PER_SECOND,
                'stochastic_conditions': self.config['algorithm']['stochastic_conditions']
            }
            self.config['algorithm']['cost'] = []

        else:
            # TODO: Fill in this case
            self.config['agent']['conditions'] += self.config['num_conds']
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
        # self.config['algorithm']['init_traj_distr']['init_gains'] = np.ones((initial_plan.dU)) * 500
        # self.config['algorithm']['init_traj_distr']['init_acc'] = np.zeros((sensor_dims[utils.ACTION_ENUM],))
        self.config['algorithm']['init_traj_distr']['dt'] = 0.005
        self.config['algorithm']['init_traj_distr']['T'] = self.config['agent']['T']

        self.config['algorithm']['policy_opt'] = {
            'type': PolicyOptTf,
            'network_params': {
                'obs_include': [utils.STATE_ENUM],
                'obs_vector_data': [utils.STATE_ENUM],
                'sensor_dims': sensor_dims,
                'n_layers': 2,
                'dim_hidden': [200, 200]
            },
            'lr': 1e-5,
            'network_model': tf_network,
            'iterations': 200000,
            'weight_decay': 0.01,
            'weights_file_prefix': EXP_DIR + 'policy',
        }

        if not self.gps:
            self.gps = GPSMain(self.config)
        else:
            # TODO: Handle this case
            self._update_agent(x0s)
            self._update_algorithm(self.config['algorithm']['cost'][-len(x0s):])
        
        self.gps.run()

    # def center_trajectories_around_demonstrations(self):
    #     alg = self.gps.algorithm
    #     agent = self.gps.agent
    #     # agent.initial_samples = True
    #     for m in range(alg.M):
    #         traj_distr = alg.cur[m].traj_distr
    #         traj_sample = agent.sample(traj_distr, m, on_policy=False)
    #         k = np.zeros((traj_distr.T, traj_distr.dU))
    #         for t in range(traj_distr.T):
    #             k[t] = traj_sample.get_U(t)
    #         traj_distr.k = k
    #     # agent.initial_samples = False

    # def center_trajectory_around_demonstration(self, alg, agent, condition):
    #     traj_distr = alg.cur[condition].traj_distr
    #     traj_sample = agent.sample(traj_distr, condition)
    #     k = np.zeros((traj_distr.T, traj_distr.dU))
    #     for t in range(traj_distr.T):
    #         k[t] = traj_sample.get_U(t)
    #     traj_distr.k = k

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


    def _backtrack_solve(self, plan, callback=None, anum=0, verbose=False, amax = None):
        if amax is None:
            amax = len(plan.actions) - 1

        if anum > amax:
            return True

        a = plan.actions[anum]
        print "backtracking Solve on {}".format(a.name)
        active_ts = a.active_timesteps
        inits = {}
        if a.name == 'moveto':
            ## find possible values for the final robot_pose
            rs_param = a.params[2]
        elif a.name == 'moveholding_basket':
            ## find possible values for the final robot_pose
            rs_param = a.params[2]
        elif a.name == 'moveholding_cloth':
            ## find possible values for the final robot_pose
            rs_param = a.params[2]
        elif a.name == 'basket_grasp':
            ## find possible ee_poses for both arms
            rs_param = a.params[-1]
        elif a.name == 'basket_putdown':
            ## find possible ee_poses for both arms
            rs_param = a.params[-1]
        elif a.name == 'open_door':
            ## find possible ee_poses for left arms
            rs_param = a.params[-3]
        elif a.name == 'close_door':
            ## find possible ee_poses for left arms
            rs_param = a.params[-3]
        elif a.name == 'cloth_grasp':
            ## find possible ee_poses for right arms
            rs_param = a.params[-1]
        elif a.name == 'cloth_putdown':
            ## find possible ee_poses for right arms
            rs_param = a.params[-1]
        elif a.name =='put_into_washer':
            rs_param = a.params[-1]
        elif a.name =='take_out_of_washer':
            rs_param = a.params[-1]
        elif a.name == 'put_into_basket':
            rs_param = a.params[-1]
        elif a.name == 'push_door':
            rs_param = a.params[-3]
        else:
            raise NotImplemented

        def recursive_solve():
            ## don't optimize over any params that are already set
            old_params_free = {}
            for p in plan.params.itervalues():
                if p.is_symbol():
                    if p not in a.params: continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in old_params_free[p].keys():
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, active_ts[1]].copy()
                        p._free_attrs[attr][:, active_ts[1]] = 0
            self.child_solver = BaxterPolicyEESolver()
            self.child_solver.gps = self.gps
            success = self.child_solver._backtrack_solve(plan, callback=callback, anum=anum+1, verbose=verbose, amax = amax)

            # reset free_attrs
            for p in plan.params.itervalues():
                if p.is_symbol():
                    if p not in a.params: continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, active_ts[1]] = old_params_free[p][attr]
            return success

        # if there is no parameter to resample or some part of rs_param is fixed, then go ahead optimize over this action
        if rs_param is None or sum([not np.all(rs_param._free_attrs[attr]) for attr in rs_param._free_attrs.keys() ]):
            ## this parameter is fixed
            if callback is not None:
                callback_a = lambda: callback(a)
            else:
                callback_a = None
            self.child_solver = BaxterPolicyEESolver()
            self.child_solver.gps = self.gps
            success = self.child_solver.solve(plan, callback=callback_a, n_resamples=10,
                                              active_ts = active_ts, verbose=verbose, force_init=True)

            if not success:
                ## if planning fails we're done
                return False
            ## no other options, so just return here
            return recursive_solve()

        ## so that this won't be optimized over
        rs_free = rs_param._free_attrs
        rs_param._free_attrs = {}
        for attr in rs_free.keys():
            rs_param._free_attrs[attr] = np.zeros(rs_free[attr].shape)

        """
        sampler_begin
        """
        robot_poses = self.obj_pose_suggester(plan, anum, resample_size=20)
        if not robot_poses:
            success = False
            # print "Using Random Poses"
            # robot_poses = self.random_pose_suggester(plan, anum, resample_size = 5)

        """
        sampler end
        """

        if callback is not None:
            callback_a = lambda: callback(a)
        else:
            callback_a = None

        for rp in robot_poses[:5]:
            for attr, val in rp.iteritems():
                setattr(rs_param, attr, val)

            success = False
            self.child_solver = BaxterPolicyEESolver()
            self.child_solver.gps = self.gps
            success = self.child_solver.solve(plan, callback=callback_a, n_resamples=10,
                                              active_ts = active_ts, verbose=verbose,
                                              force_init=True)
            if success:
                if recursive_solve():
                    break
                else:
                    success = False

        rs_param._free_attrs = rs_free
        return success


    def _solve_opt_prob(self, plan, priority, callback=None, init=True, active_ts=None, verbose=False, resample=False, smoothing = False):
        if not self.gps.agent.cond_global_pol_sample[self.gps.agent.current_cond]: # or priority < 3:
            return super(BaxterPolicyEESolver, self)._solve_opt_prob(plan, priority, callback, init, active_ts, verbose, resample, smoothing)

        self.plan = plan
        robot = plan.params['baxter']
        body = plan.env.GetRobot("baxter")
        if active_ts==None:
            active_ts = (0, plan.horizon-1)
        plan.save_free_attrs()
        model = grb.Model()
        model.params.OutputFlag = 0
        self._prob = Prob(model, callback=callback)
        self._spawn_parameter_to_ll_mapping(model, plan, active_ts)
        model.update()
        initial_trust_region_size = self.initial_trust_region_size
        if resample:
            tol = 1e-3
            """
            When Optimization fails, resample new values for certain timesteps
            of the trajectory and solver as initialization
            """
            obj_bexprs = []

            ## this is an objective that places
            ## a high value on matching the resampled values
            failed_preds = plan.get_failed_preds(active_ts = active_ts, priority=priority, tol = tol)
            rs_obj = self._resample(plan, failed_preds, sample_all = True)
            # import ipdb; ipdb.set_trace()
            # _get_transfer_obj returns the expression saying the current trajectory should be close to it's previous trajectory.
            # obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))

            self._add_all_timesteps_of_actions(plan, priority=priority,
                add_nonlin=False, active_ts= active_ts, verbose=verbose)
            obj_bexprs.extend(rs_obj)
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e3
            # import ipdb; ipdb.set_trace()
        else:
            self._bexpr_to_pred = {}
            obj_bexprs = self._get_trajopt_obj(plan, active_ts)
            self._add_obj_bexprs(obj_bexprs)
            self._add_all_timesteps_of_actions(plan, priority=priority, add_nonlin=True,
                                               active_ts=active_ts, verbose=verbose)
            tol=1e-3

        # Constrain optimization against the global policy
        # pol_sample = self.gps.agent.cond_global_pol_sample[self.gps.agent.current_cond]
        # traj_state = np.zeros((plan.symbolic_bound, plan.horizon))
        # for t in range(0,plan.horizon-1):
        #     traj_state[:, t] = pol_sample.get_X(t*utils.MUJOCO_STEPS_PER_SECOND)
        # traj_state[:,plan.horizon-1] = pol_sample.get_X((plan.horizon-1)*utils.MUJOCO_STEPS_PER_SECOND-1)
        # obj_bexprs = self._traj_policy_opt(plan, traj_state)
        # self._add_obj_bexprs(obj_bexprs)

        idx = self.gps.agent.current_cond
        while plan.actions[self.gps.agent.init_plan_states[idx][1][1]].active_timesteps[1] <= active_ts[0]:
            idx += 1
        T = active_ts[1] - active_ts[0] + 1
        pol_sample = self.gps.agent.cond_global_pol_sample[idx]
        traj_state = np.zeros((plan.symbolic_bound, T))
        for t in range(active_ts[0], active_ts[1]):
            pol_sample = self.gps.agent.cond_global_pol_sample[idx]
            traj_state[:, t - active_ts[0]] = pol_sample.get_X(t * utils.MUJOCO_STEPS_PER_SECOND)
        traj_state[:, T - 1] = pol_sample.get_X(T * utils.MUJOCO_STEPS_PER_SECOND - 1)
        obj_bexprs = self._traj_policy_opt(plan, traj_state)
        self._add_obj_bexprs(obj_bexprs)

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size

        if smoothing:
            solv.initial_penalty_coeff = self.smooth_penalty_coeff
        else:
            solv.initial_penalty_coeff = self.init_penalty_coeff

        solv.max_merit_coeff_increases = self.max_merit_coeff_increases

        success = solv.solve(self._prob, method='penalty_sqp', tol=tol, verbose=verbose)
        self._update_ll_params()

        if resample:
            # During resampling phases, there must be changes added to sampling_trace
            if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
                reward = 0
                if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
                    reward = len(plan.actions)
                else:
                    failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
                    for i in range(len(plan.actions)):
                        if failed_t > plan.actions[i].active_timesteps[1]:
                            reward += 1
                plan.sampling_trace[-1]['reward'] = reward
        ##Restore free_attrs values
        plan.restore_free_attrs()

        self.reset_variable()
        print "priority: {}\n".format(priority)
        return success

    def _traj_policy_opt(self, plan, traj_mean):
        transfer_objs = []
        for param_name, attr_name in plan.action_inds.keys():
            param = plan.params[param_name]
            attr_type = param.get_attr_type(attr_name)
            param_ll = self._param_to_ll[param]
            T = param_ll._horizon
            attr_val = traj_mean[plan.action_inds[(param_name, attr_name)], :].T
            K = attr_type.dim

            KT = K*T
            v = -1 * np.ones((KT - K, 1))
            d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
            # [:,0] allows numpy to see v and d as one-dimensional so
            # that numpy will create a diagonal matrix with v and d as a diagonal
            P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
            # P = np.eye(KT)
            Q = np.dot(np.transpose(P), P) if not param.is_symbol() else np.eye(KT)
            cur_val = attr_val.reshape((KT, 1), order='F')
            A = -2 * cur_val.T.dot(Q)
            b = cur_val.T.dot(Q.dot(cur_val))
            policy_transfer_coeff = self.gps.algorithm.policy_transfer_coeff / float(traj_mean.shape[1])

            # QuadExpr is 0.5*x^Tx + Ax + b
            quad_expr = QuadExpr(2*policy_transfer_coeff*Q,
                                 policy_transfer_coeff*A, policy_transfer_coeff*b)
            ll_attr_val = getattr(param_ll, attr_name)
            param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
            sco_var = self.create_variable(param_ll_grb_vars, cur_val)
            bexpr = BoundExpr(quad_expr, sco_var)
            transfer_objs.append(bexpr)
        return transfer_objs

if __name__ == '__main__':
    PS = BaxterPolicyEESolver()
    PS.train_policy(2)
