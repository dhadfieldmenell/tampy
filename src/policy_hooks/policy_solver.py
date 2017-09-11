import numpy as np
from gps.gps_main import GPSMain

from  pma.robot_ll_solver import RobotLLSolver
import policy_hooks.policy_hyperparams as baxter_hyperparams
import policy_hooks.policy_solver_utils as utils
from policy_hooks.tamp_agent import LaundryWorldMujocoAgent


class BaxterPolicySolver(RobotLLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        self.plans = []
        self.config = None
        self.gps = None
        self.policy_transfer_coeff = 1e-1
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)

    # TODO: Add hooks for online policy learning
    def train_policy(self, plans, hyperparams=None):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must have the same state dimension and action diemensions as the others, and equivalent parameters in both (e..g same # of 
        cloths, same table dimensions, etc.)
        '''
        if hyperparams and self.config:
            self.config.update(hyperparams)
        self._plan_to_cond = {}
        cond = 0
        for plan in plans:
            self._plan_to_cond[plan] = cond
            cond += 1
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = utils.get_plan_to_policy_mapping(plan, u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
            plan.T = plan.active_ts[1] - plan.active_ts[0] + 1
        dX, dU = plans[0].dX, plans[0].dU
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            utils.STATE_ENUM: dX,
            utils.ACTION_ENUM: dU
        }

        x0 = np.zeros((len(plans), dX))
        for i in range(len(plans)):
            plan = plans[i]
            utils.fill_vector(plan.actions[0].params, plan.state_inds, x0[i], plan.active_ts[0])
        x0 = xo.tolist()

        if not self.config:
            self.config = baxter_hyperparams.config if not hyperparams else hyperparams
            self.config['agent'] = {
                'type': LaundryWorldMujocoAgent,
                'x0': x0,
                'plans': plans,
                'T': T,
                'sensor_dims': sensor_dims,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [],
                'conditions': len(plans),
                'dX': dX,
                'dU': dU,
                'solver': self
            }
            self.config['algorithm']['cost'] = []

        else:
            self.config['agent']['conditions'] += len(plans)
            self.config['agent']['plans'].extend(plans)
            self.config['agent']['x0'].extend(x0)
        
        for cond in range(len(plans)):
            self.config['algorithm']['cost'].append({
                'type': TAMPCost,
                'plan': plans[cond],
                'dX': dX,
                'dU': dU,
        })

        if not self.gps:
            self.gps = GPSMain(config)
        else:
            self._update_agent(plans, x0)
            self._update_algorithm(plans, self.config['algorithm']['cost'][-len(plans):])
        self.gps.run()

    def _update_algorithm(self, plans, costs):
        if not self.gps: return
        alg = self.gps.algorithm
        alg.M += len(plans)
        alg._cond_idx = range(alg.M)
        alg._hyperparams['train_conditions'] = alg._cond_idx
        alg._hyperparams['test_conditions'] = alg._cond_idx

        # IterationData objects for each condition.
        alg.cur.extend([IterationData() for _ in range(len(plans))])
        alg.prev.extend([IterationData() for _ in range(len(plans))])

        init_traj_distr = alg._hyperparams['init_traj_distr']
        assert len(self.gps.agent.x0) == alg.M
        init_traj_distr['x0'] = self.gps.agent.x0
        init_traj_distr['dX'] = self.gps.agent.dX
        init_traj_distr['dU'] = self.gps.agent.dU

        for m in range(alg.M-len(plans), alg.M):
            alg.cur[m].traj_info = TrajectoryInfo()
            if alg._hyperparams['fit_dynamics']:
                alg.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            alg = extract_condition(
                alg._hyperparams['init_traj_distr'], alg._cond_idx[m]
            )
            alg.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        alg.cost.extend([
            costs[i]['type'](costs[i])
            for i in range(len(plans))
        ])

    def _update_agent(self, plans, x0):
        if not self.gps: return
        agent = self.gps.agent
        agent._samples.extend([[] for _ in range(self._hyperparams['conditions'])])
        agent.x0.extend(x0)
        agent.conditions += len(plans)
        agent.plans.extend(plans)

    def _solve_opt_prob(self, plan, priority, callback=None, init=True, active_ts=None, verbose=False, resample=False, smoothing = False):
        if priority == 3:
            policy = self.gps.algorithm.policy_opt
            cond = self._plan_to_cond[plan]
            traj_state = self.agent.run_policy(cond, policy, noise=np.zeros((self.T, self.dU)))
            obj_bexprs = self._traj_policy_opt(plan, traj_state)
            self._add_obj_bexprs(obj_bexprs)

        return super(BaxterPolicySolver, self)._solve_opt_prob(self, plan, priority, callback, init, active_ts, verbose, resample, smoothing)

    def _traj_policy_opt(self, plan, traj_state, norm='min-vel'):
        transfer_objs = []
        if norm == 'min-vel':
            for param, attr_name in plan.state_inds.keys():
                if param.is_symbol(): continue
                attr_type = param.get_attr_type(attr_name)
                param_ll = self._param_to_ll[param]
                T = param_ll._horizon
                active_ts = plan.active_ts
                attr_val = traj_state[plan.state_inds[(param, attr_name)], active_ts[0]:active_ts[1]+1]
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
        else:
            raise NotImplemented
        return transfer_objs