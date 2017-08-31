import numpy as np
from gps.gps_main import GPSMain

from  pma.robot_ll_solver import RobotLLSolver
import policy_hooks.policy_hyperparams as baxter_hyperparams
import policy_hooks.policy_solver_utils as utils


class BaxterPolicySolver(RobotLLSolver):
    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        self.plans = []
        self.config = baxter_hyperparams.config
        self.gps = None
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)

    # TODO: Add hooks for online policy learning
    def train_policy(self, plans, hyperparams=None):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must contain the exact same sequence of actions, all of which must be able to have a policy trained on (train_policy=True)
        '''
        if hyperparams and self.config:
            self.config.update(hyperparams)
        dX, state_inds, dU, action_inds = get_plan_to_policy_mapping(plans[0])
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            utils.STATE_ENUM: dX,
            utils.ACtiON_ENUM: dU
        }

        x0 = np.zeros((len(plans), dX))
        for i in range(len(plans)):
            plan = plans[i]
            utils.fill_vector(plan.actions[0].params, state_inds, x0[i], (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1]))

        if not self.config:
            self.config = baxter_hyperparams.config if not hyperparams else hyperparams
            self.config['agent'] = {
                'type': TAMPAgent,
                'x0': x0,
                'plans': plans,
                'T': T,
                'sensor_dims': SENSOR_DIMS,
                'state_include': [utils.STATE_ENUM],
                'obs_include': [],
                'conditions': len(plans),
                'state_inds': state_inds,
                'action_inds': action_inds,
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
                'state_inds': state_inds,
                'dX': dX,
                'action_inds': action_inds,
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
