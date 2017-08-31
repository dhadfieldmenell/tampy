import numpy as np
from gps.gps_main import GPSMain

from  pma.robot_ll_solver import RobotLLSolver
import policy_hooks.policy_hyperparams as hyperparams
import policy_hooks.policy_solver_utils as utils

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 64

class BaxterPolicySolver(RobotLLSolver):
    # TODO: Add hooks for online policy learning
    def train_policy(self, plans, n_samples=5, iterations=5, active_ts=None, callback=None, n_resamples=5, verbose=False):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning

        Each plan must contain the exact same sequence of actions, all of which must be able to have a policy trained on (train_policy=True)
        '''
        config = hyperparams.config
        dX, state_inds, dU, action_inds = get_plan_to_policy_mapping(plans[0])
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            utils.STATE_ENUM: dX,
            utils.ACtiON_ENUM: dU
        }

        x0 = np.zeros((len(plans), dX))
        for i in range(len(plans)):
            utils.fill_vector(plans[i].actions[0].params, state_inds, x0[i], active_ts[0])

        config['agent'] = {
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

        config['algorithm']['cost'] = []
        for cond in range(len(plans)):
            config['algorithm']['cost'].append({
                'type': TAMPCost,
                'plan': plans[cond],
                'state_inds': state_inds,
                'dX': dX,
                'action_inds': action_inds,
                'dU': dU,
            })

        gps = GPSMain(config)
        gps.run()
