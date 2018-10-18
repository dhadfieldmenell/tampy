""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier

from policy_hooks.utils.policy_solver_utils import ACTION_ENUM


class PolInfCost(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.policy_opt = hyperparams['policy_opt']

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        pol_mu, pol_sig, _, _ = self.policy_opt(sample.get_obs(), sample.task)
        inf_weight = 1. / np.diag(pol_sig[0, 0])

        for data_type in self._hyperparams['data_types']:
            if data_type != ACTION_ENUM: continue
            config = self._hyperparams['data_types'][data_type]
            wp = config['wp']
            u = sample.get_U()
            _, dim_sensor = u.shape

            wpm = get_ramp_multiplier(
                self._hyperparams['ramp_option'], T,
                wp_final_multiplier=self._hyperparams['wp_final_multiplier']
            )
            wp = inf_weight * wp * np.expand_dims(wpm, axis=-1)
            # Compute state penalty.
            dist = u - pol_mu[0]

            # Evaluate penalty term.
            l, ls, lss = evall1l2term(
                wp, dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
                np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
                self._hyperparams['l1'], self._hyperparams['l2'],
                self._hyperparams['alpha']
            )

            final_l += l

            sample.agent.pack_data_x(final_lx, ls, data_types=[data_type])
            sample.agent.pack_data_x(final_lxx, lss,
                                     data_types=[data_type, data_type])
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux