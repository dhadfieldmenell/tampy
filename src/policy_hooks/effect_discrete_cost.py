import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost

from policy_hooks.utils.policy_solver_utils import set_param_attrs
from policy_hooks.utils.tamp_eval_funcs import violated_ll_constrs, ts_constr_grad 


class EffectViolationCost(Cost):
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)

        self.plan = config['plan']
        self.compute_grad = config['compute_grad']
        Cost.__init__(self, config)

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

        _, dim_sensor = x.shape

        end_ts = self.plan.horizon - 1
        set_param_attrs(self.plan.params.values(), self.plan.state_inds, X, end_ts)
        final_l[-1] = len(violated_ll_constrs(self.plan, end_ts))
        
        if self.compute_grad:
             final_lu[-1,:], final_lx[-1,:], final_luu[-1,:], final_lxx[-1,:], final_lux[-1,:] = ts_grads(self.plan, self.plan.state_inds, t=end_ts)

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

