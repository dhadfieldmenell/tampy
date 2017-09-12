from gps.algorithm.cost.cost import Cost

import policy_hooks.policy_solver_utils as utils

class TAMPCost(Cost):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.plan = hyperparams['plan']
        self.dX = hyperparams['dX']
        self.dU = hyperparams['dU']

    def eval(self, sample):
        utils.fill_trajectory_from_sample(sample, self.plan)
        return utils.get_trajectory_cost(self.plan)
