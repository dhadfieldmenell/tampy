from gps.costs import Cost

import policy_hooks.policy_solver_utils as utils

class TAMPCost(Cost):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.plan = hyperparams['plan']
        self.state_inds = hyperparams['state_inds']
        self.dX = hyperparams['dX']
        self.action_inds = hyperparams['action_inds']
        self.dU = hyperparams['dU']

    def eval(self, sample):
        utils.fill_trajectory_from_sample(sample, self.plan, self.state_inds)
        return utils.get_trajectory_cost(self.plan, self.state_inds, self.dX, self.action_inds, self.dU)
        