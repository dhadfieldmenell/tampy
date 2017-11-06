from gps.algorithm.cost.cost import Cost

import policy_hooks.policy_solver_utils as utils

class TAMPCost(Cost):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.plan = hyperparams['plan']
        self.dX = hyperparams['dX']
        self.dU = hyperparams['dU']
        self.x0 = hyperparams['x0']

        self.first_act = self.plan.actions[self.x0[1][0]]
        self.last_act = self.plan.actions[self.x0[1][1]]
        self.init_t = self.first_act.active_timesteps[0]
        self.final_t = self.last_act.active_timesteps[1]

        params = set()
        for act in range(self.x0[1][0], self.x0[1][1]):
            next_act = self.plan.actions[act]
            params.update(next_act.params)
        self.symbols = filter(lambda p: p.is_symbol(), list(params))
        self.params = filter(lambda p: not p.is_symbol(), list(params))

    def eval(self, sample):
        self.fill_symbolic_values()
        self.fill_trajectory_from_sample(sample)
        first_act = self.plan.actions[self.x0[1][0]]
        last_act = self.plan.actions[self.x0[1][1]]
        init_t = first_act.active_timesteps[0]
        final_t = last_act.active_timesteps[1]
        return utils.get_trajectory_cost(self.plan, init_t, final_t) / (final_t - init_t + 1)

    def fill_symbolic_values(self):
        set_params_attrs(self.symbols, self.plan.state_inds, self.x0[0], 0)

    def fill_trajectory_from_sample(self, sample):
        set_params_attrs(self.params, plan.state_inds, self.x0[0], 0)
        for t in range(self.init_t+1, self.final_t+1):
            X = sample.get_X(t-init_t)
            set_params_attrs(self.params, self.plan.state_inds, X, t)
