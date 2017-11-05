from gps.algorithm.cost.cost import Cost

import policy_hooks.policy_solver_utils as utils

class TAMPCost(Cost):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.plan = hyperparams['plan']
        self.dX = hyperparams['dX']
        self.dU = hyperparams['dU']
        self.x0 = hyperparams['x0']

        self.first_act = self.plan.actions[x0[1][0]]
        self.last_act = self.plan.actions[x0[1][1]]
        self.init_t = first_act.active_timesteps[0]
        self.final_t = last_act.active_timesteps[1]

        params = set()
        for act in range(x[1][0], x[1][1]):
            next_act = self.plan.actions[act]
            params.update(next_act.params)
        self.symbols = filter(lambda p: p.is_symbol(), list(params))
        self.params = filter(lambda p: not p.is_symbol(), list(params))

    def eval(self, sample):
        self.fill_symbolic_values()
        self.fill_trajectory_from_sample(sample)
        return utils.get_trajectory_cost(self.plan)

    def fill_symbolic_values(self):
        set_params_attrs(self.symbols, self.plan.state_inds, self.x0[0], 0)

    def fill_trajectory_from_sample(self, sample):
        set_params_attrs(self.params, plan.state_inds, self.x0[0], 0)
        for t in range(self.init_t+1, self.final_t+1):
            X = self._clip_joint_angles(sample.get_X(t-init_t))
            set_params_attrs(self.params, self.plan.state_inds, X, t)

    def _clip_joint_angles(self, X):
        DOF_limits = self.plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
        lArmPose = X[self.plan.state_inds[('baxter', 'lArmPose')]]
        rArmPose = X[self.plan.state_inds[('baxter', 'rArmPose')]]
        for i in range(7):
            if lArmPose[i] < left_DOF_limits[0][i]:
                lArmPose[i] = left_DOF_limits[0][i]
            if lArmPose[i] > left_DOF_limits[1][i]:
                lArmPose[i] = left_DOF_limits[1][i]
            if rArmPose[i] < right_DOF_limits[0][i]:
                rArmPose[i] = right_DOF_limits[0][i]
            if rArmPose[i] > right_DOF_limits[1][i]:
                rArmPose[i] = right_DOF_limits[1][i]
