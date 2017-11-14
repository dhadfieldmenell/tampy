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
        self._clip_joint_angles()
        return utils.get_trajectory_cost(self.plan, init_t, final_t, time_interval=200)

    def fill_symbolic_values(self):
        utils.set_params_attrs(self.symbols, self.plan.state_inds, self.x0[0], 0)

    def fill_trajectory_from_sample(self, sample):
        utils.set_params_attrs(self.params, self.plan.state_inds, self.x0[0], 0)
        for t in range(self.init_t+1, self.final_t):
            X = sample.get_X((t-self.init_t)*200)
            utils.set_params_attrs(self.params, self.plan.state_inds, X, t)

    def _clip_joint_angles(self):
        DOF_limits = self.plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
        lArmPose = self.plan.params['baxter'].lArmPose
        lGripper = self.plan.params['baxter'].lGripper
        rArmPose = self.plan.params['baxter'].rArmPose
        rGripper = self.plan.params['baxter'].rGripper
        for t in range(self.plan.horizon):
            for i in range(7):
                if lArmPose[i, t] < left_DOF_limits[0][i]:
                    lArmPose[i, t] = left_DOF_limits[0][i]
                if lArmPose[i, t] > left_DOF_limits[1][i]:
                    lArmPose[i, t] = left_DOF_limits[1][i]
                if rArmPose[i, t] < right_DOF_limits[0][i]:
                    rArmPose[i, t] = right_DOF_limits[0][i]
                if rArmPose[i, t] > right_DOF_limits[1][i]:
                    rArmPose[i, t] = right_DOF_limits[1][i]
            if lGripper[0, t] < DOF_limits[0][9]:
                lGripper[0, t] = DOF_limits[0][9]
            if lGripper[0, t] > DOF_limits[1][9]:
                lGripper[0, t] = DOF_limits[1][9]
            if rGripper[0, t] < DOF_limits[0][17]:
                rGripper[0, t] = DOF_limits[0][17]
            if rGripper[0, t] > DOF_limits[1][17]:
                rGripper[0, t] = DOF_limits[1][17]
