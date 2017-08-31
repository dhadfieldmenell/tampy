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
        return self.get_trajectory_cost(self.plan, self.state_inds, self.dX, self.action_inds, self.dU)

    def get_trajectory_cost(self, plan, state_inds, dX, action_inds, dU):
        '''
        Calculates the constraint violations at the provided timestep for the current trajectory, as well as the first and second order approximations.
        This function handles the hierarchies of mappings from parameters to attributes to indices and translates between how the predicates consturct
        those hierachies and how the policy states & actions 
        '''
        preds = []
        for action in plan.actions:
            preds.extend(action.preds)
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        timestep_costs = np.zeros((T, )) # l
        first_order_x_approx = np.zeros((T, dX, )) # lx
        first_order_u_approx = np.zeros((T, dU, )) # lu
        second_order_xx_approx = np.zeros((T, dX, dX)) # lxx
        second_order_uu_approx = np.zeros((T, dU, dU)) # luu
        second_order_ux_approx = np.zeros((T, dU, dX)) # lux

        pred_param_attr_inds = {}
        for p in preds:
            if p in pred_param_attr_inds: continue
            pred_param_attr_inds[pred] = {}
            attr_inds = p['pred'].attr_inds
            cur_ind = 0
            for param in attr_inds:
                pred_param_attr_inds[pred][param] = {}
                for attr_name, inds in attr_inds[param]:
                    pred_param_attr_inds[pred][param][attr_name] = np.array(range(cur_ind, cur_ind+len(inds)))
                    cur_ind += len(inds)

        for t in range(active_ts[0], active_ts[1]+1):
            active_preds = plan.get_active_preds(t)
            preds_checked = []
            for p in preds:
                if p['pred'] not in active_preds or p['pred'] in preds_checked: continue
                attr_inds = p['pred'].attr_inds
                comp_expr = p['pred'].get_expr(negated=p['negated'])

                # Constant terms
                expr = comp_expr.expr if comp_expr else continue
                param_vector = expr.eval(p['pred'].get_param_vector(t))
                param_attr_inds = pred_param_attr_inds[pred]
                timestep_costs[t-active_ts[0]] += np.sum(-1 * param_vector)

                # Linear terms
                first_degree_convexification = expr.convexify(param_vector, degree=1).eval(param_vector)
                for param in param_attr_inds:
                    for attr_name in param_attr_inds[param]:
                        if param in state_inds and attr_name in state_inds[param]:
                            first_order_x_approx[t-active_ts[0], state_inds[param][attr_name]] += first_degree_convexification[param_attr_inds[param][attr_name]]
                        if param in action_inds and attr_name in action_inds[param]:
                            first_order_u_approx[t-active_ts[0], action_inds[param][attr_name]] += first_degree_convexification[param_attr_inds[param][attr_name]]

                # Quadratic terms
                second_degree_convexification = expr.convexify(param_vector, degree=2).eval(param_vector)
                for param_1 in param_attr_inds:
                    for param_2 in param_attr_inds:
                        for attr_name_1 in param_attr_inds[param_1]:
                            for attr_name_2 in param_attr_inds[param2]:
                                if param_1 in state_inds and param_2 in state_inds and attr_name_1 in state_inds[param_1] and attr_name_2 in state_inds[param_2]:
                                    x_inds_1 = state_inds[param_1][attr_name_1]
                                    x_inds_2 = state_inds[param_2][attr_name_2]
                                    pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                    pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                    assert len(x_inds_1) == len(pred_inds_1) amd len(x_inds_2) == len(pred_inds_2)
                                    second_order_xx_approx[t-active_ts[0], x_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                                if param_1 in action_inds and param_2 in action_inds and attr_name_1 in action_inds[param_1] and attr_name_2 in action_inds[param_2]:
                                    u_inds_1 = action_inds[param_1][attr_name_1]
                                    u_inds_2 = action_inds[param_2][attr_name_2]
                                    pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                    pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                    assert len(u_inds_1) == len(pred_inds_1) amd len(u_inds_2) == len(pred_inds_2)
                                    second_order_uu_approx[t-active_ts[0], u_inds_1, u_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                                if param_1 in action_inds and param_2 in state_inds and attr_name_1 in action_inds[param_1] and attr_name_2 in state_inds[param_2]:
                                    u_inds_1 = action_inds[param_1][attr_name_1]
                                    x_inds_2 = state_inds[param_2][attr_name_2]
                                    pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                    pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                    assert len(u_inds_1) == len(pred_inds_1) amd len(x_inds_2) == len(pred_inds_2)
                                    second_order_ux_approx[t-active_ts[0], u_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                preds_checked.append(p['pred'])

        return timestep_costs, first_order_x_approx, first_order_u_approx, second_order_xx_approx, second_order_uu_approx, second_order_ux_approx
