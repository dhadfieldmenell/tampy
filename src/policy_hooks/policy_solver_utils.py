import core.util_Classes.baxter_constants as const


ACTION_ENUM = 0
STATE_ENUM = 1
OBS_ENUM = 2
NOISE_ENUM = 3

# TODO: Is there actually a reason to map to the action vector? How should we handle the action vector if it contains joint torques?
def get_plan_to_policy_mapping(plan, x_params=[], u_params=[], u_attrs=[]):
    '''
    Maps the parameters of the plan actions to indices in the policy state and action vectors, and returns the dimensions of those vectors.
    This mapping should apply to any plan with the given actions

    Parameters:
        action: The action whose eparameters are being mapped
        x_params: Which parameters to include in the state; if none are specified all are included
        u_params: Which parameters to include in the action; if none are specified all robots are included

    Returns:
        The dimension of the state vector
        Mappings from parameters to indices in the state vector
        The dimension of the action vector
        Mappings from paramters to indices in the action vector
    '''
    assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(plan.actions):
        return 0, {}, 0, {}

    actions = plan.actions
    active_ts = (actions[0].active_timesteps[0], actions[-1].active_timesteps[1])
    params = set()
    for action in actions:
        params.update(action.params)
    preds = action.preds

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0
    x_params_init, u_params_init = len(x_params), len(u_params)
    for param in params:
        param_attr_map = const.ATTR_MAP[param._type]
        # Uses all parameters for state unless otherwise specified
        if not x_params_init:
            x_params.append(param)

        # Uses all robots for policy actions unless otherwise specified
        if not u_params_init and param._type == 'Robot':
            u_params.append(param)

        if (param in x_params and param in u_params):
            param_attr_map = const.ATTR_MAP[param._type]
            for attr in param_attr_map:
                x_inds = attr[1] + cur_x_ind
                cur_x_ind = x_inds[-1] + 1
                x_vel_inds = attr[1] + cur_x_ind
                cur_x_ind = x_vel_inds[-1] + 1
                params_to_x_inds[(param, attr[0])] = x_inds
                params_to_x_inds[(param, attr[0]+'__vel')] = x_vel_inds
                if atrr not in u_attrs: continue
                u_inds = attr[1] + cur_u_ind
                cur_u_ind = u_inds[-1] + 1
                params_to_u_inds[(param, attr[0])] = u_inds

        elif param in x_params:
            for attr in param_attr_map:
                inds = attr[1] + cur_x_ind
                cur_x_ind = inds[-1] + 1
                params_to_x_inds[(param, attr[0])] = inds

        elif param in u_params:
            for attr in param_attr_map:
                if attr not in u_attrs: continue
                inds = attr[1] + cur_u_ind
                cur_u_ind = inds[-1] + 1
                params_to_u_inds[(param, attr[0])] = inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds

def fill_vector(params, params_to_inds, vec, t):
    for param in params:
        for attr in const.ATTR_MAP[param._type]:
            if (param, attr) not in params_to_inds: continue
            inds = params_to_inds[(param, inds)]
            if param.is_symbol():
                vec[inds] = getattr(param, attr)[:, 0]
            else:
                vec[inds] = getattr(param, attr)[:, t]

def set_params_attrs(params, params_to_inds, vec, t):
    for param in params:
        for attr in const.ATTR_MAP[param._type]:
            if (param, attr) not in params_to_inds: continue
            if param.is_symbol():
                getattr(param, attr)[:, 0] = vec[params_to_inds[(param, attr)]]
            else:
                getattr(param, attr)[:, t] = vvec[params_to_inds[(param, attr)]]

def fill_sample_from_trajectory(sample, plan, state_inds, action_inds, u_vec, noise, t, dX):
    active_ts, params = get_plan_traj_info(plan)

    sample.set(ACTION_ENUM, u_vec, t-active_ts[0])

    X = np.zeros((dU, 1))
    fill_vector(params, state_inds, X, t)
    sample.set(STATE_ENUM, X, t-active_ts[0])

    sample.set(NOISE, noise, t-active_ts[0])

def fill_trajectory_from_sample(sample, plan, state_inds):
    params = set()
    for action in plan.actions:
        params.update(action.params)
    params = list(params)
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    for t in range(active_ts[0], active_ts[1]+1):
        X = sample.get_X(t)
        set_params_attrs(params, state_inds, X, t)

def get_trajectory_cost(plan, state_inds, dX, action_inds, dU):
    '''
    Calculates the constraint violations at the provided timestep for the current trajectory, as well as the first and second order approximations.
    This function handles the hierarchies of mappings from parameters to attributes to indices and translates between how the predicates consturct
    those hierachies and how the policy states & actions 

    state_inds & action_inds map (param. attr_name) to the the relevant indices
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
                    if (param, attr_name) in state_inds:
                        first_order_x_approx[t-active_ts[0], state_inds[(param, attr_name)]] += first_degree_convexification[param_attr_inds[param][attr_name]]
                    if (param, attr_name) in action_inds:
                        first_order_u_approx[t-active_ts[0], action_inds[(param, attr_name)]] += first_degree_convexification[param_attr_inds[param][attr_name]]

            # Quadratic terms
            second_degree_convexification = expr.convexify(param_vector, degree=2).eval(param_vector)
            for param_1 in param_attr_inds:
                for param_2 in param_attr_inds:
                    for attr_name_1 in param_attr_inds[param_1]:
                        for attr_name_2 in param_attr_inds[param2]:
                            if (param_1, attr_name_1) in state_inds and (param_2, attr_name_2) in state_inds:
                                x_inds_1 = state_inds[(param_1, attr_name_1)]
                                x_inds_2 = state_inds[(param_2, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                assert len(x_inds_1) == len(pred_inds_1) amd len(x_inds_2) == len(pred_inds_2)
                                second_order_xx_approx[t-active_ts[0], x_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                            if (param_1, attr_name_1) in action_inds[(param_1, attr_name_1)] and (param_2, attr_name_2) in action_inds[(param_2, attr_name_2)]:
                                u_inds_1 = action_inds[(param_1, attr_name_1)]
                                u_inds_2 = action_inds[(param_2, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                assert len(u_inds_1) == len(pred_inds_1) amd len(u_inds_2) == len(pred_inds_2)
                                second_order_uu_approx[t-active_ts[0], u_inds_1, u_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                            if (param_1, attr_name_1) in action_inds[(param_1, attr_name_1)] and (param_2, attr_name_2) in state_inds[(param_2, attr_name_2)]:
                                u_inds_1 = action_inds[(param_1, attr_name_1)]
                                x_inds_2 = state_inds[(param_2, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2][attr_name_2]
                                assert len(u_inds_1) == len(pred_inds_1) amd len(x_inds_2) == len(pred_inds_2)
                                second_order_ux_approx[t-active_ts[0], u_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            preds_checked.append(p['pred'])

    return timestep_costs, first_order_x_approx, first_order_u_approx, second_order_xx_approx, second_order_uu_approx, second_order_ux_approx

def map_trajectory_to_vel_acc(plan, dU, action_inds):
    '''
        Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
    '''
    params = set()
    for action in plan.actions:
        params.update(action.params)
    params = list(params)
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    T = active_ts[1] - active_ts[0] + 1
    vels = np.zeros((dU, T))
    accs = np.zeros((dU, T))

    a = np.zeros((dU,))
    v = np.zeros((dU,))
    for t in range(active_ts[0], active_ts[1]):
        U_0 = np.zeros((dU,))
        U = np.zeros((dU,))
        fill_vector(params, action_inds, U_0, t)
        fill_vector(params, action_inds, U, t+1)
        real_t = plan.params['baxter'].time[t]
        vels[:, t-active_ts[0]] = v
        a = 2*(U-U_0+v*real_t) / (real_t**2)
        accs[:, t-active_ts[0]] = a
        v = v + a*real_t
    vels[:, active_ts[1]-active_ts[0]] = v + a*plan.params['baxter'].time[action_ts[1]]

    return vels, accs

def get_plan_traj_info(plan):
    '''
        Extract active timesteps and active parameters from the plan
    '''
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    params = set()
    for action in plan.actions:
        params.update(action.params)
    params = list(params)
    return active_ts, paarams
