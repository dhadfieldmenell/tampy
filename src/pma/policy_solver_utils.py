import core.util_Classes.baxter_constants as const


ACTION_ENUM = 0
STATE_ENUM = 1
OBS_ENUM = 2
NOISE_ENUM = 3

def get_action_description(action, u_params=None):
    params = action.params
    preds = action.preds

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0
    for param in params:
        param_inds = {}
        param_attr_map = const.ATTR_MAP[param._type]
        # Uses all non-symbol parameters for policy actions unless otherwise specified
        if (u_params and param in u_params) or not (u_params or param.is_symbol()):
            param_attr_map = const.ATTR_MAP[param._type]
            attr_to_u_inds = {}
            for attr in param_attr_map:
                x_inds = attr[1] + cur_x_ind
                cur_x_ind = x_inds[-1] + 1
                x_vel_inds = attr[1] + cur_u_ind
                cur_x_ind = x_vel_inds[-1] + 1
                param_inds[attr[0]] = x_inds
                param_inds[attr[0]+'__vel'] = x_vel_inds

                u_inds = attr[1] + cur_u_ind
                cur_u_ind = u_inds[-1] + 1
                attr_to_u_inds[attr[0]] = u_inds
            params_to_u_inds[param] = attr_to_u_inds
            params_to_x_inds[param] = param_inds
        else:
            for attr in param_attr_map:
                inds = attr[1] + cur_x_ind
                cur_x_ind = inds[-1] + 1
                param_inds[attr[0]] = inds
            params_to_x_inds[param] = param_inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds

def fill_vector(params, params_to_inds, vec, t):
    for param in params:
        if param not in params_to_inds: continue
        param_inds = params_to_inds[param]
        if not param.is_symbol():
            for attr in param_inds:
                if hasattr(param, attr):
                    vec[param_inds[attr]] = getattr(param, attr)[:, t]
        else:
            for attr in param_inds:
                if hasattr(param, attr):
                    vec[param_inds[attr]] = getattr(param, attr)[:, 0]

def set_params_attrs(params, params_to_inds, vec, t):
    for param in params:
        if param not in params_to_inds: continue
        param_inds = params_to_inds[param]
        if not param.is_symbol():
            for attr in param_inds:
                if hasattr(param, attr):
                    getattr(param, attr)[:, t] = vec[param_inds[attr]]
        else:
            for attr in param_inds:
                if hasattr(param, attr):
                    getattr(param, attr)[:, 0] = vec[param_inds[attr]]

def fill_sample_ts_from_trajectory(sample, action, state_inds, action_inds, noise, t, dU, dX):
    params = action.params
    active_ts = action.active_timesteps
    U = np.zeros((dU, 1))
    # A policy action is the joint values on the next timestep
    if t < action.active_timesteps[1] - 1:
        fill_vector(params, action_inds, U, t+1)
    sample.set(ACTION_ENUM, U, t-active_ts[0])

    X = np.zeros((dU, 1))
    fill_vector(params, state_inds, X, t)
    sample.set(STATE_ENUM, X, t-active_ts[0])

    sample.set(NOISE, noise, t-active_ts[0])

def fill_trajectory_ts_from_policy(policy, action, state_inds, action_inds, noise, t, dX, obs=[]):
    params = action.params
    active_ts = action.active_timesteps
    X = np.zeros((dX,))
    fill_vector(params, state_inds, X, t)
    U = policy.act(X, obs, t-active_ts[0], noise[t-active_ts[0], :])
    set_params_attrs(params, action_inds, U, t+1)

def fill_state_from_sample(sample, action, state_inds):
    params = action.params
    active_ts = action.active_timesteps
    for t in range(active_ts[0], active_ts[1]+1):
        X = sample.get_X(t)
        set_params_attrs(params, state_inds, X, t)

def get_action_costs(action):
        active_ts = action.active_timesteps
        costs = np.zeros((active_ts[1]-active_ts[0]+1))
        preds = action.preds
        for ts in range(active_ts[0], active_ts[1]+1):
            timestep_cost = 0
            active_preds = action.get_active_preds(ts)
            for p in preds:
                if p['pred'] not in active_preds: continue
                param_vector = p['pred'].get_param_vector(ts)
                timestep_cost += np.sum(np.abs(p['pred'].get_expr(negated=p['negated']).expr.eval(param_vector)))
            costs[ts] += timestep_cost
        return costs

def reset_action(action, state_inds, x0):
    set_params_attrs(action.params, state_inds, x0, 0)
