import core.util_classes.baxter_constants as const

import numpy as np

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
    # assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(plan.actions):
        return 0, {}, 0, {}

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0
    x_params_init, u_params_init = len(x_params), len(u_params)

    if not x_params_init:
        params = plan.params.values()
        x_params_init = len(x_params)
    else:
        params = x_params

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
                params_to_x_inds[(param.name, attr[0])] = x_inds
                params_to_x_inds[(param.name, attr[0]+'__vel')] = x_vel_inds
                if attr[0] not in u_attrs: continue
                u_inds = attr[1] + cur_u_ind
                cur_u_ind = u_inds[-1] + 1
                params_to_u_inds[(param.name, attr[0])] = u_inds

        elif param in x_params:
            for attr in param_attr_map:
                inds = attr[1] + cur_x_ind
                cur_x_ind = inds[-1] + 1
                params_to_x_inds[(param.name, attr[0])] = inds

        elif param in u_params:
            for attr in param_attr_map:
                if attr[0] not in u_attrs: continue
                inds = attr[1] + cur_u_ind
                cur_u_ind = inds[-1] + 1
                params_to_u_inds[(param.name, attr[0])] = inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds

def fill_vector(params, params_to_inds, vec, t):
    for param in params:
        for attr in const.ATTR_MAP[param._type]:
            if (param.name, attr[0]) not in params_to_inds: continue
            inds = params_to_inds[(param.name, attr[0])]
            if param.is_symbol():
                vec[inds] = getattr(param, attr[0])[:, 0]
            else:
                vec[inds] = getattr(param, attr[0])[:, t]

def set_params_attrs(params, params_to_inds, vec, t):
    for param in params:
        for attr in const.ATTR_MAP[param._type]:
            if (param.name, attr[0]) not in params_to_inds: continue
            if param.is_symbol():
                getattr(param, attr[0])[:, 0] = vec[params_to_inds[(param.name, attr[0])]]
            else:
                getattr(param, attr[0])[:, t] = vec[params_to_inds[(param.name, attr[0])]]

def fill_sample_from_trajectory(sample, plan, u_vec, noise, t, dX):
    active_ts, params = get_plan_traj_info(plan)

    sample.set(ACTION_ENUM, u_vec, t-active_ts[0])

    X = np.zeros((dU, 1))
    fill_vector(params, plan.state_inds, X, t)
    sample.set(STATE_ENUM, X, t-active_ts[0])

    sample.set(NOISE, noise, t-active_ts[0])

def fill_trajectory_from_sample(sample, plan):
    params = set()
    for action in plan.actions:
        params.update(action.params)
    params = list(params)
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    for t in range(active_ts[0], active_ts[1]+1):
        X = sample.get_X(t)
        set_params_attrs(params, plan.state_inds, X, t)

def get_trajectory_cost(plan):
    '''
    Calculates the constraint violations at the provided timestep for the current trajectory, as well as the first and second order approximations.
    This function handles the hierarchies of mappings from parameters to attributes to indices and translates between how the predicates consturct
    those hierachies and how the policy states & actions 

    state_inds & action_inds map (param. attr_name) to the the relevant indices
    '''
    preds = []
    state_inds = plan.state_inds
    action_inds = plan.action_inds
    dX = plan.dX
    dU = plan.dU
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
            pred_param_attr_inds[pred][param.name] = {}
            for attr_name, inds in attr_inds[param.name]:
                pred_param_attr_inds[pred][param.name][attr_name] = np.array(range(cur_ind, cur_ind+len(inds)))
                cur_ind += len(inds)

    for t in range(active_ts[0], active_ts[1]+1):
        active_preds = plan.get_active_preds(t)
        preds_checked = []
        for p in preds:
            if p['pred'] not in active_preds or p['pred'] in preds_checked: continue
            attr_inds = p['pred'].attr_inds
            comp_expr = p['pred'].get_expr(negated=p['negated'])

            # Constant terms
            expr = comp_expr.expr if comp_expr else None
            if not expr: continue
            param_vector = expr.eval(p['pred'].get_param_vector(t))
            param_attr_inds = pred_param_attr_inds[pred]
            timestep_costs[t-active_ts[0]] += np.sum(-1 * param_vector)

            # Linear terms
            first_degree_convexification = expr.convexify(param_vector, degree=1).eval(param_vector)
            for param in param_attr_inds:
                for attr_name in param_attr_inds[param]:
                    if (param.name, attr_name) in state_inds:
                        first_order_x_approx[t-active_ts[0], state_inds[(param.name, attr_name)]] += first_degree_convexification[param_attr_inds[param.name][attr_name]]
                    if (param.name, attr_name) in action_inds:
                        first_order_u_approx[t-active_ts[0], action_inds[(param.name, attr_name)]] += first_degree_convexification[param_attr_inds[param.name][attr_name]]

            # Quadratic terms
            second_degree_convexification = expr.convexify(param_vector, degree=2).eval(param_vector)
            for param_1 in param_attr_inds:
                for param_2 in param_attr_inds:
                    for attr_name_1 in param_attr_inds[param_1.name]:
                        for attr_name_2 in param_attr_inds[param2.name]:
                            if (param_1.name, attr_name_1) in state_inds and (param_2.name, attr_name_2) in state_inds:
                                x_inds_1 = state_inds[(param_1.name, attr_name_1)]
                                x_inds_2 = state_inds[(param_2.name, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1.name][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2.name][attr_name_2]
                                assert len(x_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
                                second_order_xx_approx[t-active_ts[0], x_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                            if (param_1.name, attr_name_1) in action_inds and (param_2, attr_name_2) in action_inds:
                                u_inds_1 = action_inds[(param_1.name, attr_name_1)]
                                u_inds_2 = action_inds[(param_2.name, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1.name][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2.name][attr_name_2]
                                assert len(u_inds_1) == len(pred_inds_1) and len(u_inds_2) == len(pred_inds_2)
                                second_order_uu_approx[t-active_ts[0], u_inds_1, u_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

                            if (param_1.name, attr_name_1) in action_inds and (param_2.name, attr_name_2) in state_inds:
                                u_inds_1 = action_inds[(param_1.name, attr_name_1)]
                                x_inds_2 = state_inds[(param_2.name, attr_name_2)]
                                pred_inds_1 = param_attr_inds[param_1.name][attr_name_1]
                                pred_inds_2 = param_attr_inds[param_2.name][attr_name_2]
                                assert len(u_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
                                second_order_ux_approx[t-active_ts[0], u_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            preds_checked.append(p['pred'])

    return timestep_costs, first_order_x_approx, first_order_u_approx, second_order_xx_approx, second_order_uu_approx, second_order_ux_approx

def map_trajectory_to_vel_acc(plan):
    '''
        Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
    '''
    params = get_action_params(plan)
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    T = active_ts[1] - active_ts[0] + 1
    vels = np.zeros((plan.dU, T))
    accs = np.zeros((plan.dU, T))

    a = np.zeros((plan.dU,))
    v = np.zeros((plan.dU,))
    for t in range(active_ts[0], active_ts[1]):
        U_0 = np.zeros((plan.dU,))
        U = np.zeros((plan.dU,))
        fill_vector(params, plan.action_inds, U_0, t)
        fill_vector(params, plan.action_inds, U, t+1)
        real_t = plan.time[0, t]
        vels[:, t-active_ts[0]] = v
        a = 2*(U-U_0-v*real_t) / (real_t**2)
        accs[:, t-active_ts[0]] = a
        v = v + a*real_t
    vels[:, active_ts[1]-active_ts[0]] = v + a*plan.time[0, active_ts[1]]

    return vels, accs

def get_plan_traj_info(plan):
    '''
        Extract active timesteps and active parameters from the plan
    '''
    active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    # if hasattr(plan, 'state_inds'):
    #     params = list(set(map(lambda k: k[0], plan.state_inds.keys())))
    # else:
    #     params = plan.params.values()
    params = get_state_params(plan)
    return active_ts, params

def create_sub_plans(plan, action_sequence):
    next_plan_acts = []
    cur_seq_ind = 0
    plans = []
    for i in range(len(plan.actions)):
        act = plan.actions[i]
        if act.name == action_sequnce[cur_seq_ind]:
            next_plan_acts.append(act)
            cur_seq_ind += 1
            if cur_seq_ind >= len(action_sequnce):
                plans.append(Plan(plan.params, next_plan_acts, plan.horizon, plan.env, False))
                next_plan_acts = []
                cur_seq_ind = 0
        else:
            next_plan_acts = []
            cur_seq_ind = 0

    return plans

def get_state_params(plan):
    assert hasattr(plan, 'state_inds')
    params = map(lambda k: plan.params[k[0]], plan.state_inds.keys())
    return list(set(params))

def get_action_params(plan):
    assert hasattr(plan, 'action_inds')
    params = map(lambda k: plan.params[k[0]], plan.action_inds.keys())
    return list(set(params))
