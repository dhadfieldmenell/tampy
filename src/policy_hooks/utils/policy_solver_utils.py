from core.util_classes.robot_predicates import CollisionPredicate
import numpy as np

# Don't change these
ACTION_ENUM = 0
STATE_ENUM = 1
OBS_ENUM = 2
NOISE_ENUM = 3
EE_ENUM = 4
GRIPPER_ENUM = 5
TRAJ_HIST_ENUM = 6
COLORS_ENUM = 7
TASK_ENUM = 8
TARGETS_ENUM = 9

IM_H = 140
IM_W = 140
IM_C = 3

GPS_RATIO = 1e3

MUJOCO_STEPS_PER_SECOND = 200
POLICY_STEPS_PER_SECOND = 1


def get_state_action_inds(plan, robot_name, attr_map, x_params={}, u_params={}):
    '''
    Maps the parameters of the plan actions to indices in the policy state and action vectors, and returns the dimensions of those vectors.
    This mapping should apply to any plan with the given actions. Replaces get_plan_to_policy_mapping.
    '''
    # assert all(map(lambda a: a.train_policy, plan.actions))
    if not len(plan.actions):
        return 0, {}, 0, {}

    params_to_x_inds, params_to_u_inds = {}, {}
    cur_x_ind, cur_u_ind = 0, 0

    robot_x_attrs = x_params[robot_name]
    robot_u_attrs = u_params[robot_name]
    robot_attr_map = attr_map['Robot']
    ee_pos_attrs = ['ee_left_pos', 'ee_right_pos']
    ee_rot_attrs = ['ee_left_rot', 'ee_right_rot']
    for attr in robot_x_attrs:
        if attr in ee_pos_attrs:
            x_inds = np.array([0, 1, 2]) + cur_x_ind
            cur_x_ind = x_inds[-1] + 1
            params_to_x_inds[(robot_name, attr)] = x_inds
            continue

        if attr in ee_rot_attrs:
            x_inds = np.array([0, 1, 2, 3]) + cur_x_ind
            cur_x_ind = x_inds[-1] + 1
            params_to_x_inds[(robot_name, attr)] = x_inds
            continue

        inds = filter(lambda p: p[0]==attr, robot_attr_map)[0][1]
        x_inds = inds + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot_name, attr)] = x_inds

    for attr in robot_u_attrs:
        if attr in ee_pos_attrs:
            u_inds = np.array([0, 1, 2]) + cur_u_ind
            cur_u_ind = u_inds[-1] + 1
            params_to_u_inds[(robot_name, attr)] = u_inds
            continue

        if attr in ee_rot_attrs:
            u_inds = np.array([0, 1, 2, 3]) + cur_u_ind
            cur_u_ind = u_inds[-1] + 1
            params_to_u_inds[(robot_name, attr)] = u_inds
            continue

        inds = filter(lambda p: p[0]==attr, robot_attr_map)[0][1]
        u_inds = inds + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot_name, attr)] = u_inds

    for param_name in x_params:
        if param_name not in plan.params: continue
        param = plan.params[param_name]
        param_attr_map = attr_map[param._type]
        for attr in x_params[param_name]:
            if (param_name, attr) in params_to_x_inds: continue
            inds = filter(lambda p: p[0]==attr, attr_map[param._type])[0][1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr)] = inds

    symbolic_boundary = cur_x_ind # Used to differntiate parameters from symbols in the state vector

    for param in plan.params.values():
        if not param.is_symbol(): continue
        param_attr_map = attr_map[param._type]
        for attr in param_attr_map:
            if (param.name, attr[0]) in params_to_x_inds: continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds, symbolic_boundary

def get_target_inds(plan, attr_map, include):
    cur_ind = 0
    target_inds = {}
    for param in plan.params.values():
        if param.name in include:
            for attr in include[param.name]:
                param_attr_map = attr_map[param._type]
                inds = filter(lambda p: p[0]==attr, attr_map[param._type])[0][1] + cur_ind
                cur_ind = inds[-1] + 1
                target_inds[param.name, attr] = inds

    return cur_ind, target_inds

def get_plan_to_policy_mapping(plan, robot_name, x_params=[], u_params=[], x_attrs=[], u_attrs=[]):
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
        params = map(lambda p: plan.params[p], x_params)

    robot = plan.params[robot_name] #TODO: Make this more general
    robot_attr_map = const.ATTR_MAP[robot._type]
    ee_pos_attrs = ['ee_left_pos', 'ee_right_pos']
    ee_rot_attrs = ['ee_left_rot', 'ee_right_rot']
    for attr in robot_attr_map:
        if len(u_attrs) and attr[0] not in u_attrs: continue
        # if attr[0] != 'lArmPose' and attr[0] != 'rArmPose' and attr[0] != 'lGripper' and attr[0] != 'rGripper':
        x_inds = attr[1] + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot.name, attr[0])] = x_inds
        u_inds = attr[1] + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot.name, attr[0])] = u_inds
    for attr in ee_pos_attrs:
        if attr not in u_attrs: continue
        x_inds = np.array([0, 1, 2]) + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot.name, attr)] = x_inds
        u_inds = np.array([0, 1, 2]) + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot.name, attr)] = u_inds
    for attr in ee_rot_attrs:
        if attr not in u_attrs: continue
        x_inds = np.array([0, 1, 2, 3]) + cur_x_ind
        cur_x_ind = x_inds[-1] + 1
        params_to_x_inds[(robot.name, attr)] = x_inds
        u_inds = np.array([0, 1, 2, 3]) + cur_u_ind
        cur_u_ind = u_inds[-1] + 1
        params_to_u_inds[(robot.name, attr)] = u_inds
    # for attr in robot_attr_map:
    #     if len(u_attrs) and attr[0] not in u_attrs: continue
    #     # if attr[0] != 'lArmPose' and attr[0] != 'rArmPose' and attr[0] != 'lGripper' and attr[0] != 'rGripper':
    #     x_vel_inds = attr[1] + cur_x_ind
    #     cur_x_ind = x_vel_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr[0]+'__vel')] = x_vel_inds
    # for attr in ee_attrs:
    #     if attr not in u_attrs: continue
    #     x_vel_inds = np.array([0, 1, 2]) + cur_x_ind
    #     cur_x_ind = x_vel_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr+'__vel')] = x_vel_inds

    # for attr in robot_attr_map:
    #     if (robot.name, attr[0]) in params_to_x_inds: continue
    #     x_inds = attr[1] + cur_x_ind
    #     cur_x_ind = x_inds[-1] + 1
    #     params_to_x_inds[(robot.name, attr[0])] = x_inds

    for param in params:
        param_attr_map = const.ATTR_MAP[param._type]
        # Uses all parameters for state unless otherwise specified

        for attr in param_attr_map:
            # TODO: Remove special case for basket
            if (param.name, attr[0]) in params_to_x_inds or param.is_symbol() or (attr[0] not in x_attrs and param.name != 'basket'): continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds

    symbolic_boundary = cur_x_ind # Used to differntiate parameters from symbols in the state vector

    for param in plan.params.values():
        if not param.is_symbol(): continue
        param_attr_map = const.ATTR_MAP[param._type]
        for attr in param_attr_map:
            if (param.name, attr[0]) in params_to_x_inds: continue
            inds = attr[1] + cur_x_ind
            cur_x_ind = inds[-1] + 1
            params_to_x_inds[(param.name, attr[0])] = inds

    # dX, state index map, dU, (policy) action map
    return cur_x_ind, params_to_x_inds, cur_u_ind, params_to_u_inds, symbolic_boundary   

def fill_vector(params, params_to_inds, vec, t, use_symbols=False):
    for param_name, attr in params_to_inds:
        inds = params_to_inds[(param_name, attr)]
        param = params[param_name]
        if param.is_symbol():
            if not use_symbols: continue
            vec[inds] = getattr(param, attr)[:, 0].copy()
        else:
            vec[inds] = getattr(param, attr)[:, t].copy()       
       
def set_params_attrs(params, params_to_inds, vec, t, use_symbols=False):
    for param_name, attr in params_to_inds:
        inds = params_to_inds[(param_name, attr)]
        param = params[param_name]
        if param.is_symbol():
            if not use_symbols: continue
            getattr(param, attr)[:, 0] = vec[params_to_inds[(param.name, attr)]]
        else:
            getattr(param, attr)[:, t] = vec[params_to_inds[(param.name, attr)]]               

def fill_sample_from_trajectory(sample, plan, u_vec, noise, t, dX):
    active_ts, params = get_plan_traj_info(plan)

    sample.set(ACTION_ENUM, u_vec, t-active_ts[0])

    X = np.zeros((plan.dX,))
    fill_vector(params, plan.state_inds, X, t)
    sample.set(STATE_ENUM, X, t-active_ts[0])

    sample.set(NOISE_ENUM, noise, t-active_ts[0])

def fill_trajectory_from_sample(sample, plan, time_interval=POLICY_STEPS_PER_SECOND):
    params = set()
    for action in plan.actions:
        params.update(action.params)
    params = filter(lambda p: not p.is_symbol(), list(params))
    active_ts = (0, plan.horizon-1) # (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
    for t in range(active_ts[0], active_ts[1]):
        X = sample.get_X(t*time_interval)
        set_params_attrs(params, plan.state_inds, X, t)
    X = sample.get_X(active_ts[1]*time_interval-1)
    set_params_attrs(params, plan.state_inds, X, active_ts[1])

def get_trajectory_cost(plan, init_t, final_t, time_interval=POLICY_STEPS_PER_SECOND):
    '''
    Calculates the constraint violations at the provided timestep for the current trajectory, as well as the first and second order approximations.
    This function handles the hierarchies of mappings from parameters to attributes to indices and translates between how the predicates consturct
    those hierachies and how the policy states & actions 

    state_inds & action_inds map (param. attr_name) to the the relevant indices
    '''
    preds = []
    state_inds = plan.state_inds
    action_inds = plan.action_inds
    dX = plan.symbolic_bound
    dU = plan.dU
    for action in plan.actions:
        preds.extend(action.preds)
    active_ts = (init_t, final_t)
    T = active_ts[1] - active_ts[0]

    timestep_costs = np.zeros((T*time_interval, )) # l
    first_order_x_approx = np.zeros((T*time_interval, dX, )) # lx
    first_order_u_approx = np.zeros((T*time_interval, dU, )) # lu
    second_order_xx_approx = np.zeros((T*time_interval, dX, dX)) # lxx
    second_order_uu_approx = np.zeros((T*time_interval, dU, dU)) # luu
    second_order_ux_approx = np.zeros((T*time_interval, dU, dX)) # lux

    pred_param_attr_inds = {}
    for p in preds:
        if p['pred'] in pred_param_attr_inds: continue
        pred_param_attr_inds[p['pred']] = {}
        attr_inds = p['pred'].attr_inds
        cur_ind = 0
        for param in attr_inds:
            pred_param_attr_inds[p['pred']][param.name] = {}
            for attr_name, inds in attr_inds[param]:
                pred_param_attr_inds[p['pred']][param.name][attr_name] = np.array(range(cur_ind, cur_ind+len(inds)))
                cur_ind += len(inds)

    for t in range(active_ts[0], active_ts[1]-1):
        active_preds = plan.get_active_preds(t+1)
        preds_checked = []
        for p in preds:
            if p['pred'].__class__ not in INCLUDE_PREDS: continue
            if p['pred'] not in active_preds or p['pred'] in preds_checked: continue
            if p['pred'].active_range[1] > active_ts[1]: continue
            attr_inds = p['pred'].attr_inds
            comp_expr = p['pred'].get_expr(negated=p['negated'])

            # Constant terms
            expr = comp_expr.expr if comp_expr else None
            if not expr: continue
            param_vector = p['pred'].get_param_vector(t+1)
            param_vector[np.where(np.isnan(param_vector))] = 0
            # if np.any(np.isnan(param_vector)):
            #     import ipdb; ipdb.set_trace()

            cost_vector = expr.eval(param_vector)
            param_attr_inds = pred_param_attr_inds[p['pred']]
            time_ind = (t-active_ts[0])*time_interval
            timestep_costs[time_ind:time_ind+time_interval] -= np.sum(cost_vector)

            # if np.any(np.isnan(cost_vector)):
            #     import ipdb; ipdb.set_trace()

            # if hasattr(expr, '_grad') and expr._grad:
            #     # Linear terms
            #     first_degree_convexification = expr._grad(param_vector) # expr.convexify(param_vector, degree=1).eval(param_vector)
            #     try:
            #         for param in param_attr_inds:
            #             if plan.params[param].is_symbol(): continue
            #             for attr_name in param_attr_inds[param]:
            #                 if (param, attr_name) in state_inds:
            #                     first_order_x_approx[t-active_ts[0], state_inds[(param, attr_name)]] += np.sum(first_degree_convexification[:, param_attr_inds[param][attr_name]], axis=0)
            #                 if (param, attr_name) in action_inds:
            #                     first_order_u_approx[t-active_ts[0], action_inds[(param, attr_name)]] += np.sum(first_degree_convexification[:, param_attr_inds[param][attr_name]], axis=0)
            #     except Exception as e:
            #         import ipdb; ipdb.set_trace()
            # if hasattr(expr, '_hess') and expr._hess:
            #     # Quadratic terms
            #     try:
            #         second_degree_convexification = expr._hess(param_vector) # expr.convexify(param_vector, degree=2).eval(param_vector)
            #         for param_1 in param_attr_inds:
            #             for param_2 in param_attr_inds:
            #                 if plan.params[param_1].is_symbol() or plan.params[param_2].is_symbol(): continue
            #                 for attr_name_1 in param_attr_inds[param_1]:
            #                     for attr_name_2 in param_attr_inds[param2]:
            #                         if (param_1, attr_name_1) in state_inds and (param_2, attr_name_2) in state_inds:
            #                             x_inds_1 = state_inds[(param_1, attr_name_1)]
            #                             x_inds_2 = state_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(x_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
            #                             second_order_xx_approx[t-active_ts[0], x_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            #                         if (param_1, attr_name_1) in action_inds and (param_2, attr_name_2) in action_inds:
            #                             u_inds_1 = action_inds[(param_1, attr_name_1)]
            #                             u_inds_2 = action_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(u_inds_1) == len(pred_inds_1) and len(u_inds_2) == len(pred_inds_2)
            #                             second_order_uu_approx[t-active_ts[0], u_inds_1, u_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]

            #                         if (param_1, attr_name_1) in action_inds and (param_2, attr_name_2) in state_inds:
            #                             u_inds_1 = action_inds[(param_1, attr_name_1)]
            #                             x_inds_2 = state_inds[(param_2, attr_name_2)]
            #                             pred_inds_1 = param_attr_inds[param_1][attr_name_1]
            #                             pred_inds_2 = param_attr_inds[param_2][attr_name_2]
            #                             assert len(u_inds_1) == len(pred_inds_1) and len(x_inds_2) == len(pred_inds_2)
            #                             second_order_ux_approx[t-active_ts[0], u_inds_1, x_inds_2] += second_degree_convexification[pred_inds_1, pred_inds_2]
            #     except:
            #         import ipdb; ipdb.set_trace()
            preds_checked.append(p['pred'])

    return timestep_costs, first_order_x_approx, first_order_u_approx, second_order_xx_approx, second_order_uu_approx, second_order_ux_approx

# def map_trajectory_to_vel_acc(plan):
#     '''
#         Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
#     '''
#     params = get_action_params(plan)
#     active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
#     T = active_ts[1] - active_ts[0] + 1
#     vels = np.zeros((plan.dU, T))
#     accs = np.zeros((plan.dU, T))

#     a = np.zeros((plan.dU,))
#     v = np.zeros((plan.dU,))
#     for t in range(active_ts[0], active_ts[1]):
#         U_0 = np.zeros((plan.dU,))
#         U = np.zeros((plan.dU,))
#         fill_vector(params, plan.action_inds, U_0, t)
#         fill_vector(params, plan.action_inds, U, t+1)
#         real_t = plan.time[0, t]
#         vels[:, t-active_ts[0]] = v
#         a = 2*(U-U_0-v*real_t) / (real_t**2)
#         accs[:, t-active_ts[0]] = a
#         v = v + a*real_t
#     vels[:, active_ts[1]-active_ts[0]] = v + a*plan.time[0, active_ts[1]]

#     return vels, accs


# def timestep_vel_acc(plan, t, int_vel, real_ts_offset):
#     '''
#         Perform basic kienmatic calculations to find the velocity and acceleration of each joint at each timestep
#     '''
#     assert t < plan.T
#     params = get_action_params(plan)
#     a = np.zeros((plan.dU,))
#     v = int_vel.copy()
#     U_0 = np.zeros((plan.dU,))
#     U = np.zeros((plan.dU,))
#     fill_vector(params, plan.action_inds, U_0, t)
#     fill_vector(params, plan.action_inds, U, t+1)
#     real_t = plan.time[0, t] - real_ts_offset
#     vels[:, t-active_ts[0]] = v
#     a = 2*(U-U_0-v*real_t) / (real_t**2)

#     return a

# def get_plan_traj_info(plan):
#     '''
#         Extract active timesteps and active parameters from the plan
#     '''
#     active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
#     # if hasattr(plan, 'state_inds'):
#     #     params = list(set(map(lambda k: k[0], plan.state_inds.keys())))
#     # else:
#     #     params = plan.params.values()
#     params = get_state_params(plan)
#     return active_ts, params

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

def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose