from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import robot_predicates
from core.util_classes.items import Item
from core.util_classes.robots import Robot
import core.util_classes.transform_utils as T

from functools import reduce
import core.util_classes.baxter_constants as const
from collections import OrderedDict
from sco.expr import Expr
import math
import numpy as np
import pybullet as p
PI = np.pi
DEBUG = False

#These functions are helper functions that can be used by many robots
#@profile
def get_random_dir():
    """
        This helper function generates a random 2d unit vectors
    """
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir/np.linalg.norm(rand_dir)
    return rand_dir

#@profile
def get_random_theta():
    """
        This helper function generates a random angle between -PI to PI
    """
    theta =  2*PI*np.random.rand(1) - PI
    return theta[0]

#@profile
def smaller_ang(x):
    """
        This helper function takes in an angle in radius, and returns smaller angle
        Ex. 5pi/2 -> PI/2
            8pi/3 -> 2pi/3
    """
    return (x + PI)%(2*PI) - PI

#@profile
def closer_ang(x,a,dir=0):
    """
        find angle y (==x mod 2*PI) that is close to a
        dir == 0: minimize absolute value of difference
        dir == 1: y > x
        dir == 2: y < x
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*PI)
    elif dir == -1:
        return a + (x-a)%(2*PI) - 2*PI

#@profile
def closest_arm_pose(arm_poses, cur_arm_pose):
    """
        Given a list of possible arm poses, select the one with the least displacement from current arm pose
    """
    min_change = np.inf
    chosen_arm_pose = arm_poses[0] if len(arm_poses) else None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

#@profile
def closest_base_poses(base_poses, robot_base):
    """
        Given a list of possible base poses, select the one with the least displacement from current base pose
    """
    val, chosen = np.inf, robot_base
    if len(base_poses) <= 0:
        return chosen
    for base_pose in base_poses:
        diff = base_pose - robot_base
        distance = reduce(lambda x, y: x**2 + y, diff, 0)
        if distance < val:
            chosen = base_pose
            val = distance
    return chosen

#@profile
def lin_interp_traj(start, end, time_steps):
    """
    This helper function returns a linear trajectory from start pose to end pose
    """
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps+1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps+1)
        traj[i, :] = traj_row
    return traj

#@profile
def get_expr_mult(coeff, expr):
    """
        Multiply expresions with coefficients
    """
    new_f = lambda x: coeff*expr.eval(x)
    new_grad = lambda x: coeff*expr.grad(x)
    return Expr(new_f, new_grad)

# Sample base values to face the target
#@profile
def sample_base(target_pose, base_pose):
    vec = target_pose[:2] - np.zeros((2,))
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    return theta

#@profile
def process_traj(raw_traj, timesteps):
    """
        Process raw_trajectory so that it's length is desired timesteps
        when len(raw_traj) > timesteps
            sample Trajectory by space to reduce trajectory size
        when len(raw_traj) < timesteps
            append last timestep pose util the size fits

        Note: result_traj includes init_dof and end_dof
    """
    result_traj = []
    if len(raw_traj) == timesteps:
        result_traj = raw_traj.copy()
    else:
        traj_arr = [0]
        result_traj.append(raw_traj[0])
        #calculate accumulative distance
        for i in range(len(raw_traj)-1):
            traj_arr.append(traj_arr[-1] + np.linalg.norm(raw_traj[i+1] - raw_traj[i]))
        step_dist = traj_arr[-1]/(timesteps - 1)
        process_dist, i = 0, 1
        while i < len(traj_arr)-1:
            if traj_arr[i] == process_dist + step_dist:
                result_traj.append(raw_traj[i])
                process_dist += step_dist
            elif traj_arr[i] < process_dist+step_dist < traj_arr[i+1]:
                dist = process_dist+step_dist - traj_arr[i]
                displacement = (raw_traj[i+1] - raw_traj[i])/(traj_arr[i+1]-traj_arr[i])*dist
                result_traj.append(raw_traj[i]+displacement)
                process_dist += step_dist
            else:
                i += 1
    result_traj.append(raw_traj[-1])
    return np.array(result_traj).T


#@profile
def resample_pred(pred, negated, t, plan):
    res, attr_inds = [], OrderedDict()
    # Determine which action failed first
    rs_action, ref_index = None, None
    for i in range(len(plan.actions)):
        active = plan.actions[i].active_timesteps
        if active[0] <= t <= active[1]:
            rs_action, ref_index = plan.actions[i], i
            break

    if rs_action.name == 'moveto' or rs_action.name == 'movetoholding':
        return resample_move(plan, t, pred, rs_action, ref_index)
    elif rs_action.name == 'grasp' or rs_action.name == 'putdown':
        return resample_pick_place(plan, t, pred, rs_action, ref_index)
    else:
        raise NotImplemented

#@profile
def resample_move(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    robot.openrave_body.set_dof({'rGripper': 0.02})

    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    sampling_trace = None
    #rs_param is pdp_target0
    rs_param = rs_action.params[2]
    if ref_index + 1 < len(plan.actions):
        # take next action's ee_pose and find it's ik value.
        # ref_param is ee_target0
        ref_action = plan.actions[ref_index + 1]
        ref_range = ref_action.active_timesteps
        ref_param = ref_action.params[4]
        event_timestep = (ref_range[1] - ref_range[0])/2
        pose = robot.pose[:, event_timestep]

        arm_pose = get_ik_from_pose(ref_param.value, ref_param.rotation, body, manip_name)
        # In the case ee_pose wasn't even feasible, resample other preds
        if arm_pose is None:
            return None, None

        sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'rArmPose': arm_pose, 'value': pose}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
        add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('rArmPose', arm_pose), ('value', pose)])

    else:
        arm_pose = rs_action.params[2].rArmPose[:,0].flatten()
        pose = rs_action.params[2].value[:,0]

    """Resample Trajectory by BiRRT"""
    init_arm_dof = rs_action.params[1].rArmPose[:,0].flatten()
    init_pose_dof = rs_action.params[1].value[:,0]
    init_dof = np.hstack([init_pose_dof, init_arm_dof])
    end_dof = np.hstack([pose, arm_pose])

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None and sampling_trace != None:
        # In the case resampled poses resulted infeasible rrt trajectory
        sampling_trace['reward'] = -1
        plan.sampling_trace.append(sampling_trace)
        return np.array(res), attr_inds
    elif raw_traj == None and sampling_trace == None:
        # In the case resample is just not possible, resample other preds
        return None, None
    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, act_range[1] - act_range[0] + 2).T[1:-1]
    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(act_range[0] + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1
    return np.array(res), attr_inds

#@profile
def resample_pick_place(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    #rs_param is ee_poses, ref_param is target
    rs_param = rs_action.params[4]
    ref_param = rs_action.params[2]
    ee_poses = get_ee_from_target(ref_param.value, ref_param.rotation)
    for samp_ee in ee_poses:
        arm_pose = get_ik_from_pose(samp_ee[0].flatten(), samp_ee[1].flatten(), body, manip_name)
        if arm_pose is not None:
            break

    if arm_pose is None:
        return None, None
    sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'value': samp_ee[0], 'rotation': samp_ee[1]}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
    add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('value', samp_ee[0].flatten()), ('rotation', samp_ee[1].flatten())])

    """Resample Trajectory by BiRRT"""
    if t < (act_range[1] - act_range[0])/2 and ref_index >= 1:
        # if resample time occured before grasp or putdown.
        # resample initial poses as well
        # ref_action is move
        ref_action = plan.actions[ref_index - 1]
        ref_range = ref_action.active_timesteps

        start_pose = ref_action.params[1]
        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - ref_range[0] + 2

        init_timestep = ref_range[0]

    else:
        start_pose = rs_action.params[3]

        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - act_range[0] + 2

        init_timestep = act_range[0]

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None:
        # In the case resampled poses resulted infeasible rrt trajectory
        plan.sampling_trace.append(sampling_trace)
        plan.sampling_trace[-1]['reward'] = -1
        return np.array(res), attr_inds

    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, timesteps).T[1:-1]

    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1

    if init_timestep != act_range[0]:
        sampling_trace['data'][start_pose.name] = {'type': start_pose.get_type(), 'rArmPose': robot.rArmPose[:, act_range[0]], 'value': robot.pose[:, act_range[0]]}
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, start_pose, [('rArmPose', sampling_trace['data'][start_pose.name]['rArmPose']), ('value', sampling_trace['data'][start_pose.name]['value'])])

    return np.array(res), attr_inds

def resample_gripper_down_rot(pred, negated, t, plan, arms=[]):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot = pred.robot
    if not len(arms): arms = pred.arms
    geom = robot.geom
    axis = pred.axis
    pose = robot.openrave_body.param_fwd_kinematics(robot, arms, t)
    iks = {}
    robot.openrave_body.set_dof({pred.arm: np.zeros(len(robot.geom.jnt_names[pred.arm]))})
    for arm in arms:
        ee_name = geom.ee_link_names[arm]
        ee_link = geom.get_ee_link(arm)
        pos = pose[arm]['pos']
        quat = pred.quats[arm]
        iks[arm] = np.array(robot.openrave_body.get_ik_from_pose(pos, quat, arm))

    add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, iks[arm]) for arm in arms])
    return res, attr_inds


#@profile
def resample_obstructs(pred, negated, t, plan):
    # viewer = OpenRAVEViewer.create_viewer(plan.env)
    attr_inds, res = OrderedDict(), OrderedDict()
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] < t and  t <= act.active_timesteps[1]][0]

    robot, obstacle = pred.robot, pred.obstacle
    rave_body, obs_body = robot.openrave_body, obstacle.openrave_body
    r_geom, obj_geom = rave_body._geom, obs_body._geom
    dof_map = {arm: getattr(robot, arm)[:,t] for arm in r_geom.arms}
    for gripper in r_geom.grippers: dof_map[gripper] = getattr(robot, gripper)[:,t]
    rave_body.set_pose(robot.pose[:,t], robot.rotation[:,t])

    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot:
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    collisions = p.getClosestPoints(rave_body.body_id, obs_body.body_id, 0.01)
    arm = None
    for col in collisions:
        r_link, obj_link = col[3], col[4]
        for a in r_geom.arms:
            if r_link in r_geom.get_arm_inds(a) or \
               r_link in r_geom.gripper_inds['{}_gripper'.format(a)] or \
               r_link-1 in r_geom.get_arm_inds(a) or \
               r_link-1 in r_geom.gripper_inds['{}_gripper'.format(a)]:
                arm = a
                break
        if arm is not None: break
    if arm is None:
        return None, None

    ee_link = r_geom.get_ee_link(arm)
    info = rave_body.fwd_kinematics(arm)
    ee_pos, orn = info['pos'], info['quat']
    
    attempt, step = 0, 1
    new_pos = None
    while attempt < 15 and len(collisions) > 0:
        attempt += 1
        target_ee = ee_pos + step * np.multiply(np.random.sample(3), const.RESAMPLE_FACTOR)
        rave_body.set_dof(dof_map)
        arm_pose = rave_body.get_ik_from_pose(target_ee, orn, arm)
        step += 1
        rave_body.set_dof({arm: arm_pose})
        collisions = p.getClosestPoints(rave_body.body_id, obs_body.body_id, 0.01)
        link_f = lambda col: col[3]-1 in r_geom.arm_inds[arm] or col[3]-1 in r_geom.gripper_inds['{}_gripper'.format(arm)]
        collisions = list(filter(link_f, collisions))
        if not len(collisions):
            add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, arm_pose)])
            new_pos = arm_pose
            break

    #if not const.PRODUCTION:
    #    print("resampling at {} action".format(action.name))
    act_start, act_end = action.active_timesteps
    res_arm = arm
    if new_pos is None:
        return None, None

    N_STEPS = 3
    act_start = np.maximum(act_start+1, t-N_STEPS)
    act_end = np.minimum(act_end-1, t+N_STEPS)
    if True or action.name.find("moveto") >=0  or action.name.find("moveholding") >= 0:
        timesteps_1 = t - act_start
        pose_traj_1 = lin_interp_traj(robot.pose[:, act_start], robot.pose[:, t], timesteps_1)
        old_traj = getattr(robot, arm)
        arm_traj_1 = lin_interp_traj(old_traj[:, act_start], new_pos, timesteps_1)
        for i in range(act_start+1, t):
            traj_ind = i - act_start
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('pose', pose_traj_1[:, traj_ind])])
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj_1[:, traj_ind])])

        timesteps_2 = act_end - t
        arm_traj_2 = lin_interp_traj(new_pos, old_traj[:, act_end], timesteps_2)
        pose_traj_2 = lin_interp_traj(robot.pose[:, t], robot.pose[:, act_end], timesteps_2)
        for i in range(t+1, act_end):
            traj_ind = i - t
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('pose', pose_traj_2[:, traj_ind])])
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj_2[:, traj_ind])])

    return res, attr_inds

#@profile
def resample_rcollides(pred, negated, t, plan):
    # Variable that needs to added to BoundExpr and latter pass to the planner
    JOINT_STEP = 20
    STEP_DECREASE_FACTOR = 1.5
    ATTEMPT_SIZE = 7
    LIN_SAMP_RANGE = 5

    attr_inds = OrderedDict()
    res = OrderedDict()
    robot, rave_body = pred.robot, pred._param_to_body[pred.robot]
    body = rave_body.env_body
    manip = body.GetManipulator("right_arm")
    arm_inds = manip.GetArmIndices()
    lb_limit, ub_limit = body.GetDOFLimits()
    step_factor = JOINT_STEP
    joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
    original_pose, arm_pose = robot.rArmPose[:, t].copy(), robot.rArmPose[:, t].copy()
    rave_body.set_pose([0,0,robot.pose[:, t]])
    rave_body.set_dof({"lArmPose": robot.lArmPose[:, t].flatten(),
                       "lGripper": robot.lGripper[:, t].flatten(),
                       "rArmPose": robot.rArmPose[:, t].flatten(),
                       "rGripper": robot.rGripper[:, t].flatten()})

    ## Determine the range we should resample
    pred_list = [act_pred['active_timesteps'] for act_pred in plan.actions[0].preds if act_pred['pred'].spacial_anchor == True]
    start, end = 0, plan.horizon-1
    for action in plan.actions:
        if action.active_timesteps[0] <= t and action.active_timesteps[1] > t:
            for act_pred in plan.actions[0].preds:
                if act_pred['pred'].spacial_anchor == True:
                    if act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0] > t:
                        end = min(end, act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0])
                    if act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1] < t:
                        start = max(start, act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1])

    desired_end_pose = robot.rArmPose[:, end]
    current_end_pose = robot.rArmPose[:, t]
    col_report = CollisionReport()
    collisionChecker = RaveCreateCollisionChecker(plan.env,'pqp')
    count = 1
    while (body.CheckSelfCollision() or
           collisionChecker.CheckCollision(body, report=col_report) or
           col_report.minDistance <= pred.dsafe):
        step_sign = np.ones(len(arm_inds))
        step_sign[np.random.choice(len(arm_inds), len(arm_inds)/2, replace=False)] = -1
        # Ask in collision pose to randomly move a step, hopefully out of collision
        arm_pose = original_pose + np.multiply(step_sign, joint_step)
        rave_body.set_dof({"rArmPose": arm_pose})
        # arm_pose = body.GetActiveDOFValues()[arm_inds]
        if not count % ATTEMPT_SIZE:
            step_factor = step_factor/STEP_DECREASE_FACTOR
            joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
        count += 1

        # For Debug
        rave_body.set_pose([0,0,robot.pose[:, t]])
    add_to_attr_inds_and_res(t, attr_inds, res, robot,[('rArmPose', arm_pose)])
    robot._free_attrs['rArmPose'][:, t] = 0


    start, end = max(start, t-LIN_SAMP_RANGE), min(t+LIN_SAMP_RANGE, end)
    rcollides_traj = np.hstack([lin_interp_traj(robot.rArmPose[:, start], arm_pose, t-start), lin_interp_traj(arm_pose, robot.rArmPose[:, end], end - t)[:, 1:]]).T
    i = start + 1
    for traj in rcollides_traj[1:-1]:
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [('rArmPose', traj)])
        i +=1


    return np.array(res), attr_inds

#@profile
def sample_arm_pose(robot_body, old_arm_pose=None):
    dof_inds = robot_body.GetManipulator("right_arm").GetArmIndices()
    lb_limit, ub_limit = robot_body.GetDOFLimits()
    active_ub = ub_limit[dof_inds].flatten()
    active_lb = lb_limit[dof_inds].flatten()
    if old_arm_pose is not None:
        arm_pose = np.random.random_sample((len(dof_inds),)) - 0.5
        arm_pose = np.multiply(arm_pose, (active_ub - active_lb)/5) + old_arm_pose
    else:
        arm_pose = np.random.random_sample((len(dof_inds),))
        arm_pose = np.multiply(arm_pose, active_ub - active_lb) + active_lb
    return arm_pose

#@profile
def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    # param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]

#@profile
def test_resample_order(attr_inds, res):
    for p in attr_inds:
        i = 0
        for attr, inds, t in attr_inds[p]:
            if not np.allclose(getattr(p, attr)[inds, t], res[p][i:i+len(inds)]):
                print(getattr(p, attr)[inds, t])
                print("v.s.")
                print(res[p][i:i+len(inds)])
            i += len(inds)


#@profile
def resample_eereachable(pred, negated, t, plan, inv=False, use_pos=True, use_rot=True, rel=True):
    attr_inds, res = OrderedDict(), OrderedDict()
    acts = [a for a in plan.actions if a.active_timesteps[0] < t and a.active_timesteps[1] >= t]
    if not len(acts): return None, None
    x = pred.get_param_vector(t)
    obj_trans, robot_trans, axises, arm_joints = pred.robot_obj_kinematics(x)
    robot, robot_body = pred.robot, pred._param_to_body[pred.robot]

    act = acts[0]
    a_st, a_et = act.active_timesteps

    if hasattr(pred, 'obj'):
        targ_pos = pred.obj.pose[:,t].copy()
        targ_rot = pred.obj.rotation[:,t].copy()
    elif hasattr(pred, 'targ'):
        targ_pos = pred.targ.value[:,0].copy()
        targ_rot = pred.targ.rotation[:,0].copy()

    arm = pred.arm
    targ_quat = T.euler_to_quaternion(targ_rot, 'xyzw')
    gripper_axis = robot.geom.get_gripper_axis(pred.arm)
    quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, pred.axis)
    robot_mat = T.quat2mat(quat)
    obj_mat = T.quat2mat(targ_quat)
    quat = T.mat2quat(obj_mat.dot(robot_mat))
    robot_body.set_pose(robot.pose[:,t], robot.rotation[:,t])
    robot_body.set_dof({arm: getattr(robot, arm)[:,t]})
    cur_info = robot_body.fwd_kinematics(arm)
    cur_pos, cur_quat = cur_info['pos'], cur_info['quat']

    st, et = pred.active_range
    base_targ_pos = targ_pos
    p_st, p_et = max(a_st, t+st), min(a_et, t+et+1)
    for ts in range(p_st, p_et):
        if use_pos:
            #dist = pred.approach_dist if ts <= t else pred.retreat_dist
            #vec = -np.array(pred.rel_pt) - dist * np.abs(t-ts) * pred.axis
            vec = np.array(pred.rel_pt) + pred.get_rel_pt(ts-t)
            if rel:
                #vec = obj_mat.dot(vec)
                vec = obj_trans[:3,:3].dot(vec)
            targ_pos = base_targ_pos+vec
        else:
            targ_pos = np.array(cur_pos)

        quat = quat if use_rot else cur_quat
        ik = robot_body.get_ik_from_pose(targ_pos, quat, arm)
        add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(arm, np.array(ik).flatten())])
    return res, attr_inds


#@profile
def resample_in_gripper(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()
    acts = [a for a in plan.actions if a.active_timesteps[0] < t and a.active_timesteps[1] >= t]
    if not len(acts): return None, None
    x = pred.get_param_vector(t)
    obj_trans, robot_trans, axises, arm_joints = pred.robot_obj_kinematics(x)
    robot, robot_body = pred.robot, pred._param_to_body[pred.robot]
    obj, obj_body = pred.obj, pred._param_to_body[pred.obj]

    act = acts[0]
    a_st, a_et = act.active_timesteps

    arm = pred.arm
    targ_quat = T.euler_to_quaternion(targ_rot, 'xyzw')
    gripper_axis = robot.geom.get_gripper_axis(pred.arm)
    quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, pred.axis)
    robot_mat = T.quat2mat(quat)
    obj_mat = T.quat2mat(targ_quat)
    quat = T.mat2quat(obj_mat.dot(robot_mat))
    robot_body.set_pose(robot.pose[:,t], robot.rotation[:,t])
    robot_body.set_dof({arm: getattr(robot, arm)[:,t]})
    cur_info = robot_body.fwd_kinematics(arm)
    cur_pos, cur_quat = cur_info['pos'], cur_info['quat']

    st, et = pred.active_range
    for ts in range(max(a_st, t+st), min(a_et-1, t+et)):
        if use_pos:
            dist = pred.approach_dist if ts <= t else pred.retreat_dist
            vec = -pred.rel_pt - dist * np.abs(t-ts) * pred.axis
            mask = pred.mask
            if rel:
                vec = obj_mat.dot(vec)
                mask = obj_mat.dot(mask)
            targ_pos = targ_pos+vec
            for ind, val in enumerate(mask):
                if np.abs(val) < 1e-1:
                    targ_pos[ind] = cur_pos[ind]
        else:
            targ_pos = np.array(cur_pos)

        quat = quat if use_rot else cur_quat
        ik = robot_body.get_ik_from_pose(targ_pos, quat, arm)
        add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(arm, np.array(ik).flatten())])
    return res, attr_inds
