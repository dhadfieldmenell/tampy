from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import (
    matrixFromAxisAngle,
    IkParameterization,
    IkParameterizationType,
    IkFilterOptions,
)
import core.util_classes.pr2_constants as const
from sco.expr import Expr
import math
import numpy as np
from functools import reduce

PI = np.pi


def get_random_dir():
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    return rand_dir


def get_random_theta():
    theta = 2 * PI * np.random.rand(1) - PI
    return theta[0]


def sample_base_pose(target_pose, base_pose_seed=None, dist=const.DEFAULT_DIST):
    rand_dir = get_random_dir()
    bp = rand_dir * dist + target_pose[:2]

    vec = target_pose[:2] - bp
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    if base_pose_seed is not None:
        theta = closer_ang(theta, base_pose_seed[2])
    pose = np.array([bp[0], bp[1], theta])
    return pose


def get_ee_transform_from_pose(pose, rotation):
    ee_trans = OpenRAVEBody.transform_from_obj_pose(pose, rotation)
    # the rotation is to transform the tool frame into the end effector transform
    rot_mat = matrixFromAxisAngle([0, PI / 2, 0])
    ee_rot_mat = ee_trans[:3, :3].dot(rot_mat[:3, :3])
    ee_trans[:3, :3] = ee_rot_mat
    return ee_trans


def set_torso_and_arm_to_ik_soln(robot, torso_arm_pose, t):
    torso_pose = torso_arm_pose[:1]
    arm_pose = torso_arm_pose[1:]
    robot.rArmPose[:, t] = arm_pose[:]
    robot.backHeight[:, t] = torso_pose[:]


def get_torso_and_arm_pose_from_ik_soln(ik_solution):
    if ik_solution is None:
        return None, None
    torso_pose = ik_solution[:1]
    arm_pose = ik_solution[1:]
    return torso_pose, arm_pose


def smaller_ang(x):
    return (x + PI) % (2 * PI) - PI


def closer_ang(x, a, dir=0):
    """
    find angle y (==x mod 2*PI) that is close to a
    dir == 0: minimize absolute value of difference
    dir == 1: y > x
    dir == 2: y < x
    """
    if dir == 0:
        return a + smaller_ang(x - a)
    elif dir == 1:
        return a + (x - a) % (2 * PI)
    elif dir == -1:
        return a + (x - a) % (2 * PI) - 2 * PI


def closer_joint_angles(pos, seed):
    result = np.array(pos)
    for i in [2, 4, 6]:
        result[i] = closer_ang(pos[i], seed[i], 0)
    return result


# def closer_arm_pose(pos, seed):
#     assert pos.shape == (7,1)
#     assert seed.shape == (7,1)
#     arm_pose = np.zeros((7,1))
#
#     for i in [2,4,6]:
#         arm_pose[i] = closer_ang(pos[i],seed[i],0)
#     return arm_pose


def get_torso_arm_ik(robot_body, target_trans, old_arm_pose=None):
    manip = robot_body.env_body.GetManipulator("rightarm_torso")
    iktype = IkParameterizationType.Transform6D

    solution = manip.FindIKSolution(
        IkParameterization(target_trans, iktype), IkFilterOptions.CheckEnvCollisions
    )
    if solution is None:
        return None, None
    torso_pose, arm_pose = get_torso_and_arm_pose_from_ik_soln(solution)
    if old_arm_pose is not None:
        arm_pose = closer_joint_angles(arm_pose, old_arm_pose)
    return torso_pose, arm_pose


def get_col_free_base_pose_around_target(
    t, plan, target_pose, robot, callback=None, save=False, dist=const.DEFAULT_DIST
):
    base_pose = None
    old_base_pose = robot.pose[:, t].copy()
    for i in range(const.NUM_BASE_RESAMPLES):
        base_pose = sample_base_pose(
            target_pose, base_pose_seed=old_base_pose, dist=dist
        )
        robot.pose[:, t] = base_pose
        if callback is not None:
            callback()
        _, collision_preds = plan.get_param(
            "RCollides", 1, negated=True, return_preds=True
        )
        # check to ensure collision_preds are correct

        collision_free = True
        for pred in collision_preds:
            if not pred.test(t, negated=True):
                collision_free = False
                base_pose = None
                break
        if collision_free:
            break

    if not save:
        robot.pose[:, t] = old_base_pose
    return base_pose


def get_col_free_torso_arm_pose(
    t, pos, rot, robot_param, robot_body, arm_pose_seed=None, save=False, callback=None
):
    target_trans = get_ee_transform_from_pose(pos, rot)

    # save arm pose and back height
    old_arm_pose = robot_param.rArmPose[:, t].copy()
    old_back_height = robot_param.backHeight[:, t].copy()

    if arm_pose_seed is None:
        arm_pose_seed = old_arm_pose

    torso_pose, arm_pose = get_torso_arm_ik(
        robot_body, target_trans, old_arm_pose=arm_pose_seed
    )
    if torso_pose is not None:
        robot_param.rArmPose[:, t] = arm_pose
        robot_param.backHeight[:, t] = torso_pose
        if callback is not None:
            trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
            callback(trans)
            # callback(target_trans)

    # setting parameter values back
    robot_param.rArmPose[:, t] = old_arm_pose
    robot_param.backHeight[:, t] = old_back_height
    return torso_pose, arm_pose


def get_ee_from_target(targ_pos, targ_rot):
    """
    Sample all possible EE Pose that pr2 can grasp with

    target_pos: position of target we want to sample ee_pose form
    target_rot: rotation of target we want to sample ee_pose form
    return: list of ee_pose tuple in the format of (ee_pos, ee_rot) around target axis
    """
    possible_ee_poses = []
    ee_pos = targ_pos.copy()
    target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
    # rotate can's local z-axis by the amount of linear spacing between 0 to 2pi
    angle_range = np.linspace(0, PI * 2, num=const.EE_ANGLE_SAMPLE_SIZE)
    for rot in angle_range:
        target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
        # rotate new ee_pose around can's rotation axis
        rot_mat = matrixFromAxisAngle([0, 0, rot])
        ee_trans = target_trans.dot(rot_mat)
        ee_rot = OpenRAVEBody.obj_pose_from_transform(ee_trans)[3:]
        possible_ee_poses.append((ee_pos, ee_rot))
    return possible_ee_poses


def closest_arm_pose(arm_poses, cur_arm_pose):
    """
    Given a list of possible arm poses, select the one with the least change from current arm pose
    """
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_poses) ** 2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose


def get_base_poses_around_pos(t, robot, pos, sample_size, dist=const.DEFAULT_DIST):
    base_poses = []
    old_base_pose = robot.pose[:, t].copy()
    for i in range(sample_size):
        if np.any(old_base_pose):
            base_pose = sample_base_pose(pos.flatten(), dist=dist)
        else:
            base_pose = sample_base_pose(
                pos.flatten(), base_pose_seed=old_base_pose.flatten(), dist=dist
            )
        if base_pose is not None:
            base_poses.append(base_pose)
    return base_poses


def closest_base_poses(base_poses, robot_base):
    val, chosen = np.inf, None
    if len(base_poses) <= 0:
        return chosen
    for base_pose in base_poses:
        diff = base_pose - robot_base
        distance = reduce(lambda x, y: x ** 2 + y, diff, 0)
        if distance < val:
            chosen = base_pose
            val = distance
    return chosen


# Obtain constants for EEReachable in robot_predicates
# from core.util_classes.robot_predicates import OBJ_RING_SAMPLING_RADIUS, NUM_EEREACHABLE_RESAMPLE_ATTEMPTS
OBJ_RING_SAMPLING_RADIUS = 0.6
NUM_EEREACHABLE_RESAMPLE_ATTEMPTS = 10


def get_expr_mult(coeff, expr):
    new_f = lambda x: coeff * expr.eval(x)
    new_grad = lambda x: coeff * expr.grad(x)
    return Expr(new_f, new_grad)


# Nope
def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        res.extend(val[inds].flatten().tolist())
        param_attr_inds.append((attr_name, inds, t))
    if param in attr_inds:
        attr_inds[param].extend(param_attr_inds)
    else:
        attr_inds[param] = param_attr_inds


def set_robot_body_to_pred_values(pred, t):
    robot_body = pred._param_to_body[pred.robot]
    robot_body.set_pose(pred.robot.pose[:, t])
    robot_body.set_dof(
        pred.robot.backHeight[:, t],
        pred.robot.lArmPose[:, t],
        pred.robot.lGripper[:, t],
        pred.robot.rArmPose[:, t],
        pred.robot.rGripper[:, t],
    )


# Nope
def plot_transform(env, T, s=0.1):
    """
    Plots transform T in openrave environment.
    S is the length of the axis markers.
    """
    h = []
    x = T[0:3, 0]
    y = T[0:3, 1]
    z = T[0:3, 2]
    o = T[0:3, 3]
    h.append(
        env.drawlinestrip(
            points=np.array([o, o + s * x]),
            linewidth=3.0,
            colors=np.array([(1, 0, 0), (1, 0, 0)]),
        )
    )
    h.append(
        env.drawlinestrip(
            points=np.array([o, o + s * y]),
            linewidth=3.0,
            colors=np.array(((0, 1, 0), (0, 1, 0))),
        )
    )
    h.append(
        env.drawlinestrip(
            points=np.array([o, o + s * z]),
            linewidth=3.0,
            colors=np.array(((0, 0, 1), (0, 0, 1))),
        )
    )
    return h


def resample_bp_around_target(
    pred, t, plan, target_pose, dist=OBJ_RING_SAMPLING_RADIUS
):
    v = OpenRAVEViewer.create_viewer()

    bp = get_col_free_base_pose_around_target(
        t, plan, target_pose, pred.robot, save=True, dist=dist
    )
    v.draw_plan_ts(plan, t)

    attr_inds = OrderedDict()
    res = []
    robot_attr_name_val_tuples = [("pose", bp)]
    add_to_attr_inds_and_res(t, attr_inds, res, pred.robot, robot_attr_name_val_tuples)
    return np.array(res), attr_inds


def lin_interp_traj(start, end, time_steps):
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps + 1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps + 1)
        traj[i, :] = traj_row
    return traj


def ee_reachable_resample(pred, negated, t, plan):
    assert not negated
    handles = []
    v = OpenRAVEViewer.create_viewer()

    def target_trans_callback(target_trans):
        handles.append(plot_transform(v.env, target_trans))
        v.draw_plan_ts(plan, t)

    def plot_time_step_callback():
        v.draw_plan_ts(plan, t)

    plot_time_step_callback()

    targets = plan.get_param("GraspValid", 1, {0: pred.ee_pose})
    assert len(targets) == 1
    # confirm target is correct
    target_pose = targets[0].value[:, 0]
    set_robot_body_to_pred_values(pred, t)

    theta = 0
    robot = pred.robot
    robot_body = pred._param_to_body[robot]
    for _ in range(NUM_EEREACHABLE_RESAMPLE_ATTEMPTS):
        # generate collision free base pose
        base_pose = get_col_free_base_pose_around_target(
            t,
            plan,
            target_pose,
            robot,
            save=True,
            dist=OBJ_RING_SAMPLING_RADIUS,
            callback=plot_time_step_callback,
        )
        if base_pose is None:
            print("we should always be able to sample a collision-free base pose")
            st()
        # generate collision free arm pose
        target_rot = np.array([get_random_theta(), 0, 0])

        torso_pose, arm_pose = get_col_free_torso_arm_pose(
            t,
            target_pose,
            target_rot,
            robot,
            robot_body,
            save=True,
            arm_pose_seed=None,
            callback=target_trans_callback,
        )
        st()
        if torso_pose is None:
            print("we should be able to find an IK")
            continue

        # generate approach IK
        ee_trans = OpenRAVEBody.transform_from_obj_pose(target_pose, target_rot)
        rel_pt = pred.get_rel_pt(-pred._steps)
        target_pose_approach = np.dot(ee_trans, np.r_[rel_pt, 1])[:3]

        torso_pose_approach, arm_pose_approach = get_col_free_torso_arm_pose(
            t,
            target_pose_approach,
            target_rot,
            robot,
            robot_body,
            save=True,
            arm_pose_seed=arm_pose,
            callback=target_trans_callback,
        )
        st()
        if torso_pose_approach is None:
            continue

        # generate retreat IK
        ee_trans = OpenRAVEBody.transform_from_obj_pose(target_pose, target_rot)
        rel_pt = pred.get_rel_pt(pred._steps)
        target_pose_retreat = np.dot(ee_trans, np.r_[rel_pt, 1])[:3]

        torso_pose_retreat, arm_pose_retreat = get_col_free_torso_arm_pose(
            t,
            target_pose_retreat,
            target_rot,
            robot,
            robot_body,
            save=True,
            arm_pose_seed=arm_pose,
            callback=target_trans_callback,
        )
        st()
        if torso_pose_retreat is not None:
            break
    else:
        print("we should always be able to sample a collision-free base and arm pose")
        st()

    attr_inds = OrderedDict()
    res = []
    arm_approach_traj = lin_interp_traj(arm_pose_approach, arm_pose, pred._steps)
    torso_approach_traj = lin_interp_traj(torso_pose_approach, torso_pose, pred._steps)
    base_approach_traj = lin_interp_traj(base_pose, base_pose, pred._steps)

    arm_retreat_traj = lin_interp_traj(arm_pose, arm_pose_retreat, pred._steps)
    torso_retreat_traj = lin_interp_traj(torso_pose, torso_pose_retreat, pred._steps)
    base_retreat_traj = lin_interp_traj(base_pose, base_pose, pred._steps)

    arm_traj = np.hstack((arm_approach_traj, arm_retreat_traj[:, 1:]))
    torso_traj = np.hstack((torso_approach_traj, torso_retreat_traj[:, 1:]))
    base_traj = np.hstack((base_approach_traj, base_retreat_traj[:, 1:]))

    # add attributes for approach and retreat
    for ind in range(2 * pred._steps + 1):
        robot_attr_name_val_tuples = [
            ("rArmPose", arm_traj[:, ind]),
            ("backHeight", torso_traj[:, ind]),
            ("pose", base_traj[:, ind]),
        ]
        add_to_attr_inds_and_res(
            t + ind - pred._steps,
            attr_inds,
            res,
            pred.robot,
            robot_attr_name_val_tuples,
        )
    st()

    ee_pose_attr_name_val_tuples = [("value", target_pose), ("rotation", target_rot)]
    add_to_attr_inds_and_res(
        t, attr_inds, res, pred.ee_pose, ee_pose_attr_name_val_tuples
    )
    # v.draw_plan_ts(plan, t)
    v.animate_range(plan, (t - pred._steps, t + pred._steps))
    # check that indexes are correct
    # import ipdb; ipdb.set_trace()

    return np.array(res), attr_inds
