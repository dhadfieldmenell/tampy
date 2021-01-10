import numpy as np
import pybullet as p

import core.util_classes.common_constants as const
from core.util_classes.openrave_body import OpenRAVEBody


def vertical_gripper(robot, arm, gripper_open=True, rand=False):
    robot_body = robot.openrave_body
    robot_body.set_from_param(robot, start_ts)
    old_arm_pose = getattr(robot, arm)[:, start_ts].copy()
    obj = params[1]
    target_loc = obj.pose[:, start_ts] + np.array([0, 0, const.GRASP_DIST])
    gripper_axis = robot.geom.get_gripper_axis(arm)
    target_axis = [0, 0, -1]
    quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, target_axis)

    iks = []
    attempt = 0
    while not len(iks) and attempt < 20:
        if rand:
            target_loc += np.multiply(np.random.sample(3) - [0.5,0.5,0.5], [0.01, 0.01, 0])
        iks = robot_body.get_ik_from_pose(target_loc, quat, arm)
        rand = not len(iks)
        attempt += 1
    if not len(iks): return None

    arm_pose = robot_sampling.closest_arm_pose(iks, old_arm_pose.flatten()).reshape((-1,1))
    pose = {arm: arm_pose}
    gripper = robot.geom.get_gripper(arm)
    if gripper is not None:
        pose[gripper] = robot.geom.get_gripper_open_val() if gripper_open else robot.geom.get_gripper_close_val()
    for aux_arm in robot.geom.arms:
        if aux_arm == arm: continue
        old_pose = getattr(robot, aux_arm)[:, start_ts].copy()
        pose[aux_arm] = old_pose
        aux_gripper = robot.geom.get_gripper(aux_arm)
        if aux_gripper is not None:
            pose[aux_arm] = getattr(robot, aux_gripper)[:, start_ts].copy()
    return pose


def obj_pose_suggester(self, plan, anum, resample_size=20):
    robot_pose = []
    assert anum + 1 <= len(plan.actions)

    if anum + 1 < len(plan.actions):
	act, next_act = plan.actions[anum], plan.actions[anum+1]
    else:
	act, next_act = plan.actions[anum], None


    for i in range(resample_size):
        ### Cases for when behavior can be inferred from current action
        if act.name.lower() == 'grasp':
            robot = act.params[0]
            arm = robot.geoms.arms[0]
            pose = vertical_gripper(robot, arm, False, rand=(i>0))

        elif act.name.lower() == 'left_grasp':
            robot = act.params[0]
            pose = vertical_gripper(robot, 'left', False, rand=(i>0))

        elif act.name.lower() == 'right_grasp':
            robot = act.params[0]
            pose = vertical_gripper(robot, 'right', False, rand=(i>0))

        elif act.name.lower() == 'putdown':
            robot = act.params[0]
            arm = robot.geoms.arms[0]
            pose = vertical_gripper(robot, arm, True, rand=(i>0))

        elif act.name.lower() == 'left_putdown':
            robot = act.params[0]
            pose = vertical_gripper(robot, 'left', True, rand=(i>0))

        elif act.name.lower() == 'right_putdown':
            robot = act.params[0]
            pose = vertical_gripper(robot, 'right', True, rand=(i>0))

        ### Cases for when behavior cannot be inferred from current action
        elif next_act is None:
            pose = None

        elif next_act.name.lower() == 'grasp':
            robot = next_act.params[0]
            arm = robot.geoms.arms[0]
            pose = vertical_gripper(robot, arm, True, rand=(i>0))

        elif next_act.name.lower() == 'left_grasp':
            robot = next_act.params[0]
            pose = vertical_gripper(robot, 'left', True, rand=(i>0))

        elif next_act.name.lower() == 'right_grasp':
            robot = next_act.params[0]
            pose = vertical_gripper(robot, 'right', True, rand=(i>0))

        elif next_act.name.lower() == 'putdown':
            robot = next_act.params[0]
            arm = robot.geoms.arms[0]
            pose = vertical_gripper(robot, arm, False, rand=(i>0))

        elif next_act.name.lower() == 'left_putdown':
            robot = next_act.params[0]
            pose = vertical_gripper(robot, 'left', False, rand=(i>0))

        elif next_act.name.lower() == 'right_putdown':
            robot = next_act.params[0]
            pose = vertical_gripper(robot, 'right', False, rand=(i>0))

        if pose is None: break
        robot_pose.append(pose)

    return robot_pose

