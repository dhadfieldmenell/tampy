import numpy as np
from openravepy import (
    matrixFromAxisAngle,
    IkParameterization,
    IkParameterizationType,
    IkFilterOptions,
    Planner,
    RaveCreatePlanner,
    RaveCreateTrajectory,
    matrixFromAxisAngle,
    CollisionReport,
    RaveCreateCollisionChecker,
)

import core.util_classes.hsr_constants as constants
from pma import bactrack_ll_solver_gurobi


class HSRSolver(backtrack_ll_solver_gurobi.BacktrackLLSolver):
    def get_resample_param(self, a):
        if a.name == "moveto":
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == "moveholding_can":
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == "can_grasp":
            ## sample the grasp/grasp_pose
            rs_param = a.params[-1]
        elif a.name == "can_putdown":
            ## sample the end pose
            rs_param = a.params[-1]
        elif a.name == "stack_cans":
            rs_param = a.params[-1]
        else:
            raise NotImplemented

        return rs_param

    def obj_pose_suggester(self, plan, anum, resample_size=10):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum + 1]
        else:
            act, next_act = plan.actions[anum], None

        robot = plan.params["hsr"]
        robot_body = robot.openrave_body
        start_ts, end_ts = act.active_timesteps
        old_pose = robot.pose[:, start_ts].reshape((3, 1))
        robot_body.set_pose(old_pose[:, 0])
        for p in list(plan.params.values()):
            if not p.is_symbol() and "hsr" not in p.name:
                p.openrave_body.set_pose(p.pose[:, start_ts], p.rotation[:, start_ts])

        plan.params["can0"].openrave_body.set_pose([1.25, 0, 0.43])
        attempt_limit = 10
        col_report = CollisionReport()
        collisionChecker = RaveCreateCollisionChecker(plan.env, "pqp")

        for i in range(resample_size):
            if next_act != None and (
                next_act.name == "can_grasp" or next_act.name == "can_putdown"
            ):
                target = next_act.params[2]
                target_pos = target.value[:, 0]
                x, y, theta = target_pos
                if x == 0:
                    x = 0.001
                angle_offset = np.arctan(y / x)
                if x < old_pose[0]:
                    angle_offset += np.pi

                robot_pos = None
                attempt = 0
                while (
                    robot_pos is None
                    or collisionChecker.CheckCollision(
                        robot_body.env_body, report=col_report
                    )
                    or col_report.minDistance <= 0.01
                ):

                    robot_angle = np.random.uniform(
                        angle_offset - np.pi / 3, angle_offset + np.pi / 3
                    )
                    hand_angle = constants.GRIPPER_OFFSET_ANGLE + robot_angle
                    robot_x = x - np.cos(hand_angle) * constants.GRIPPER_OFFSET_DISP
                    robot_y = y - np.sin(hand_angle) * constants.GRIPPER_OFFSET_DISP
                    robot_pos = np.array([[robot_x], [robot_y], [robot_angle]])
                    robot_body.set_pose(robot_pos.flatten())

                    wrist_angle = hand_angle - target.rotation[2]
                    lift = np.maximum(0, target_pos[2] - 0.16 + 0.125)
                    arm = np.array(
                        [[lift], [-np.pi / 2], [0], [-np.pi / 2], [wrist_angle]]
                    )
                    robot_body.set_dof({"arm": arm.flatten()})

                    attempt += 1
                    if attempt > attempt_limit:
                        robot_pos = None
                        break

                if robot_pos is None:
                    continue
                robot_pose.append(
                    {
                        "value": robot_pos,
                        "gripper": np.array([[constants.GRIPPER_CLOSE]])
                        if next_act.name == "can_putdown"
                        else np.array([[constants.GRIPPER_OPEN]]),
                        "arm": arm,
                    }
                )
            elif act.name == "can_grasp" or act.name == "can_putdown":
                target = act.params[2]
                target_pos = target.value[:, 0]
                x, y, theta = target_pos
                if x == 0:
                    x = 0.001
                angle_offset = np.arctan(y / x)
                if x < old_pose[0]:
                    angle_offset += np.pi

                robot_pos = None
                attempt = 0
                while (
                    robot_pos is None
                    or collisionChecker.CheckCollision(
                        robot_body.env_body, report=col_report
                    )
                    or col_report.minDistance <= 0.01
                ):

                    robot_angle = np.random.uniform(
                        angle_offset - np.pi / 3, angle_offset + np.pi / 3
                    )
                    hand_angle = constants.GRIPPER_OFFSET_ANGLE + robot_angle
                    robot_x = x - np.cos(hand_angle) * constants.GRIPPER_OFFSET_DISP
                    robot_y = y - np.sin(hand_angle) * constants.GRIPPER_OFFSET_DISP
                    robot_pos = np.array([[robot_x], [robot_y], [robot_angle]])
                    robot_body.set_pose(robot_pos.flatten())

                    wrist_angle = hand_angle - target.rotation[2]
                    lift = np.maximum(0, target_pos[2] - 0.16 + 0.125)
                    arm = np.array(
                        [[lift], [-np.pi / 2], [0], [-np.pi / 2], [wrist_angle]]
                    )
                    robot_body.set_dof({"arm": arm.flatten()})

                    attempt += 1
                    if attempt > attempt_limit:
                        robot_pos = None
                        break

                if robot_pos is None:
                    continue

                robot_pose.append(
                    {
                        "value": old_pose,
                        "gripper": np.array([[constants.GRIPPER_OPEN]])
                        if act.name == "can_putdown"
                        else np.array([[constants.GRIPPER_CLOSE]]),
                        "arm": arm,
                    }
                )
            else:
                raise NotImplementedError

        if not len(robot_pose):
            print("COULD NOT FIND POSE TO PLAN TO.")
        return robot_pose
