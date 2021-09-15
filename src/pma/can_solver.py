import numpy as np
from pma import backtrack_ll_solver_gurobi


attr_map = {
    "Robot": ["backHeight", "lArmPose", "lGripper", "rArmPose", "rGripper", "pose"],
    "RobotPose": [
        "backHeight",
        "lArmPose",
        "lGripper",
        "rArmPose",
        "rGripper",
        "value",
    ],
    "EEPose": ["value", "rotation"],
    "Can": ["pose", "rotation"],
    "Target": ["value", "rotation"],
    "Obstacle": ["pose", "rotation"],
}

DOWN_ROT = np.array([0, np.pi / 2, 0])
GRIPPER_OPEN_VAL = 0.02
GRIPPER_CLOSE_VAL = 0.015


class CanSolver(backtrack_ll_solver_gurobi.BacktrackLLSolver):
    def get_rs_param(self, a):
        if a.name == "moveto":
            ## moveto: (?robot - Robot ?start - RobotPose ?end - RobotPose)
            ## sample ?end - RobotPose
            rs_param = a.params[2]
        elif a.name == "movetoholding":
            ## movetoholding: (?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can)
            ## sample ?end - RobotPose
            rs_param = a.params[2]
        elif a.name == "grasp":
            ## grasp: (?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)
            ## sample ?ee - EEPose
            rs_param = a.params[4]
            lookup_preds = "InContact"
        elif a.name == "putdown":
            ## putdown: (?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)
            ## sample ?ep - RobotPose
            rs_param = a.params[5]
        elif a.name == "move_both_holding_cloth":
            rs_param = a.params[4]
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

        robot = plan.params["pr2"]
        robot_body = robot.openrave_body
        start_ts, end_ts = act.active_timesteps
        old_l_arm_pose = robot.lArmPose[:, start_ts].reshape((7, 1))
        old_r_arm_pose = robot.rArmPose[:, start_ts].reshape((7, 1))
        old_pose = robot.pose[:, start_ts].reshape((3, 1))
        old_backHeight = robot.backHeight[:, start_ts].reshape((1, 1))
        robot_body.set_dof(
            {
                "lArmPose": old_l_arm_pose.flatten(),
                "rArmPose": old_r_arm_pose.flatten(),
                "lGripper": [0.02],
                "rGripper": [0.02],
            }
        )
        robot_body.set_pose(old_pose[:, 0])
        for i in range(resample_size):
            if next_act and (
                next_act.name == "right_grasp" or next_act.name == "right_putdown"
            ):
                target = next_act.params[2]
                target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
                next_act.params[1].openrave_body.set_pose(
                    target.pose[:, 0], target.rotation[:, 0]
                )
                robot.openrave_body.set_dof({"lArmPose": old_l_arm_pose})
                rposes = robot.openrave_body.get_ik_from_pose(
                    target_pos, DOWN_ROT, "rightarm_torso"
                )
                if not len(rposes):
                    pass

                closest_ind = np.argmin(np.sum(np.abs(rposes - old_r_arm_pose), axis=1))
                rpose = rposes[closest_ind]

                robot_pose.append(
                    {
                        "lArmPose": old_l_arm_pose,
                        "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "rArmPose": rpose,
                        "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "pose": old_pose,
                        "backHeight": old_backHeight,
                    }
                )
            elif next_act and (
                next_act.name == "left_grasp" or next_act.name == "left_putdown"
            ):
                target = next_act.params[2]
                target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
                next_act.params[1].openrave_body.set_pose(
                    target.pose[:, 0], target.rotation[:, 0]
                )
                robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
                lposes = robot.openrave_body.get_ik_from_pose(
                    target_pos, DOWN_ROT, "leftarm_torso"
                )
                if not len(lposes):
                    pass

                closest_ind = np.argmin(np.sum(np.abs(lposes - old_l_arm_pose), axis=1))
                lpose = lposes[closest_ind]

                robot_pose.append(
                    {
                        "lArmPose": lpose,
                        "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "rArmPose": old_r_arm_pose,
                        "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "pose": old_pose,
                        "backHeight": old_backHeight,
                    }
                )
            elif act.name == "right_grasp" or act.name == "right_putdown":
                target = act.params[2]
                target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
                act.params[1].openrave_body.set_pose(
                    target.pose[:, 0], target.rotation[:, 0]
                )
                robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
                rposes = robot.openrave_body.get_ik_from_pose(
                    target_pos, DOWN_ROT, "rightarm_torso"
                )
                if not len(rposes):
                    pass

                closest_ind = np.argmin(np.sum(np.abs(rposes - old_r_arm_pose), axis=1))
                rpose = rposes[closest_ind]

                robot_pose.append(
                    {
                        "lArmPose": old_l_arm_pose,
                        "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "rArmPose": rpose,
                        "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "pose": old_pose,
                        "backHeight": old_backHeight,
                    }
                )
            elif act.name == "left_grasp" or act.name == "left_putdown":
                target = act.params[2]
                target_pos = target.pose[:, 0] + np.array([0, 0, 0.1])
                next_act.params[1].openrave_body.set_pose(
                    target.pose[:, 0], target.rotation[:, 0]
                )
                robot.openrave_body.set_dof({"rArmPose": old_r_arm_pose})
                lposes = robot.openrave_body.get_ik_from_pose(
                    target_pos, DOWN_ROT, "leftarm_torso"
                )
                if not len(lposes):
                    pass

                closest_ind = np.argmin(np.sum(np.abs(lposes - old_l_arm_pose), axis=1))
                lpose = lposes[closest_ind]

                robot_pose.append(
                    {
                        "lArmPose": lpose,
                        "lGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "rArmPose": old_r_arm_pose,
                        "rGripper": np.array([[GRIPPER_OPEN_VAL]]),
                        "pose": old_pose,
                        "backHeight": old_backHeight,
                    }
                )
            else:
                raise NotImplemented

        return robot_pose
