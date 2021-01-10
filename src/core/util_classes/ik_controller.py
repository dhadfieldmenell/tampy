"""
Adapted from: https://github.com/StanfordVL/robosuite/blob/master/robosuite/controllers/baxter_ik_controller.py

@inproceedings{corl2018surreal,
  title={SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark},
  author={Fan, Linxi and Zhu, Yuke and Zhu, Jiren and Liu, Zihua and Zeng, Orien and Gupta, Anchit and Creus-Costa, Joan and Savarese, Silvio and Fei-Fei, Li},
  booktitle={Conference on Robot Learning},
  year={2018}
}
"""

import os
import numpy as np
import pybullet as p

from core.util_classes import transform_utils as T


class IKController(object):
    def __init__(self, body_id, geom, cur_pos, use_rot_mat=False):
        # Set up inverse kinematics
        self.ik_robot = body_id
        self.geom = geom
        self.use_rot_mat = use_rot_mat
        self.setup_inverse_kinematics()
        self.commanded_joint_positions = cur_pos
        self.sync_state()

    def get_control(self, arms, cur_pos):
        # Sync joint positions for IK.
        self.sync_ik_robot(cur_pos)

        # Compute target joint positions
        self.commanded_joint_positions = self.joint_positions_for_eef_command(arms)

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(geom.arm_dim)
        deltas = self._get_current_error(
            self.robot_jpos_getter(), self.commanded_joint_positions
        )

        for i, delta in enumerate(deltas):
            velocities[i] = -2 * delta
        velocities = self.clip_joint_velocities(velocities)

        self.commanded_joint_velocities = velocities
        return velocities

        # For debugging purposes: set joint positions directly
        # robot.set_joint_positions(self.commanded_joint_positions)

    def sync_state(self, cur_pos):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot(cur_pos)

        # make sure target pose is up to date
        self.ik_robot_target_pos = {}
        self.ik_robot_target_orn = {}
        for arm in self.geom.arms:
            pos, orn = self.ik_robot_eef_joint_cartesian_pose(arm)
            self.ik_robot_target_pos[arm] = pos
            self.ik_robot_target_orn[arm] = orn

    def sync_ik_robot(self, joint_positions, simulate=False, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (list): a list or flat numpy array of joint positions.
            simulate (bool): If True, actually use physics simulation, else 
                write to physics state directly.
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """
        num_joints = len(joint_positions)
        if not sync_last:
            num_joints -= 1

        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    self.ik_robot,
                    self.geom.actual[i],
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                p.resetJointState(self.ik_robot, self.geom.actual[i], joint_positions[i])

    def sync_ik_from_attrs(self, attr_vals):
        for key in attr_vals:
            for i, jnt_ind in enumerate(self.geom.dof_map[key]):
                p.resetJointState(self.ik_robot, self.geom.dof_map[key][i], attr_vals[key][i])
           
    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect
        to the base frame as a (pos, orn) tuple where orn is a x-y-z-w quaternion.
        """
        out = []
        for arm in self.geom.arms:
            eff = self.geom.get_ee_link(arm)
            eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, eff)[0])
            eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, eff)[1])
            eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

            base_pos_in_world = np.array(
                p.getBasePositionAndOrientation(self.ik_robot)[0]
            )
            base_orn_in_world = np.array(
                p.getBasePositionAndOrientation(self.ik_robot)[1]
            )
            base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
            world_pose_in_base = T.pose_inv(base_pose_in_world)

            eef_pose_in_base = T.pose_in_A_to_pose_in_B(
                pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
            )
            out.extend(T.mat2pose(eef_pose_in_base))

        return out

    def get_manip_trans(self, arm):
        eff = self.geom.get_ee_link(arm)
        eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, eff)[0])
        eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, eff)[1])
        eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

        pos, quat = p.getBasePositionAndOrientation(self.ik_robot)
        base_pos_in_world = np.array(pos)
        base_orn_in_world = np.array(quat)
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        eef_pose_in_base = T.pose_in_A_to_pose_in_B(
            pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
        )
        return eef_pose_in_base

    def get_jnt_angles(self, arm):
        jnts = self.geom.get_arm_inds(arm)
        jnt_info = p.getJointStates(jnts)
        pos = jnt_info[0]
        return pos

    def inverse_kinematics(
        self,
        target_position,
        target_orientation,
        arm,
        rest_poses,
    ):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position_{right, left}: A tuple, list, or numpy array of size 3 for position.
            target_orientation_{right, left}: A tuple, list, or numpy array of size 4 for
                a orientation quaternion.
            rest_poses: A list of size @num_joints to favor ik solutions close by.

        Returns:
            A list of size @num_joints corresponding to the joint angle solution.
        """
        lower, upper = self.geom.get_jnt_bnds(arm)
        ranges = (np.array(upper)-np.array(lower)).tolist()
        ik_solution = list(
            p.calculateInverseKinematics(
                self.ik_robot,
                self.geom.get_ee_link(arm),
                target_position,
                targetOrientation=target_orientation,
                restPoses=rest_poses,
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=ranges,
            )
        )
        free_inds = self.geom.get_free_inds(arm)
        return np.array(ik_solution)[list(free_inds)].tolist()

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        Args:
            pose_in_base: a (pos, orn) tuple.

        Returns:
            pose_in world: a (pos, orn) tuple.
        """
        pose_in_base = T.pose2mat(pose_in_base)

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = T.pose_in_A_to_pose_in_B(
            pose_A=pose_in_base, pose_A_in_B=base_pose_in_world
        )
        return T.mat2pose(pose_in_world)

    def joint_positions_for_eef_command(self, cmd, arm, rest_poses):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.
        Same arguments as @get_control.
        Returns:
            A list of size @num_joints corresponding to the target joint angles.
        """

        dpos = cmd["dpos"]
        rotation = cmd["rotation"]
        self.target_pos[arm] = self.ik_robot_target_pos[arm]
        self.ik_robot_target_pos[arm] = dpos
        self.ik_robot_target_orn[arm] = rotation
        world_targets = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos[arm], self.ik_robot_target_orn[arm])
        )
        arm_joint_pos = self.inverse_kinematics(
            world_targets[0],
            world_targets[1],
            use_right,
            rest_poses=rest_poses.tolist(),
            maxNumIterations=500,
        )
        both_arm_joint_pos = rest_poses.copy()
        bnds = self.geom.get_arm_bnds(arm)
        both_arm_joints_pos[bnds[0]:bnds[1]] = arm_joint_pos
        self.sync_ik_robot(both_arm_joint_pos, sync_last=True)

        return arm_joint_pos

    def _get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current: the current joint positions.
            set_point: the joint positions that are desired as a numpy array.

        Returns:
            the current error in the joint positions.
        """
        error = current - set_point
        return error

    def clip_joint_velocities(self, velocities):
        """
        Clips joint velocities into a valid range.
        """
        for i in range(len(velocities)):
            if velocities[i] >= 1.0:
                velocities[i] = 1.0
            elif velocities[i] <= -1.0:
                velocities[i] = -1.0
        return velocities


