import numpy as np
import pybullet as p

import core.util_classes.common_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.robot_sampling as robot_sampling
import core.util_classes.transform_utils as T
from pma import backtrack_ll_solver


class RobotSolver(backtrack_ll_solver.BacktrackLLSolver):
    def get_resample_param(self, a):
        return a.params[0]


    def vertical_gripper(self, robot, arm, obj, gripper_open=True, ts=(0,0), rand=False):
        start_ts, end_ts = ts
        robot_body = robot.openrave_body
        robot_body.set_from_param(robot, start_ts)

        old_arm_pose = getattr(robot, arm)[:, start_ts].copy()
        if obj.is_symbol():
            target_loc = obj.value[:, 0] + np.array([0, 0, const.GRASP_DIST])
        else:
            target_loc = obj.pose[:, start_ts] + np.array([0, 0, const.GRASP_DIST])

        gripper_axis = robot.geom.get_gripper_axis(arm)
        target_axis = [0, 0, -1]
        quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, target_axis)

        if 'box' in obj.get_type(True):
            euler = obj.rotation[:,ts[0]]
            obj_quat = T.euler_to_quaternion(euler, 'xyzw')
            robot_mat = T.quat2mat(quat)
            obj_mat = T.quat2mat(obj_quat)
            quat = T.mat2quat(obj_mat.dot(robot_mat))

        iks = []
        attempt = 0
        while not len(iks) and attempt < 20:
            if rand:
                target_loc += np.clip(np.random.normal(0, 0.015, 3), -0.03, 0.03)

            iks = robot_body.get_ik_from_pose(target_loc, quat, arm)
            rand = not len(iks)
            attempt += 1
        if not len(iks): return None
        arm_pose = np.array(iks).reshape((-1,1))
        pose = {arm: arm_pose}
        gripper = robot.geom.get_gripper(arm)
        if gripper is not None:
            pose[gripper] = robot.geom.get_gripper_open_val() if gripper_open else robot.geom.get_gripper_closed_val()
            pose[gripper] = np.array(pose[gripper]).reshape((-1,1))
        for aux_arm in robot.geom.arms:
            if aux_arm == arm: continue
            old_pose = getattr(robot, aux_arm)[:, start_ts].reshape((-1,1))
            pose[aux_arm] = old_pose
            aux_gripper = robot.geom.get_gripper(aux_arm)
            if aux_gripper is not None:
                pose[aux_gripper] = getattr(robot, aux_gripper)[:, start_ts].reshape((-1,1))
        for arm in robot.geom.arms:
            robot_body.set_dof({'left': pose[arm].flatten().tolist()})
            info = robot_body.fwd_kinematics(arm)
            pose['{}_ee_pos'.format(arm)] = np.array(info['pos']).reshape((-1,1))
        return pose


    def obj_pose_suggester(self, plan, anum, resample_size=20):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum+1]
        else:
            act, next_act = plan.actions[anum], None

        robot = act.params[0]
        robot_body = robot.openrave_body
        start_ts = act.active_timesteps[0]
        for arm in robot.geom.arms:
            attr = '{}_ee_pos'.format(arm)
            if hasattr(robot, attr):
                info = robot_body.fwd_kinematics(arm)
                getattr(robot, attr)[:, start_ts] = info['pos']

        for arm in robot.geom.arms:
            robot.openrave_body.set_dof({arm: getattr(robot, arm)[:,0]})
        obj = act.params[1]
        for param in plan.params.values():
            if hasattr(param, 'openrave_body') and param.openrave_body is not None:
                param.openrave_body.set_pose(param.pose[:,0])

        a_name = act.name.lower()
        arm = robot.geom.arms[0]
        st, et = act.active_timesteps
        if a_name.find('left') >= 0: arm = 'left'
        if a_name.find('right') >= 0: arm = 'right'

        gripper_open = False
        if a_name.find('move_to_grasp') >= 0 or (a_name.find('putdown') >= 0 and a_name.find('move_to') < 0):
            gripper_open = True

        rand = False # a_name.find('move') >= 0

        if next_act is not None:
            next_obj = next_act.params[1]
            next_a_name = next_act.name.lower()
            next_arm = robot.geom.arms[0]
            next_gripper_open = not gripper_open
            next_st, next_et = next_act.active_timesteps
            if next_a_name.find('left') >= 0: arm = 'left'
            if next_a_name.find('right') >= 0: arm = 'right'

        ### Sample poses
        for i in range(resample_size):
            ### Cases for when behavior can be inferred from current action
            if a_name.find('grasp') >= 0 or a_name.find('putdown') >= 0:
                pose = self.vertical_gripper(robot, arm, obj, gripper_open, (st, et), rand=(rand or (i>0)))

            ### Cases for when behavior cannot be inferred from current action
            elif next_act is None:
                pose = None

            elif next_a_name.find('grasp') >= 0 or next_a_name.find('putdown') >= 0:
                pose = self.vertical_gripper(robot, next_arm, next_obj, next_gripper_open, (next_st, next_et), rand=(rand or (i>0)))

            if pose is None: break
            robot_pose.append(pose)

        return robot_pose


    def _cleanup_plan(self, plan, active_ts):
        for param in plan.params.values():
            if 'Robot' not in param.get_type(True): continue
            for arm in param.geom.arms:
                attr = '{}_ee_pos'.format(arm)
                if not hasattr(param, attr): continue
                for t in range(active_ts[0], active_ts[1]):
                    if np.any(np.isnan(getattr(param, arm)[:,t])): continue
                    param.openrave_body.set_dof({arm: getattr(param, arm)[:,t]})
                    info = param.openrave_body.fwd_kinematics(arm)
                    getattr(param, attr)[:,t] = info['pos']

