import numpy as np

import core.util_classes.hsr_constants as constants
from pma import backtrack_ll_solver

class HSRSolver(backtrack_ll_solver.BacktrackLLSolver):
    def get_resample_param(self, a):
        if a.name == 'moveto':
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == 'move_holding_can':
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == 'can_grasp':
            ## sample the grasp/grasp_pose
            rs_param = a.params[-1]
        elif a.name == 'can_putdown':
            ## sample the end pose
            rs_param = a.params[-1]
        elif a.name == 'stack_cans':
            rs_param = a.params[-1]
        else:
            raise NotImplemented

        return rs_param

 
    def obj_pose_suggester(self, plan, anum, resample_size=10):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum+1]
        else:
            act, next_act = plan.actions[anum], None

        robot = plan.params['hsr']
        robot_body = robot.openrave_body
        start_ts, end_ts = act.active_timesteps
        old_pose = robot.pose[:, start_ts].reshape((3, 1))
        robot_body.set_pose(old_pose[:, 0])
        for i in range(resample_size):
            if next_act != None and (next_act.name == 'can_grasp' or next_act.name == 'can_putdown'):
                target = next_act.params[2]
                target_pos = target.value[:,0]
                x, y, theta = target_pos
                angle_offset = np.arctan(y / x)
                if x < old_pose[0]:
                    angle_offset += np.pi
                robot_angle = np.random.uniform(angle_offset-np.pi/4, angle_offset+np.pi/4)
                hand_angle = constants.GRIPPER_OFFSET_ANGLE + robot_angle
                robot_x = x - np.cos(hand_angle) * constants.GRIPPER_OFFSET_DISP
                robot_y = y - np.sin(hand_angle) * constants.GRIPPER_OFFSET_DISP
                robot_pos = [[x], [y], [robot_angle]]
                robot_pose.append({'value': robot_pos, 'gripper': np.array([[constants.GRIPPER_CLOSE]]) if next_act.name == 'putdown' else np.array([[constants.GRIPPER_OPEN]]), 'arm': [[target_pos[2]-0.16+0.2], [-np.pi/2], [0], [-np.pi/2], [target.rotation[0]]]})
            elif act.name == 'movetograsp':
                target = act.params[2]
                grasp = act.params[5]
                target_pos = target.value + grasp.value + [[np.random.normal(0, 0.05)], [-np.random.uniform(0.1, 0.3)]]
                robot_pose.append({'value': target_pos, 'gripper': np.array([[1.]])})
            elif act.name == 'place_at':
                target = act.params[4]
                grasp = act.params[5]
                target_pos = target.value + grasp.value
                robot_pose.append({'value': target_pos, 'gripper': np.array([[1.]])})
            elif act.name == 'grasp' or act.name == 'putdown':
                target = act.params[2]
                radius1 = act.params[0].geom.radius
                radius2 = act.params[1].geom.radius
                dist = radisu1 + radius2
                target_pos = target.value - [[0], [dist]]
                robot_pose.append({'value': target_pos, 'gripper': np.array([[0.]]) if act.name == 'putdown' else np.array([[1.]])})
            else:
                raise NotImplementedError

        return robot_pose

