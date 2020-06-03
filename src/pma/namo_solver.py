import numpy as np

from pma import backtrack_ll_solver

class NAMOSolver(backtrack_ll_solver.BacktrackLLSolver):
    def get_resample_param(self, a):
        return a.params[0] # Experiment with avoiding robot pose symbols
        
        if a.name == 'moveto':
            ## find possible values for the final pose
            rs_param = None # a.params[2]
        elif a.name == 'movetoholding':
            ## find possible values for the final pose
            rs_param = None # a.params[2]
        elif a.name.find('grasp') >= 0:
            ## sample the grasp/grasp_pose
            rs_param = a.params[4]
        elif a.name == 'putdown':
            ## sample the end pose
            rs_param = None # a.params[4]
        elif a.name.find('place') >= 0:
            rs_param = a.params[2]
        elif a.name.find('moveto') >= 0:
            rs_param = a.params[4]
            # rs_param = None
        elif a.name.find('place_at') >= 0:
            # rs_param = None
            rs_param = a.params[2]
        elif a.name == 'short_grasp':
            rs_param = a.params[4]
            # rs_param = None
        elif a.name == 'short_movetograsp':
            rs_param = a.params[4]
            # rs_param = None
        elif a.name == 'short_place_at':
            # rs_param = None
            rs_param = a.params[2]
        else:
            raise NotImplementedError

        return rs_param

 
    def obj_pose_suggester(self, plan, anum, resample_size=1):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum+1]
        else:
            act, next_act = plan.actions[anum], None

        robot = plan.params['pr2']
        robot_body = robot.openrave_body
        start_ts, end_ts = act.active_timesteps
        old_pose = robot.pose[:, start_ts].reshape((2, 1))
        robot_body.set_pose(old_pose[:, 0])
        for i in range(resample_size):
            if next_act != None and (next_act.name == 'grasp' or next_act.name == 'putdown'):
                target = next_act.params[2]
                target_pos = target.value - [[0], [0.]]
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]]) if next_act.name == 'putdown' else np.array([[1.]])})
            elif act.name == 'new_quick_movetograsp' or act.name == 'quick_moveto':
                target = act.params[2]
                grasp = act.params[5]
                target_pos = target.value + grasp.value
                # robot_pose.append({'value': target_pos, 'gripper': np.array([[1.]])})
                robot_pose.append({'pose': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name == 'short_movetograsp':
                target = act.params[2]
                grasp = act.params[5]
                target_pos = target.value + grasp.value + np.array([[0.], [-0.3]])
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name == 'short_grasp':
                target = act.params[2]
                grasp = act.params[5]
                target_pos = target.value + grasp.value + np.array([[0.], [-0.1]])
                robot_pose.append({'value': target_pos, 'gripper': np.array([[1.]])})
            elif act.name == 'grasp' or act.name == 'new_grasp':
                target = act.params[2]
                grasp = act.params[5]
                target_pos = target.value + 2*grasp.value
                robot_pose.append({'value': target_pos, 'gripper': np.array([[1.]])})
            elif act.name == 'quick_place_release':
                target = act.params[4]
                grasp = act.params[5]
                target_pos = target.value + 2*grasp.value
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name == 'new_quick_place_at':
                target = act.params[4]
                grasp = act.params[5]
                target_pos = target.value + grasp.value
                # robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
                robot_pose.append({'pose': target_pos, 'gripper': np.array([[1.]])})
                # robot_pose.append({'pose': target_pos + grasp.value / 2., 'gripper': np.array([[-1.]])})
            elif act.name == 'short_place_at':
                target = act.params[4]
                grasp = act.params[5]
                target_pos = target.value + grasp.value
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name == 'place' or act.name == 'new_place':
                target = act.params[4]
                grasp = act.params[5]
                target_pos = target.value + 2 * grasp.value
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name.find('movetograsp') >= 0:
                target = act.params[2]
                grasp = act.params[5]
                if i > 0:
                    target_pos = target.value + grasp.value + [[np.random.normal(0, 0.05)], [-np.random.uniform(0.1, 0.3)]]
                else:
                    target_pos = target.value + grasp.value + np.array([[0.], [-0.2]])# + [[np.random.normal(0, 0.05)], [-np.random.uniform(0.1, 0.3)]]
                    # target_pos = target.value + grasp.value + np.array([[0.], [-0.05]])# + [[np.random.normal(0, 0.05)], [-np.random.uniform(0.1, 0.3)]]
                    # disp = np.linalg.norm(old_pose - target_pos)
                    # if disp > 3:
                    #     target_pos = old_pose + np.random.uniform(2, disp)*(target_pos - old_pose) / disp
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name.find('place') >= 0:
                target = act.params[4]
                grasp = act.params[5]
                # target_pos = target.value + grasp.value
                target_pos = target.value + grasp.value # + np.array([[0.], [-0.05]])
                # robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]])})
            elif act.name == 'grasp' or act.name == 'putdown':
                target = act.params[2]
                radius1 = act.params[0].geom.radius
                radius2 = act.params[1].geom.radius
                dist = radisu1 + radius2
                target_pos = target.value - [[0], [dist]]
                robot_pose.append({'value': target_pos, 'gripper': np.array([[-1.]]) if act.name == 'putdown' else np.array([[1.]])})
            else:
                raise NotImplementedError

        return robot_pose

