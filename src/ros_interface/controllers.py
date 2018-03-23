import core.util_classes.baxter_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanDeserializer, PlanSerializer
from pma.robot_ll_solver import RobotLLSolver
# from ros_interface.baxter_controller import 
# from ros_interface.environment_monitor import EnvironmentMonitor

import rospy

import std_msgs.msg

import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import CollisionAvoidanceState

import numpy as np


ROS_RATE = 500

left_joints = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
right_joints = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

joint_velocity_limits = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
# error_limits = np.array([.01, .075, .05, .075, .075, .01, .01])
error_limits = np.array([.01, .025, .025, .01, .01, .01, .01])
stop_error_limits = np.array([.05, .1, .1, .5, .5, .1, .1])

def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    cur_arm_pose = np.array(cur_arm_pose).flatten()
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

class TrajectoryController(object):
    def __init__(self):
        self.left = baxter_interface.limb.Limb('left')
        self.right = baxter_interface.limb.Limb('right')
        self.left_grip = baxter_interface.gripper.Gripper('left')
        self.right_grip = baxter_interface.gripper.Gripper('right')
        self.left_col_pub = rospy.Publisher("/robot/limb/left/suppress_collision_avoidance", std_msgs.msg.Empty, queue_size=1)
        self.right_col_pub = rospy.Publisher("/robot/limb/right/suppress_collision_avoidance", std_msgs.msg.Empty, queue_size=1)
        self.left_col_sub = rospy.Subscriber("/robot/limb/left/collision_avoidance_state", CollisionAvoidanceState, self.left_col_callback)
        self.right_col_sub = rospy.Subscriber("/robot/limb/right/collision_avoidance_state", CollisionAvoidanceState, self.right_col_callback)

    def left_col_callback(self, msg):
        self.left_col_pub.publish(std_msgs.msg.Empty())

    def right_col_callback(self, msg):
        self.right_col_pub.publish(std_msgs.msg.Empty())

    def execute_timestep(self, baxter_parameter, ts, real_t=1, limbs=['left', 'right'], check_collision=True):
        use_left = 'left' in limbs
        use_right = 'right' in limbs
        left_target = baxter_parameter.lArmPose[:, ts]
        right_target = baxter_parameter.rArmPose[:, ts]
        left_gripper_open = baxter_parameter.lGripper[:, ts] > const.GRIPPER_CLOSE_VALUE
        right_gripper_open = baxter_parameter.rGripper[:, ts] > const.GRIPPER_CLOSE_VALUE
        if np.any(np.isnan(left_target)) or np.any(np.isnan(right_target)):
            print "Experienced NaN in controller."
            return False

        current_left = map(lambda j: self.left.joint_angles()[j], left_joints)
        current_right = map(lambda j: self.right.joint_angles()[j], right_joints)

        cur_left_err = left_target - current_left
        cur_right_err = right_target - current_right

        real_t = float(real_t)
        # left_vel_ratio = min(np.mean((np.abs(cur_left_err) / real_t) / joint_velocity_limits), 1.0)
        # self.left.set_joint_position_speed(left_vel_ratio)
        # right_vel_ratio = min(np.mean((np.abs(cur_right_err) / real_t) / joint_velocity_limits), 1.0)
        # self.right.set_joint_position_speed(right_vel_ratio)

        self.left.set_joint_position_speed(0.05)
        self.right.set_joint_position_speed(0.05)
        self.left_grip.set_holding_force(95)
        self.right_grip.set_holding_force(95)
        self.left_grip.set_moving_force(95)
        self.right_grip.set_moving_force(95)

        attempt = 0.0
        r = rospy.Rate(ROS_RATE)
        while attempt <= int(ROS_RATE * real_t) and ((np.any(np.abs(left_target - current_left) > error_limits) and use_left) or (np.any(np.abs(right_target - current_right) > error_limits) and use_right)):
            next_left_target = left_target # current_left + (left_target - current_left) / 3.5 # (attempt / int(ROS_RATE * real_t)) * cur_left_err
            next_right_target = right_target # current_right + (right_target - current_right) / 3.5 # (attempt / int(ROS_RATE * real_t)) * cur_right_err
            left_target_dict = dict(zip(left_joints, next_left_target))
            right_target_dict = dict(zip(right_joints, next_right_target))

            if not check_collision:
                self.left_col_pub.publish(std_msgs.msg.Empty())
                self.right_col_pub.publish(std_msgs.msg.Empty())
            
            if 'left' in limbs:
                self.left.set_joint_positions(left_target_dict)
            if 'right' in limbs:
                self.right.set_joint_positions(right_target_dict)

            current_left = map(lambda j: self.left.joint_angles()[j], left_joints)
            current_right = map(lambda j: self.right.joint_angles()[j], right_joints)
            cur_left_err = left_target - current_left
            cur_right_err = right_target - current_right
            attempt += 1.0
            rospy.sleep(0.005)
            # r.sleep()

        self.left_grip.open()  if left_gripper_open else self.left_grip.close()
        self.right_grip.open() if right_gripper_open else self.right_grip.close()

        return (np.all(np.abs(left_target - current_left) < stop_error_limits) or not use_left) and (np.all(np.abs(right_target - current_right) < stop_error_limits) or not use_right)

    def execute_plan(self, plan, mode='position', active_ts=None, controller=None, limbs=['left', 'right'], stop_on_fail=False, check_collision=True):
        # rospy.Rate(ROS_RATE)
        if mode == 'position':
            return self._execute_position_control(plan, active_ts, limbs, stop_on_fail, check_collision)
        else:
            return self._execute_torque_control(plan, active_ts, controller)

    def _execute_position_control(self, plan, active_ts=None, limbs=['left', 'right'], stop_on_fail=False, check_collision=True):
        if active_ts is None:
            active_ts = (0, plan.horizon-1)

        act_index = 0
        # cur_action = plan.actions[0]
        # if cur_action.active_timesteps[0] > active_ts[0]:
        #     raise Exception("Invalid timestep (< min) passed to plan execution.")
        
        # while active_ts[0] > cur_action.active_timesteps[1]:
        #     act_index += 1
        #     if act_index >= len(plan.actions):
        #         raise Exception("Invalid timestep (> max) passed to plan execution")
        #     cur_action = plan.actions[act_index]

        cur_ts = active_ts[0]
        baxter = plan.params['baxter']
        act_index = 0
        while cur_ts <= active_ts[1] and cur_ts < plan.horizon:
            cur_action = plan.actions[act_index]
            success = self.execute_timestep(baxter, cur_ts, 2, limbs=limbs, check_collision=check_collision)
            if not success:
                print 'Failed timestep {}'.format(cur_ts)
                if stop_on_fail:
                    return False
            #     self._adjust_for_failed_execute(plan, cur_ts)
            if cur_action.name == 'basket_grasp' and cur_ts > cur_action.active_timesteps[0] + 12:
                if self.left_grip.position() < 20 or self.right_grip.position < 20:
                    self.left_grip.open()
                    self.right_Grip.open()
                    return False
            cur_ts += 1
            if cur_ts > cur_action.active_timesteps[1]:
                act_index += 1
            # if cur_ts >= cur_action.active_timesteps[1] and cur_ts < active_ts[1]:
            #     act_index += 1
            #     if act_index >= len(plan.actions):
            #         raise Exception("Invalid timestep (> max) passed to plan execution")
            #     cur_action = plan.actions[act_index]
        # print 'Execution finished'
        return True

    def _execute_torque_control(self, plan, active_ts, controller):
        if active_ts is None:
            active_ts = (0, plan.horizon-1)

        act_index = 0
        cur_action = plan.actions[0]
        if active_ts < active_ts[0]:
            raise Exception("Invalid timestep (< min) passed to plan execution.")
        
        while active_ts[0] > cur_action.active_timesteps[1]:
            act_index += 1
            if act_index >= len(plan.actions):
                raise Exception("Invalid timestep (> max) passed to plan execution")
            cur_action = plan.actions[act_index]

        cur_ts = active_ts[0]
        baxter = plan.params['baxter']
        self._update_plan(plan, cur_action.name)
        while cur_ts <= active_ts[1] and cur_ts < plan.horizon:
            success = self.execute_timestep(baxter, cur_ts, plan.time[:, cur_ts])
            if not success:
                self._adjust_for_failed_execute(plan, cur_ts)
            cur_ts += 1
            if cur_ts >= cur_action.active_timesteps[1] and cur_ts < active_ts[1]:
                act_index += 1
                if act_index >= len(plan.actions):
                    raise Exception("Invalid timestep (> max) passed to plan execution")
                cur_action = plan.actions[act_index]
                self._update_plan(plan, cur_action.name)

        print 'Execution finished'

    def _execute_torques_for_ts(self, plan, ts, controller, real_t):
        cur_t = 0
        while cur_t < real_t:
            torques = controller.step_control_loop(plan, ts+1, cur_t)
            left_target_dict = dict(zip(left_joints, torques[8:14]))
            right_target_dict = dict(zip(right_joints, torques[:7]))

            self.left.set_joint_torques(left_target_dict)
            self.right.set_joint_torques(right_target_dict)

            cur_t += const.time_delta

    def execute_action(self, plan, act_n):
        if act_n >= len(plan.actions):
            raise Exception("Invalid action number (> max) passed to action execution.")
        cur_action = plan.actions[act_n]
        active_ts = cur_action.active_timesteps

        cur_ts = active_ts[0]
        baxter = plan.params['baxter']
        while cur_ts <= active_ts[1] and cur_ts < plan.horizon:
            success = self.execute_timestep(baxter, cur_ts, 1, ['left', 'right'])
            cur_ts += 1

        print 'Execution finished'

    def _update_plan(self, plan, type):
        pass

    def _adjust_for_failed_execute(self, plan, ts):
        print 'Failed timestep {0}'.format(ts)
        pass


class EEController(object):
    def __init__(self):
        self.env = ParamSetup.setup_env()
        self.left = baxter_interface.limb.Limb('left')
        self.right = baxter_interface.limb.Limb('right')
        self.left_grip = baxter_interface.gripper.Gripper('left')
        self.right_grip = baxter_interface.gripper.Gripper('right')
        self.left_target = []
        self.right_target = []
        self.baxter = ParamSetup.setup_baxter()
        env = ParamSetup.setup_env()
        self.baxter.openrave_body = OpenRAVEBody(env, 'baxter', self.baxter.geom)
        self.left.set_joint_position_speed(0.1)
        self.right.set_joint_position_speed(0.1)

    def update_targets(self, ee_left_pos, ee_left_rot, ee_right_pos, ee_right_rot):
        if len(ee_left_pos) and len(ee_left_rot):
            left_targets = self.baxter.openrave_body.get_ik_from_pose(ee_left_pos, ee_left_rot, "left_arm")
            if len(left_targets):
                self.left_target = closest_arm_pose(left_targets, self.left.joint_angles().values())

        if len(ee_right_pos) and len(ee_right_rot):
            right_targets = self.baxter.openrave_body.get_ik_from_pose(ee_right_pos, ee_right_rot, "right_arm")
            if len(right_targets):
                self.right_target = closest_arm_pose(right_targets, self.right.joint_angles().values())

    def move_toward_targets(self, limbs=['left', 'right']):
        use_left = 'left' in limbs
        use_right = 'right' in limbs
        if (not len(self.left_target) and use_left) or (not len(self.right_target) and use_right):
            print 'Targets uninitialized, call update_targets\n'
            return

        left_target_dict = dict(zip(left_joints, self.left_target))
        right_target_dict = dict(zip(right_joints, self.right_target))
        if 'left' in limbs:
            self.left.set_joint_positions(left_target_dict)
        if 'right' in limbs:
            self.right.set_joint_positions(right_target_dict)

    def move_to_targets(self, limbs=['left', 'right'], max_iters=1000):
        use_left = 'left' in limbs
        use_right = 'right' in limbs
        if (not len(self.left_target) and use_left) or (not len(self.right_target) and use_right):
            print 'Targets uninitialized, call update_targets\n'
            return

        left_target_dict = dict(zip(left_joints, self.left_target))
        right_target_dict = dict(zip(right_joints, self.right_target))

        iters = 0
        all_close = False
        try:
            while not all_close:
                if (iters > max_iters):
                    print 'Maxed out iterations'
                    break
                iters += 1
                self.move_toward_targets(limbs)
                left_angles = self.left.joint_angles()
                right_angles = self.right.joint_angles()
                all_close = True
                if use_left:
                    for key in left_joints:
                        if np.abs(left_angles[key] - left_target_dict[key]) > 0.05:
                            all_close = False
                            break

                if use_right:
                    for key in right_joints:
                        if np.abs(right_angles[key] - right_target_dict[key]) > 0.05:
                            all_close = False
                            break

                rospy.sleep(0.01)

        except KeyboardInterrupt:
            import ipdb; ipdb.set_trace()


    def close_grippers(self, grippers=['left', 'right']):
        if 'left' in grippers: self.left_grip.close()
        if 'right' in grippers: self.right_grip.close()

    def open_grippers(self, grippers=['left', 'right']):
        if 'left' in grippers: self.left_grip.open()
        if 'right' in grippers: self.right_grip.open()
