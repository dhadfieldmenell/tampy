""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

from mujoco_py import mjcore, mjconstants, mjviewer
from mujoco_py.mjlib import mjlib

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
# from policy_hooks.action_utils import action_durations
from policy_hooks.baxter_controller import BaxterMujocoController
from policy_hooks.policy_solver_utils import STATE_ENUM, OBS_ENUM, ACTION_ENUM, NOISE_ENUM
import policy_hooks.policy_solver_utils as utils

'''
Mujoco specific
'''
BASE_POS_XML = '../models/baxter/mujoco/baxter_mujoco_pos.xml'
BASE_MOTOR_XML = '../models/baxter/mujoco/baxter_mujoco.xml'
ENV_XML = 'policy_hooks/current_env.xml'

MUJOCO_JOINT_ORDER = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_e2', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint'\
                      'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']

MUJOCO_MODEL_Z_OFFSET = -0.686

N_CONTACT_LIMIT = 12

# TODO: Handle mapping Mujoco state to TAMPy state
# TODO: Split this into two agents for the different simulators (Mujooc & Bullet).
class LaundryWorldMujocoAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        
        self.plan = self._hyperparams['plan']
        self.solver = self._hyperparams['solver']
        self.init_plan_states = self._hyperparams['x0s']
        self.x0 = self._hyperparams['x0']
        self.sim = 'mujoco'
        self.viewer = None
        self.motor_viewer = None
        self.pos_model = self.setup_mujoco_model(self.plan, motor=False, view=True)
        # self.motor_model = self.setup_mujoco_model(self.plan, motor=True, view=False)
        # self.controller = BaxterMujocoController(self.motor_model, pos_gains=2.5e2, vel_gains=1e1)
        self.demonstrations = np.ones((len(self.init_plan_states))) * self._hyperparams['demonstrations']
        self.expert_ratio = self._hyperparams['expert_ratio'] # Probability a demonstration has no added noise

        self.symbols = filter(lambda p: p.is_symbol(), self.plan.params.values())
        self.params = filter(lambda p: not p.is_symbol(), self.plan.params.values())

        self.initial_samples = True
        self.current_cond = 0
        self.cond_optimal_traj_sample = [Sample(self) for _ in  range(len(self.x0))] # Samples from the initial bootstrap
        self.cond_optimal_traj = [None for _ in range(len(self.x0))]
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

    def _generate_xml(self, plan, motor=True):
        '''
            Search a plan for cloths, tables, and baskets to create an XML in MJCF format
        '''
        base_xml = xml.parse(BASE_MOTOR_XML) if motor else xml.parse(BASE_POS_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        active_ts = (0, plan.horizon)
        params = plan.params.values()

        # real_t = np.sum([t for t in plan.time[:, active_ts[0]:active_ts[1]+1]])
        # root.append(xml.fromstring(options_xml.format(MUJOCO_TIME_DELTA, real_t/MUJOCO_TIME_DELTA)))

        for param in params:
            if param.is_symbol(): continue
            if param._type == 'Cloth':
                height = param.geom.height
                radius = param.geom.radius
                x, y, z = param.pose[:, active_ts[0]]
                cloth_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                # cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
                cloth_geom = xml.SubElement(cloth_body, 'geom', {'name': param.name, 'type':'sphere', 'size':"{}".format(radius), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
                cloth_intertial = xml.SubElement(cloth_body, 'inertial', {'pos':'0 0 0', 'quat':'0 0 0 1', 'mass':'0.1', 'diaginertia': '0.01 0.01 0.01'})
            # We might want to change this; in Openrave we model tables as hovering box so there's no easy translation to Mujoco
            elif param._type == 'Obstacle': 
                length = param.geom.dim[0]
                width = param.geom.dim[1]
                thickness = param.geom.dim[2]
                x, y, z = param.pose[:, active_ts[0]]
                table_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 'type':'box', 'size':"{} {} {}".format(length, width, thickness)})
            elif param._type == 'Basket':
                x, y, z = param.pose[:, active_ts[0]]
                yaw, pitch, roll = param.rotation[:, active_ts[0]]
                basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(roll, pitch, yaw)})
                basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
                basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})
        base_xml.write(ENV_XML)

    def setup_mujoco_model(self, plan, motor=False, view=False):
        '''
            Create the Mujoco model and intiialize the viewer if desired
        '''
        self._generate_xml(plan, motor)
        model = mjcore.MjModel(ENV_XML)
        if motor and view:
            self.motor_viewer = mjviewer.MjViewer()
            self.motor_viewer.start()
            self.motor_viewer.set_model(model)
            self.motor_viewer.cam.distance = 3
            self.motor_viewer.cam.azimuth = 180.0
            self.motor_viewer.cam.elevation = -22.5
            self.motor_viewer.loop_once()
        elif view:
            self.viewer = mjviewer.MjViewer()
            self.viewer.start()
            self.viewer.set_model(model)
            self.viewer.cam.distance = 3
            self.viewer.cam.azimuth = 180.0
            self.viewer.cam.elevation = -22.5
            self.viewer.loop_once()
        return model

    # def run_ee_policy(self, cond, policy, noise):
    #     '''
    #         Run the policy (for end effector control) in simulation
    #     '''
    #     plan = self.plans[cond]
    #     baxter = plan.params[p'baxter']
    #     x0 = self.x0[conds]
    #     if plan in self._traj_info_cache:
    #         active_ts, params = self._traj_info_cache[plan]
    #     else:
    #         active_ts, params = utils.get_plan_traj_info(plan)
    #         self._traj_info_cache[plan] = (active_ts, params)
    #     self._set_simulator_state(x0, plan)
    #     trajectory_state = np.zeros((plan.dX, plan.T))
        
    #     x = x0.copy()
    #     for ts in range(active_ts[0], active_ts[1]):
    #         obs = self._get_obs(cond, t)
    #         u = policy.act(x, obs, noise[:, ts - active_ts[0]], ts - active_ts[0])

    #         right_ee_vec = u[:7]
    #         left_ee_vec = u[7:]

    #         right_grip = const.GRIPPER_OPEN_VALUE if right_ee_vec[6] > 0.5 else const.GRIPPER_CLOSE_VALUE
    #         left_grip = const.GRIPPER_OPEN_VALUE if left_ee_vec[6] > 0.5 else const.GRIPPER_CLOSE_VALUE

    #         cur_right_joints = x[plan.state_inds[(param.name, 'rArmPose')]]
    #         cur_left_joints = x[plan.state_inds[(param.name, 'lArmPose')]]

    #         baxter.set_dof('lArmPose':cur_left_joints, 'rArmPose':cur_right_joints, 'lGripper':left_grip, 'rGripper':right_grip)
    #         ee_right = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         next_right_poses = baxter.openrave_body.get_ik_from_pose(right_ee_vec[:3], right_ee_vec[3:6], 'right_arm')
    #         next_left_poses = baxter.openrave_body.get_ik_from_pose(left_ee_vec[:3], left_ee_vec[3:6], 'left_arm')

    #         next_right_pose = utils.closest_arm_pose(next_right_poses, x[plan.state_inds[('baxter', 'rArmPose')]])
    #         next_left_pose = utils.closest_arm_pose(next_left_poses, x[plan.state_inds[('baxter', 'lArmPose')]])

    #         while(np.any(np.abs(cur_right_joints - next_right_pose) > 0.01) or np.any(np.abs(cur_left_joints - next_left_pose) > 0.01)):
    #             self.pos_model.data.ctrl = np.r_[next_right_pose, right_grip, -right_grip, next_left_pose, left_grip, -left_grip]
    #             self.pos_model.step()
    #             cur_right_joints = self.pos_model.data.qpos[1:8]
    #             cur_left_joints = self.pos_model.data.qpos[10:17]

    #         x = self._get_simulator_state()
    #         trajectory_state[:, ts] = x
    #     return trajectory_state

    def _set_simulator_state(self, x, plan, motor=False):
        '''
            Set the simulator to the state of the specified condition, except for the robot
        '''
        model  = self.pos_model if not motor else self.motor_model
        xpos = model.body_pos.copy()
        xquat = model.body_quat.copy()
        param = plan.params.values()

        for param in self.params:
            if param._type != 'Robot': # and (param.name, 'rotation') in plan.state_inds:
                param_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, param.name)
                if param_ind == -1: continue
                if (param.name, 'pose') in plan.state_inds:
                    pos = x[plan.state_inds[(param.name, 'pose')]]
                    xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET]) + np.array([0, 0, 0.025])
                if (param.name, 'rotation') in plan.state_inds:
                    rot = x[plan.state_inds[(param.name, 'rotation')]]
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, rot)[:3,:3])
                if param.name == 'basket':
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, [0, 0, np.pi/2])[:3,:3])

        model.body_pos = xpos
        model.body_quat = xquat
        model.data.qpos = self._baxter_to_mujoco(x, plan.state_inds).reshape((19,1))
        model.forward()

    def _baxter_to_mujoco(self, x, x_inds):
        return np.r_[0, x[x_inds['baxter', 'rArmPose']], x[x_inds['baxter', 'rGripper']], -x[x_inds['baxter', 'rGripper']], x[x_inds['baxter', 'lArmPose']], x[x_inds['baxter', 'lGripper']], -x[x_inds['baxter', 'lGripper']]]

    def _get_simulator_state(self, x_inds, dX, motor=False):
        model = self.pos_model if not motor else self.motor_model
        X = np.zeros((dX,))

        for param in self.params:
            if param._type != "Robot":
                param_ind = model.body_names.index(param.name)
                if (param.name, "pose") in self.plan.state_inds:
                    X[x_inds[param.name, 'pose']] = model.data.xpos[param_ind].flatten() - np.array([0,0, MUJOCO_MODEL_Z_OFFSET])
                if (param.name, "rotation") in self.plan.state_inds:
                    quat = model.data.xquat[param_ind].flatten()
                    rotation = [np.arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2)), np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1])), \
                                np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))]

                    X[x_inds[param.name, 'rotation']] = rotation

            elif param._type == "Robot":
                robot_name = param.name
                right_arm = model.data.qpos[1:8]
                X[x_inds[('baxter', 'rArmPose')]] = right_arm.flatten()
                X[x_inds[('baxter', 'rGripper')]] = model.data.qpos[8, 0]

                left_arm = model.data.qpos[10:17]
                X[x_inds[('baxter', 'lArmPose')]] = left_arm.flatten()
                X[x_inds[('baxter', 'lGripper')]] = model.data.qpos[17, 0]

                X[x_inds[('baxter', 'rArmPose__vel')]] = model.data.qvel[1:8].flatten()
                X[x_inds[('baxter', 'rGripper__vel')]] = model.data.qvel[8]
                X[x_inds[('baxter', 'lArmPose__vel')]] = model.data.qvel[10:17].flatten()
                X[x_inds[('baxter', 'lGripper__vel')]] = model.data.qvel[17]

        return X
    
    def _get_obs(self, cond, t):
        o_t = np.zeros((self.plan.symbolic_bound))
        return o_t

    # def sample(self, policy, condition, on_policy=True, verbose=False, save=True, noisy=False):
    #     '''
    #         Take a sample for an entire trajectory for the given condition
    #     '''
    #     self.current_cond = condition
    #     noise = np.random.uniform(-1, 1, (self.T, self.dU)) * 0.0001 # Trick to ensure covariance maintains rank
    #     if on_policy:
    #         sample = self.sample_joint_trajectory_policy(policy, condition, noise)
    #     else:
    #         sample = self.sample_joint_angle_trajectory(condition)

    #     if save:
    #         self._samples[condition].append(sample)
    #     return sample

    def sample(self, policy, condition, on_policy=True, save_global=False, verbose=False, save=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.current_cond = condition
        x0 = self.init_plan_states[condition]
        sample = Sample(self)
        if on_policy:
            if self.stochastic_conditions and self.save_global:
                self.replace_cond(condition)

            print 'Starting on-policy sample for condition {0}.'.format(condition)
            if noisy:
                noise = np.random.uniform(-1, 1, (self.T, self.dU))
            else:
                noise = np.zeros((self.T, self.dU))

            self._set_simulator_state(x0[0], self.plan)
            last_success_X = x0[0]
            for t in range(self.T):
                X = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
                U = policy.act(X, X, t, noise[0])
                sample.set(STATE_ENUM, X, t)
                sample.set(OBS_ENUM, X, t)
                sample.set(ACTION_ENUM, U, t)
                sample.set(NOISE_ENUM, noise[0], t)
                success = self.run_policy_step(U)
                if success:
                    last_success_X = X
                else:
                    self._set_simulator_state(last_success_X, self.plan)
                if not t % utils.MUJOCO_STEPS_PER_SECOND and self.viewer:
                    self.viewer.loop_once()
            if save_global:
                self.cond_global_pol_sample[condition] = sample
            print 'Finished on-policy sample.\n'.format(condition)
        else:
            success = self.sample_joint_trajectory(condition, sample)

        if save:
            self._samples[condition].append(sample)
        return sample

    # def sample_joint_angle_trajectory(self, condition):
    #     sample = Sample(self)

    #     # Initialize the plan for the given condition
    #     x0 = self.init_plan_states[condition]

    #     first_act = self.plan.actions[x0[1][0]]
    #     last_act = self.plan.actions[x0[1][1]]
    #     init_t = first_act.active_timesteps[0]
    #     final_t = last_act.active_timesteps[1]

    #     old_first_pose = first_act.params[1]
    #     first_act.params[1] = self.plan.params['robot_init_pose'] # Cloth world specific
    #     utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #     utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], final_t)
    #     for param in x0[2]:
    #         self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
    #         self.plan.params[param].rotation[:,:] = x0[0][self.plan.state_inds[(param, 'rotation')]].reshape(3,1)
    #     # self.plan._determine_free_attrs()
    #     success = self.solver._backtrack_solve(self.plan)

    #     X = x0[0][:self.plan.symbolic_bound]
    #     U_x0 = np.zeros(self.plan.dU)
    #     utils.fill_vector(self.params, self.plan.action_inds, U_x0, init_t)
    #     self._set_simulator_state(x0[0], self.plan)
    #     print "Reset Mujoco Sim to Condition {0}".format(condition)
    #     for ts in range(init_t, final_t):
    #         U_x1 = np.zeros(self.plan.dU)
    #         utils.fill_vector(self.params, self.plan.action_inds, U_x1, ts+1)
    #         U = U_x1 - U_x0
    #         # U_x0 = U_x1

    #         sample.set(STATE_ENUM, X, ts)
    #         sample.set(OBS_ENUM, X, ts)
    #         sample.set(ACTION_ENUM, U_x1, ts)
    #         sample.set(NOISE_ENUM, np.zeros((self.plan.dU)), ts)
    #         self.run_traj_policy_step(U_x1, self.plan.action_inds, mode='state_abs')
    #         X = self._get_simulator_state(self.plan.state_inds, self.plan.dX)[:self.plan.symbolic_bound]
    #         utils.set_params_attrs(self.params, self.plan.state_inds, X, ts+1)
    #         utils.fill_vector(self.params, self.plan.action_inds, U_x0, ts+1)
    #         obs = X

    #     first_act.params[1] = old_first_pose
    #     return sample

    # def sample_joint_trajectory_policy(self, pol, cond, noise, plan=None):
    #     sample = Sample(self)

    #     if not plan:
    #         x0 = self.init_plan_states[cond]
    #         self._set_simulator_state(x0[0], self.plan)
    #         print "Reset Mujoco Sim to Condition {0}".format(cond)

    #         first_act = self.plan.actions[x0[1][0]]
    #         last_act = self.plan.actions[x0[1][1]]

    #         init_t = first_act.active_timesteps[0]
    #         final_t = last_act.active_timesteps[1]

    #         plan = self.plan
    #         X = x0[0][:self.plan.symbolic_bound]
    #     else:
    #         X = np.zeros(plan.symbolic_bound)
    #         utils.fill_vector(self.params, plan.state_inds, X, 0)
    #         first_act = plan.actions[0]
    #         last_act = plan.actions[-1]
    #         init_t = first_act.active_timesteps[0]
    #         final_t = last_act.active_timesteps[1]

    #     obs = X
    #     for ts in range(0, final_t-init_t):
    #         U = pol.act(X, obs, ts, noise[ts, :])
    #         sample.set(STATE_ENUM, X, ts)
    #         sample.set(OBS_ENUM, obs, ts)
    #         sample.set(ACTION_ENUM, U, ts)
    #         sample.set(NOISE_ENUM, noise[ts, :], ts)
    #         self.run_traj_policy_step(U, plan.action_inds, mode='state_abs')
    #         X = self._get_simulator_state(plan.state_inds, plan.dX)[:plan.symbolic_bound]
    #         obs = X

        # sample.set(STATE_ENUM, X, ts+1)
        # sample.set(OBS_ENUM, X, ts+1)
        # sample.set(ACTION_ENUM, np.zeros((plan.dU)), ts+1)
        # sample.set(NOISE_ENUM, np.zeros((plan.dU)), ts+1)

        # return sample

    # # TODO: When switching to torques, change this
    # def run_traj_policy_step(self, u, u_inds, mode='state_delta', timelimit=1.0):
    #     if mode == 'state_abs':
    #         cur_right_joints = self.pos_model.data.qpos[1:8].flatten()
    #         cur_left_joints = self.pos_model.data.qpos[10:17].flatten()
    #         next_right_pose = u[u_inds[('baxter', 'rArmPose')]]
    #         next_left_pose = u[u_inds[('baxter', 'lArmPose')]]
    #         r_grip = u[u_inds[('baxter', 'rGripper')]]
    #         l_grip = u[u_inds[('baxter', 'lGripper')]]
    #         if r_grip > const.GRIPPER_CLOSE_VALUE:
    #             r_grip = const.GRIPPER_OPEN_VALUE
    #         else:
    #             r_grip = 0.001

    #         if l_grip > const.GRIPPER_CLOSE_VALUE:
    #             l_grip = const.GRIPPER_OPEN_VALUE
    #         else:
    #             l_grip = 0.001
    #     else:
    #         cur_right_joints = self.pos_model.data.qpos[1:8].flatten()
    #         cur_left_joints = self.pos_model.data.qpos[10:17].flatten()
    #         next_right_pose = cur_right_joints.flatten() + u[u_inds[('baxter', 'rArmPose')]]
    #         next_left_pose = cur_left_joints.flatten() + u[u_inds[('baxter', 'lArmPose')]]

    #         if u[u_inds[('baxter', 'rGripper')]] > 0.005:
    #             r_grip = const.GRIPPER_OPEN_VALUE
    #         elif u[u_inds[('baxter', 'rGripper')]] < -0.005:
    #             r_grip = 0.001
    #         else:
    #             r_grip = self.pos_model.data.qpos[8, 0]

    #         if u[u_inds[('baxter', 'lGripper')]] > 0.005:
    #             l_grip = const.GRIPPER_OPEN_VALUE
    #         elif u[u_inds[('baxter', 'lGripper')]] < -0.005:
    #             l_grip = 0.001
    #         else:
    #             l_grip = self.pos_model.data.qpos[17, 0]

    #     self.pos_model.data.ctrl = np.r_[next_right_pose, r_grip, -r_grip, next_left_pose, l_grip, -l_grip].reshape(-1,1)
    #     self.avg_vels = np.zeros((18,))
    #     steps = 0
    #     error_limit = 0.01
    #     if self.viewer:
    #         self.viewer.loop_once()
    #     while(np.any(np.abs(cur_right_joints - next_right_pose) > error_limit) or np.any(np.abs(cur_left_joints - next_left_pose) > error_limit)):
    #         # Avoid crashing from excess collisions
    #         if self.pos_model.data.ncon >= N_CONTACT_LIMIT:
    #             print 'Collision Limit Exceeded'
    #             x0 = self.init_plan_states[self.current_cond]
    #             self._set_simulator_state(x0[0], self.plan)
    #             break
    #         self.pos_model.step()
    #         cur_right_joints = self.pos_model.data.qpos[1:8].flatten()
    #         cur_left_joints = self.pos_model.data.qpos[10:17].flatten()
    #         if hasattr(self.pos_model, 'qvel'):
    #             self.avg_vels += self.pos_model.qvel[1:].flatten()
    #         steps += 1
    #         if not steps % 100:
    #             error_limit += 0.05
    #         import ipdb; ipdb.set_trace()
    #     # print '\nSteps in simulation: ', steps
    #     # print 'Contacts: ', self.pos_model.data.ncon
    #     # print 'Error Limit: ', error_limit
    #     # print 'Right Joint Errors: ', np.round(np.abs(cur_right_joints - next_right_pose), 3)
    #     # print 'Left Joint Errors: ', np.round(np.abs(cur_left_joints - next_left_pose), 3), '\n'
    #     self.avg_vels /= (steps or 1)

    def sample_joint_trajectory(self, condition, sample, bootstrap=False):
        # Initialize the plan for the given condition
        x0 = self.init_plan_states[condition]

        first_act = self.plan.actions[x0[1][0]]
        last_act = self.plan.actions[x0[1][1]]
        init_t = first_act.active_timesteps[0]
        final_t = last_act.active_timesteps[1]

        old_first_pose = first_act.params[1]
        first_act.params[1] = self.plan.params['robot_init_pose'] # Cloth world specific
        utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
        utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], final_t)

        # Perpetuate stationary objects' locations
        for param in x0[2]:
            self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
            # self.plan.params[param].rotation[:,:] = x0[0][self.plan.state_inds[(param, 'rotation')]].reshape(3,1)

        if bootstrap:
            # In order to bootstrap the process, start by solving the motion plans from scratch
            if self.initial_samples:
                success = self.solver._backtrack_solve(self.plan)
                traj = np.zeros((self.plan.horizon, self.plan.dX))
                utils.fill_vector(self.symbols, self.plan.state_inds, traj[0], 0)
                for t in range(0, self.plan.horizon):
                    utils.fill_vector(self.params, self.plan.state_inds, traj[t], t)
                self.cond_optimal_traj[condition] = traj
            else:
                # utils.fill_trajectory_from_sample(self.cond_optimal_traj_sample[condition], self.plan)
                traj = self.cond_optimal_traj[condition]
                utils.set_params_attrs(self.symbols, self.plan.state_inds, traj[0], 0)
                for t in range(0, self.plan.horizon):
                    utils.set_params_attrs(self.params, self.plan.state_inds, traj[t], t)
                success = self.solver.traj_smoother(self.plan)
        else:
            success = self.solver._backtrack_solve(self.plan)

            while not success and self.initial_samples:
                print "Solve failed."
                self.replace_cond(m)
                x0 = self.init_plan_states[m]
                utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], 0)
                utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], 0)
                for param in x0[2]:
                    self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
                success = self.solver._backtrack_solve(self.plan)

        self._set_simulator_state(x0[0], self.plan)
        print "Reset Mujoco Sim to Condition {0}".format(condition)
        for ts in range(init_t, final_t):
            U = np.zeros(self.plan.dU)
            utils.fill_vector(self.params, self.plan.action_inds, U, ts+1)
            success = self.run_traj_step(U, self.plan.action_inds, sample, ts)
            if not success:
                print 'Collision Limit Exceeded'

        first_act.params[1] = old_first_pose

        if self.initial_samples:
            self.cond_optimal_traj_sample[condition] = sample

        return success

    def run_traj_step(self, u, u_inds, sample, plan_t=0, timelimit=1.0):
        next_right_pose = u[u_inds[('baxter', 'rArmPose')]]
        next_left_pose = u[u_inds[('baxter', 'lArmPose')]]
        r_grip = u[u_inds[('baxter', 'rGripper')]]
        l_grip = u[u_inds[('baxter', 'lGripper')]]
        if r_grip > const.GRIPPER_CLOSE_VALUE:
            r_grip = const.GRIPPER_OPEN_VALUE
        else:
            r_grip = 0.0001

        if l_grip > const.GRIPPER_CLOSE_VALUE:
            l_grip = const.GRIPPER_OPEN_VALUE
        else:
            l_grip = 0.0001

        steps = int(timelimit * utils.MUJOCO_STEPS_PER_SECOND)

        self.pos_model.data.ctrl = np.r_[next_right_pose, r_grip, -r_grip, next_left_pose, l_grip, -l_grip].reshape((18, 1))
        # if self.viewer:
        #     self.viewer.loop_once()
        success = True
        X = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
        last_success_X = X
        qpos_0 = self.pos_model.data.qpos.flatten()
        for t in range(0, steps):
            # Avoid crashing from excess collisions            
            X = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
            sample.set(STATE_ENUM, X, plan_t*steps+t)
            sample.set(OBS_ENUM, X, plan_t*steps+t)
            self.pos_model.step()
            if self.pos_model.data.ncon >= N_CONTACT_LIMIT:
                print 'Collision Limit Exceeded in Position Model.'
                self._set_simulator_state(last_success_X, self.plan)
                qpos_delta = np.zeros((19,))
                success = False
            else:
                last_success_X = X
                qpos = self.pos_model.data.qpos.flatten()
                qpos_delta = qpos - qpos_0
                qpos_0 = qpos
            # self.motor_model.data.ctrl = frc[1:].reshape((-1,1))
            # self.motor_model.step()
            # import ipdb; ipdb.set_trace()
            U = np.zeros((self.plan.dU,))
            # U[u_inds[('baxter', 'rArmPose')]] = qpos[1:8]
            # U[u_inds[('baxter', 'rGripper')]] = qpos[8]
            # U[u_inds[('baxter', 'lArmPose')]] = qpos[10:17]
            # U[u_inds[('baxter', 'lGripper')]] = qpos[17]
            U[u_inds[('baxter', 'rArmPose')]] = next_right_pose
            U[u_inds[('baxter', 'rGripper')]] = r_grip
            U[u_inds[('baxter', 'lArmPose')]] = next_left_pose
            U[u_inds[('baxter', 'lGripper')]] = l_grip
            sample.set(ACTION_ENUM, U, plan_t*steps+t)
            sample.set(NOISE_ENUM, np.zeros((self.plan.dU,)), plan_t*steps+t)
        if self.viewer:
            self.viewer.loop_once()
        return success

    # def run_torque_step(self, u):
    #     u_inds = self.plan.action_inds
    #     r_joints = u[u_inds[('baxter', 'rArmPose')]]
    #     l_joints = u[u_inds[('baxter', 'lArmPose')]]
    #     r_grip = u[u_inds[('baxter', 'rGripper')]]
    #     l_grip = u[u_inds[('baxter', 'lGripper')]]
    #     success = True
    #     if self.motor_model.data.ncon < N_CONTACT_LIMIT:
    #         self.motor_model.data.ctrl = np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1))
    #     else:
    #         print 'Collision Limit Exceeded in Motor Model.'
    #         self.motor_model.data.ctrl = np.zeros((18,1))
    #         success = False
    #     self.motor_model.step()
    #     self.avg_vels = self.motor_model.qvel[1:].flatten() if hasattr(self.motor_model, 'qvel') else np.zeros((18,))
    #     return success

    def run_policy_step(self, u):
        u_inds = self.plan.action_inds
        r_joints = u[u_inds[('baxter', 'rArmPose')]]
        l_joints = u[u_inds[('baxter', 'lArmPose')]]
        r_grip = u[u_inds[('baxter', 'rGripper')]]
        l_grip = u[u_inds[('baxter', 'lGripper')]]
        success = True
        if self.pos_model.data.ncon < N_CONTACT_LIMIT:
            self.pos_model.data.ctrl = np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1)) # self.pos_model.data.qpos[1:] + np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1))
        else:
            print 'Collision Limit Exceeded in Position Model.'
            self.pos_model.data.ctrl = np.zeros((18,1))
            success = False
        self.pos_model.step()
        return success

    def _clip_joint_angles(self, X):
        DOF_limits = self.plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
        lArmPose = X[self.plan.state_inds[('baxter', 'lArmPose')]]
        rArmPose = X[self.plan.state_inds[('baxter', 'rArmPose')]]
        for i in range(7):
            if lArmPose[i] < left_DOF_limits[0][i]:
                lArmPose[i] = left_DOF_limits[0][i]
            if lArmPose[i] > left_DOF_limits[1][i]:
                lArmPose[i] = left_DOF_limits[1][i]
            if rArmPose[i] < right_DOF_limits[0][i]:
                rArmPose[i] = right_DOF_limits[0][i]
            if rArmPose[i] > right_DOF_limits[1][i]:
                rArmPose[i] = right_DOF_limits[1][i]

    def optimize_trajectories(self, alg):
        for m in range(alg.M):
            U = np.zeros((self.T, self.plan.dU))
            traj_distr = alg.cur[m].traj_distr
            traj_sample = self.sample(traj_distr, m, on_policy=False)
            k = np.zeros((traj_distr.T, traj_distr.dU))
            for t in range(traj_distr.T):
                k[t] = traj_sample.get_U(t)
            traj_distr.k = k

        self.initial_samples = False

    def replace_cond(self, cond):
        x0s = get_randomized_initial_state(self.plan)
        self.init_plan_states[cond] = x0s
        self.x0[cond] = x0s[0][:self.plan.symbolic_bound]

    def reset_conditions(self):
        for m in range(len(self.init_plan_states)):
            self.replace_cond(m)

    # def _sample_ee_trajectory(self, condition, noise):
    #     sample = Sample(self)
    #     plan = self.plans[condition]
    #     if plan in self._traj_info_cache:
    #         active_ts, params = self._traj_info_cache[plan]
    #     else:
    #         active_ts, params = utils.get_plan_traj_info(plan)
    #         self._traj_info_cache[plan] = (active_ts, params)

    #     t0 = active_ts[0]
    #     baxter = plan.params['baxter']
    #     actions = filter(plan.actions, lambda a: a.active_timesteps[0] >= t0)

    #     for ts in range(active_ts[0], active_ts[1]):
    #         baxter.set_dof('lArmPose': baxter.lArmPose[:, ts], 'rArmPose': baxter.rArmPose[:, ts])
    #         ee_right = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         baxter.set_dof('lArmPose': baxter.lArmPose[:, ts+1], 'rArmPose': baxter.rArmPose[:, ts+1])
    #         ee_right_next = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left_next = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         right_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_right[:3.:3])
    #         left_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_left[:3,:3])
    #         next_right_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_right_next[:3.:3])
    #         next_left_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_left_next[:3,:3])

    #         right_open = 0 if plan.params['baxter'].lGripper[:, ts+1] < const.GRIPPER_OPEN_VALUE else 1
    #         left_open = 0 if plan.params['baxter'].rGripper[:, ts+1] < const.GRIPPER_OPEN_VALUE else 1

    #         # right_vec = np.r_[ee_right_next[:3,3] - ee_right[:3,3], next_right_rot - right_rot, right_open]
    #         # left_vec = np.r_[ee_left_next[:3,3] - ee_left[:3,3], next_left_rot - left_rot, left_open]
    #         right_vec = np.r_[ee_right_next[:3,3], next_right_rot, right_open]
    #         left_vec = np.r_[ee_left_next[:3,3], next_left_rot, left_open]


    #         sample.set(ACTION_ENUM, np.r_[left_vec, right_vec], ts - active_ts[0])
    #         X = np.zeros((plan.dX,))
    #         utils.fill_vector(params, plan.state_inds, X, ts)
    #         sample.set(STATE_ENUM, X, ts - active_ts[0])
    #         sample.set(NOISE_ENUM, noise, ts - active_ts[0])

    #     return sample
