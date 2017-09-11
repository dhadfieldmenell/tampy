""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

from mujoco_py import mjcore, mjconstants
from mujoco_py.mjlib import mjlib

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
import policy_hooks.policy_solver_utils as utils

'''
Mujoco specific
'''
BASE_POS_XML = '../models/baxter/mujoco/baxter_mujoco_pos.xml'
BASE_MOTOR_XML = '../models/baxter/mujoco/baxter_mujoco.xml'
ENV_XML = 'policy_hooks/current_env.xml'

MUJOCO_TIME_DELTA = 0.01

MUJOCO_JOINT_ORDER = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_e2', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint'\
                                              'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']

MUJOCO_MODEL_Z_OFFSET = -0.686

# TODO: Split this into two agents for the different simulators (Mujooc & Bullet).
class LaundryWorldMujocoAgent(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        self.plans = self._hyperparams['plans']
        self.solver = self._hyperparams['solver']
        self.x0 = self._hyperparams['x0']
        self._traj_info_cache = {}
        self.sim = 'mujoco'
        # self.pos_model = self.setup_mujoco_model(self.plans[0], motor=False)
        self.motor_model = self.setup_mujoco_model(self.plans[0])
        self.viewer = None

        Agent.__init__(self, config)

    def _generate_xml(self, plan, motor=True):
        '''
            Search a plan for cloths, tables, and baskets to create a proper XML file that Mujoco can generate a model from
        '''
        base_xml = xml.parse(BASE_MOTOR_XML) if motor else xml.parse(BASE_POS_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        if plan in self._traj_info_cache:
            active_ts, params = self._traj_info_cache[plan]
        else:
            active_ts, params = utils.get_plan_traj_info(plan)
            self._traj_info_cache[plan] = (active_ts, params)
        real_t = np.sum([t for t in plan.params['baxter'].time[:, active_ts[0]:active_ts[1]+1]])

        # root.append(xml.fromstring(options_xml.format(MUJOCO_TIME_DELTA, real_t/MUJOCO_TIME_DELTA)))

        for param in params:
            if param.is_symbol(): continue
            if param._type == 'Cloth':
                height = param.geom.height
                radius = param.geom.radius
                x, y, z = param.pose[:, active_ts[0]]
                cloth_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1"})
            # We might want to change this; in Openrave we model tables as hovering box so there's no easy translation to Mujoco
            elif param._type == 'Obstacle': 
                length = param.geom.dim[0]
                width = param.geom.dim[1]
                thickness = param.geom.dim[2]
                x, y, z = param.pose[:, active_ts[0]]
                table_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x, y, MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 'type':'box', 'size':"{} {} {}".format(length, width, z+thickness/2.0)})
            elif param._type == 'Basket':
                x, y, z = param.pose[:, active_ts[0]]
                yaw, pitch, roll = param.rotation[:, active_ts[0]]
                basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(roll, pitch, yaw)})
                basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"1", 'diaginertia':"2 1 1"})
                basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})
        base_xml.write(ENV_XML)

    def setup_mujoco_model(self, plan, pos=True, view=False):
        '''
            Create the Mujoco model and intiialize the viewer if desired
        '''
        self._generate_xml(plan, pos)
        model = mjcore.MjModel(ENV_XML)
        if view:
            self.viewer = mjviewer.MjViewer()
            self.viewer.start()
            self.viewer.set_model(model)
            self.viewer.loop_once()
        return model

    def run_policy(self, cond, policy, noise):
        '''
            Run the policy in simulation and get  the joint values for the plan
        '''
        plan = self.plans[cond]
        x0 = self.x0[conds]
        if plan in self._traj_info_cache:
            active_ts, params = self._traj_info_cache[plan]
        else:
            active_ts, params = utils.get_plan_traj_info(plan)
            self._traj_info_cache[plan] = (active_ts, params)
        self._set_simulator_state(x0, plan, active_ts[0])
        trajectory_state = np.zeros((plan.dX, plan.T))
        for t in range(active_ts[0], active_ts[1]+1):
            obs = self._get_obs()
            u = policy.act(x0, obs, noise[:, t-active_ts[0]], t-active_ts[0])
            # The grippers need some special handling as they have binary state (open or close) on the real robot
            u[7] = (u[7] - 0.5)*10
            u[15] = (u[15] - 0.5)*10
            mj_u = np.zeros((18,1))
            mj_u[:8] = u[:8].reshape(-1, 1)
            mj_u[8] = -u[7]
            mj_u[9:17] = u[8:].reshape(-1, 1)
            mj_u[18] = -u[15]
            real_t = plan.params['baxter'].time[:,t]
            self.motor_model.data.ctrl = mj_u
            start_t = self.motor_model.data.time
            cur_t = start_t
            while cur_t < start_t + real_t:
                self.motor_model.step()
                cur_t += 0.01 # Make sure this value matches the time increment used in Mujoco

            for param in params:
                param_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, param.name)
                if param_ind == -1: continue
                pose = self.motor_model.data.xpos[param_ind].flatten()
                quat = self.motor_model.data.xquat[param_ind].flatten()
                rotation = [np.atan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2)), np.arcsin(2*(quat[0]*quat[2] - q[3]*quat[1])), \
                                    np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))]
                trajectory_state[plan.state_inds[(param, 'pose')]] = pose - np.array([0, 0, MUJOCO_MODEL_Z_OFFSET])
                trajectory_state[plan.state_inds[(param, 'rotation')]] = rotation

                if param._type == 'Robot':
                    # Assume Baxter joints, order head, left arm, left gripper, right arm, right gripper
                    trajectory_state[plan.state_inds[(param, 'lArmPose')]] = self.motor_model.data.qpos[1:8].flatten()
                    trajectory_state[plan.state_inds[(param, 'lGripper')] ]= self.motor_model.data.qpos[8][0]
                    trajectory_state[plan.state_inds[(param, 'rArmPose')]] = self.motor_model.data.qpos[10:18].flatten()
                    trajectory_state[plan.state_inds[(param, 'rGripper')]] = self.motor_model.data.qpos[18][0]
                    trajectory_state[plan.state_inds[(param, 'lArmPose__vel')]] = self.motor_model.data.qvel[1:8].flatten()
                    trajectory_state[plan.state_inds[(param, 'lGripper__vel')] ]= self.motor_model.data.qvel[8][0]
                    trajectory_state[plan.state_inds[(param, 'rArmPose__vel')]] = self.motor_model.data.qvel[10:18].flatten()
                    trajectory_state[plan.state_inds[(param, 'rGripper__vel')]] = self.motor_model.data.qvel[18][0]
            return trajectory_state


    def _set_simulator_state(self, x, cond, t):
        '''
            Set the simulator to the state of the specified condition at the specified timestep, except for the robot
        '''
        plan  = self.plans[cond]
        model  = self.motor_model
        xpos = model.data.xpos.copy()
        xquat = model.data.xquat.copy()
        if plan in self._traj_info_cache:
            active_ts, params = self._traj_info_cache[plan]
        else:
            active_ts, params = utils.get_plan_traj_info(self.plans[cond])
            self._traj_info_cache[plan] = (active_ts, params)

        for param in params:
            if param._type != 'Robot':
                param_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, param.name)
                if param_ind == -1: continue
                xpos[param_ind] = param.pose[:, t] + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET])
                xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(param.pose[:,t], param.rotation[:,t])[:3,:3])

        model.data.xpos = xpos
        model.data.xquat = xquat

    
    # TODO: Fill this in
    def _get_obs(self):
        return []


    def _inverse_dynamics(self, plan):
        '''
            Compute the joint forces necessary to achieve the provided trajectory.

            The state is assumed to be ordered (left then right) s0, s1, e0, e1, w0, w1, w2, gripper where gripper is one value in the plan and two in 
            Mujoco.
        '''
        vel, acc = utils.map_trajecotory_to_vel_acc(plan)
        if plan in self._traj_info_cache:
            active_ts, _ = self._traj_info_cache[plan]
        else:
            active_ts, params = utils.get_plan_traj_info(plan)
            self._traj_info_cache[plan] = (active_ts, params)
        T = active_ts[1] - active_ts[0] + 1
        U = np.zeros((self.dU, T))
        if self.simulator == 'mujoco':
            for t in range(active_ts[0], active_ts[1]+1):
                x_t = np.zeros((self.dX))
                utils.fill_vector(map(lambda p: p[0], plan.state_inds.keys()), plan.state_inds, x_t, t)
                self._set_simulator_state(x_t, plan, t)
                self.model.data.qpos  = self._baxter_to_mujoco(plan, t)
                vel_t = np.zeros((18,1))
                vel_t[:8] = vel[:8,t]
                vel_t[8] = -vel[7,t]
                vel_t[9:17] = vel[8:16, t]
                vel_tl[17] = -vel[15, t]
                vel_t = np.r_[0, vel_t] # Mujoco includes the head joint which we don't use
                self.model.data.qvel = vel_t
                acc_t = np.zeros((18,1))
                acc_t[:8] = acc[:8,t]
                acc_t[8] = -acc[7,t]
                acc_t[9:17] = acc[8:16, t]
                acc_t[17] = -acc[15, t]
                acc_t = np.r_[0, acc_t] # Mujoco includes the head joint which we don't use
                self.model.data.qacc = acc_t
                mjlib.mj_inverse(self.model.ptr, self.model.data.ptr)
                qfrc = np.delete(self.model.data.qfrc_inverse, [0, 9, 18], axis=0) # Only want the joints we use
                qfrc[7] = 0 if plan.plan.params['baxter'].lGripper[:, t] < const.GRIPPER_CLOSE_VALUE else 1
                qfrc[15] = 0 if plan.plan.params['baxter'].rGripper[:, t] < const.GRIPPER_OPEN_VALUE else 1
                U[:, t-active_ts[0]] = qfrc
        elif self.simulator == 'bullet':
            # TODO: Fill this in using the bullet physics simulator & openrave inverse dynamics
            pass

        return U

    def _baxter_to_mujoco(self, plan, t):
        baxter = plan.params['baxter']
        return np.r_[0, baxter.lArmPos[:,t], baxter.lGripper[:,t], -baxter.lGripper[:,t], baxter.rArmPose[:,t], baxter.rGripper[:,t], -baxter.rGripper[:,t]]

    def sample(self, policy, condition, verbose=False, save=True, noisy=False):
        '''
            Take samples for an entire trajectory for the given condition
        '''
        sample = Sample(self)
        plan = self.plans[condition]

        if plan in self._traj_info_cache:
            active_ts, params = self._traj_info_cache[plan]
        else:
            active_ts, params = utils.get_plan_traj_info(plan)
            self._traj_info_cache[plan] = (active_ts, params)

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        utils.set_params_attrs(params, plan.state_inds, x0, active_ts[0])

        #TODO: Enforce this sample is close to the global policy
        self.solver.solve(self.plans[condition], n_resamples=5, active_ts=active_ts, force_init=True)
        U = self._inverse_dynamics(plan)
        for t in range(active_ts[0], active_ts[1]+1):
            utils.fill_sample_from_trajectory(sample, plan, U[:, t-active_ts[0]], noise[t-active_ts[0], :], t, self.dX)
        if save:
            self._samples[condition].append(sample)
        return sample
