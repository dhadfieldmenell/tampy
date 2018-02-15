""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import cPickle as pickle

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
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
from policy_hooks.baxter_controller import BaxterMujocoController
from policy_hooks.cloth_world_policy_utils import *
from policy_hooks.policy_solver_utils import STATE_ENUM, OBS_ENUM, ACTION_ENUM, NOISE_ENUM, EE_ENUM
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
# MUJOCO_MODEL_Z_OFFSET = -0.761

N_CONTACT_LIMIT = 9

left_lb = [-1.70167994, -2.147, -3.05417994, -0.05, -3.059, -1.57079633, -3.059]
left_ub = [1.70167994, 1.047, 3.05417994, 2.618, 3.059, 2.094, 3.059]


class LaundryWorldClothAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        
        self.plan = self._hyperparams['plan']
        self.solver = self._hyperparams['solver']
        self.init_plan_states = self._hyperparams['x0s']
        self.num_cloths = self._hyperparams['num_cloths']
        self.x0 = self._hyperparams['x0']
        self.sim = 'mujoco'
        self.viewer = None
        self.left_grip_l_ind = -1
        self.left_grip_r_ind = -1
        self.cloth_inds = []
        self.pos_model = self.setup_mujoco_model(self.plan, motor=False, view=True)
        self.symbols = filter(lambda p: p.is_symbol(), self.plan.params.values())
        self.params = filter(lambda p: not p.is_symbol(), self.plan.params.values())
        self.current_cond = 0
        self.global_policy_samples = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']
        self.saved_trajs = np.zeros((len(self.init_plan_states), self.T, self.plan.symbolic_bound))


    def _generate_xml(self, plan, motor=True):
        '''
            Search a plan for cloths, tables, and baskets to create an XML in MJCF format
        '''
        base_xml = xml.parse(BASE_MOTOR_XML) if motor else xml.parse(BASE_POS_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        active_ts = (0, plan.horizon)
        params = plan.params.values()
        contacts = root.find('contact')
        equality = root.find('equality')

        cur_eq_ind = 0
        self.equal_active_inds = {}
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
                xml.SubElement(equality, 'connect', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip', 'anchor': "0 0 0", 'active':'false'})
                self.equal_active_inds[(param.name, 'left_gripper')] = cur_eq_ind
                cur_eq_ind += 1
                # Exclude collisions between the left hand and the cloth to help prevent exceeding the contact limit
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_wrist'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_hand'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_base'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'basket'})
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
                basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(pitch, roll, yaw)})
                basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
                basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})

                basket_left_handle = xml.SubElement(basket_body, 'body', {'name': 'basket_left_handle', 'pos':"{} {} {}".format(0.317, 0, 0), 'euler':'0 0 0'})
                basket_geom = xml.SubElement(basket_left_handle, 'geom', {'type':'sphere', 'size': '0.01'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'basket'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'left_gripper_l_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'left_gripper_r_finger_tip'})
                xml.SubElement(equality, 'connect', {'body1': 'basket_left_handle', 'body2': 'basket', 'anchor': "0 0.317 0", 'active':'true'})
                xml.SubElement(equality, 'connect', {'body1': 'basket_left_handle', 'body2': 'left_gripper_r_finger_tip', 'anchor': "0 0 0", 'active':'false'})
                self.equal_active_inds[('basket', 'left_handle')] = cur_eq_ind
                cur_eq_ind += 1
                self.equal_active_inds[('left_handle', 'left_gripper')] = cur_eq_ind
                cur_eq_ind += 1

                basket_right_handle = xml.SubElement(basket_body, 'body', {'name': 'basket_right_handle', 'pos':"{} {} {}".format(-0.317, 0, 0), 'euler':'0 0 0'})
                basket_geom = xml.SubElement(basket_right_handle, 'geom', {'type':'sphere', 'size': '0.01'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'basket'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'right_gripper_l_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'right_gripper_r_finger_tip'})
                xml.SubElement(equality, 'connect', {'body1': 'basket_right_handle', 'body2': 'basket', 'anchor': "0 -0.317 0", 'active':'true'})
                xml.SubElement(equality, 'connect', {'body1': 'basket_right_handle', 'body2': 'right_gripper_l_finger_tip', 'anchor': "0 0 0", 'active':'false'})
                self.equal_active_inds[('basket', 'right_handle')] = cur_eq_ind
                cur_eq_ind += 1
                self.equal_active_inds[('right_handle', 'right_gripper')] = cur_eq_ind
                cur_eq_ind += 1
        base_xml.write(ENV_XML)


    def setup_mujoco_model(self, plan, motor=False, view=False):
        '''
            Create the Mujoco model and intiialize the viewer if desired
        '''
        self._generate_xml(plan, motor)
        model = mjcore.MjModel(ENV_XML)
        
        self.baxter_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'base')
        self.left_grip_l_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'left_gripper_l_finger_tip')
        self.left_grip_r_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'left_gripper_r_finger_tip')
        self.right_grip_l_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'right_gripper_l_finger_tip')
        self.right_grip_r_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'right_gripper_r_finger_tip')
        self.basket_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'basket')
        self.basket_left_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'basket_left_ind')
        self.basket_right_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'basket_right_ind')
        self.cloth_inds = []
        for i in range(self.num_cloths):
            self.cloth_inds.append(mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'cloth_{0}'.format(i)))

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


    def _set_simulator_state(self, x, plan, motor=False, joints=[]):
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
                if param.name == 'basket' and (param.name, 'rotation') not in plan.state_inds:
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, [0, 0, np.pi/2])[:3,:3])

        model.body_pos = xpos
        model.body_quat = xquat
        x_inds = plan.state_inds
        l_joints = x[x_inds['baxter', 'lArmPose']] if ('baxter', 'lArmPose') in x_inds else joints[10:17]
        r_joints = x[x_inds['baxter', 'rArmPose']] if ('baxter', 'rArmPose') in x_inds else joints[1:8]
        l_grip = x[x_inds['baxter', 'lGripper']] if ('baxter', 'lGripper') in x_inds else joints[17]
        r_grip = x[x_inds['baxter', 'rGripper']] if ('baxter', 'rGripper') in x_inds else joints[8]
        model.data.qpos = self._baxter_to_mujoco(x, x_inds, l_joints, r_joints, l_grip, r_grip).reshape((19,1))
        model.forward()

    def _baxter_to_mujoco(self, x, x_inds, l_joints, r_joints, l_grip, r_grip):
        return np.r_[0, r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip]


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

                left_arm = model.data.qpos[10:17]
                if ('baxter', 'larmPose') in x_inds:
                    X[x_inds[('baxter', 'lArmPose')]] = left_arm.flatten()
                if ('baxter', 'ee_left_pos') in x_inds:
                    X[x_inds['baxter', 'ee_left_pos']] = model.data.xpos[self.left_grip_l_ind]
                if ('baxter', 'lGripper') in x_inds:
                    X[x_inds[('baxter', 'lGripper')]] = model.data.qpos[17, 0]

                right_arm = model.data.qpos[1:8]                
                if ('baxter', 'rArmPose') in x_inds:
                    X[x_inds[('baxter', 'rArmPose')]] = right_arm.flatten()
                if ('baxter', 'ee_right_pos') in x_inds:
                    X[x_inds['baxter', 'ee_right_pos']] = model.data.xpos[self.right_grip_r_ind]
                if ('baxter', 'rGripper') in x_inds:
                    X[x_inds[('baxter', 'rGripper')]] = model.data.qpos[8, 0]

                # X[x_inds[('baxter', 'lArmPose__vel')]] = model.data.qvel[10:17].flatten()
                # X[x_inds[('baxter', 'lGripper__vel')]] = model.data.qvel[17]

        return X, np.r_[model.data.xpos[self.left_grip_l_ind], model.data.xpos[self.right_grip_r_ind]]

    
    def _get_obs(self, cond, t):
        o_t = np.zeros((self.plan.symbolic_bound))
        return o_t


    def sample(self, policy, condition, on_policy=True, save_global=False, verbose=False, save=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.current_cond = condition
        x0 = self.init_plan_states[condition]
        sample = Sample(self)
        if on_policy:
            print 'Starting on-policy sample for condition {0}.'.format(condition)
            # if self.stochastic_conditions and save_global:
            #     self.replace_cond(condition)

            if noisy:
                noise = np.random.normal(0, 0.1, (self.T, self.dU))
                total_noise = np.random.normal(0, 0.1, (self.dU,))
                noise[:, self.plan.action_inds[('baxter', 'lGripper')]] *= 5
                noise[:, self.plan.action_inds[('baxter', 'rGripper')]] *= 5
                total_noise[self.plan.action_inds[('baxter', 'lGripper')]] *= 0
                total_noise[self.plan.action_inds[('baxter', 'rGripper')]] *= 0
                # noise[:, self.plan.action_inds[('baxter', 'lArmPose')]][1] *= 1.25
                # total_noise[self.plan.action_inds[('baxter', 'lArmPose')]][1] *= 2
                # noise[:, self.plan.action_inds[('baxter', 'lArmPose')]][3] *= 1.25
                # total_noise[self.plan.action_inds[('baxter', 'lArmPose')]][3] *= 2
            else:
                noise = np.zeros((self.T, self.dU))
                total_noise = np.zeros((self.dU, ))

            joints = x0[3] if len(x0) > 3 else []
            self._set_simulator_state(x0[0], self.plan, joints=joints)
            # last_success_X = x0[0]
            # r = np.random.uniform(0, 1)
            for t in range(self.T):
                X, ee_pos = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
                U = policy.act(X.copy(), X.copy(), t, noise[t]+total_noise)
                sample.set(STATE_ENUM, X.copy(), t)
                sample.set(OBS_ENUM, X.copy(), t)
                sample.set(ACTION_ENUM, U.copy(), t)
                sample.set(NOISE_ENUM, noise[t]+total_noise, t)
                sample.set(EE_ENUM, ee_pos, t)
                iteration = 0
                while iteration < 200: # and np.any(np.abs(self.pos_model.data.qpos[1:]-self.pos_model.data.ctrl) > 0.05):
                # for delta in range(utils.MUJOCO_STEPS_PER_SECOND/utils.POLICY_STEPS_PER_SECOND):
                    U[self.plan.action_inds[('baxter', 'rGripper')]] = 0 if U[self.plan.action_inds['baxter', 'rGripper']] <= 0.5 else const.GRIPPER_OPEN_VALUE
                    U[self.plan.action_inds[('baxter', 'lGripper')]] = 0 if U[self.plan.action_inds['baxter', 'lGripper']] <= 0.5 else const.GRIPPER_OPEN_VALUE
                    success = self.run_policy_step(U)
                    # if success:
                    #     last_success_X = X
                    # else:
                    #     self._set_simulator_state(last_success_X, self.plan)
                    iteration += 1
                self.viewer.loop_once()
            if save_global:
                self.global_policy_samples[condition] = sample
            print 'Finished on-policy sample.\n'.format(condition)
        else:
            success = self.sample_joint_trajectory(condition, sample)

        if save:
            self._samples[condition].append(sample)
        return sample


    def run_policy(self, policy):
        '''
            Run one action of the policy on the current state
        '''
        X, _ = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
        X_copy = X .copy()
        U = policy.act(X, X_copy, t, 0)
        for delta in range(utils.MUJOCO_STEPS_PER_SECOND/utils.POLICY_STEPS_PER_SECOND):
            success = self.run_policy_step(U)
            if success:
                last_success_X = X
            else:
                self._set_simulator_state(last_success_X, self.plan)
        self.viewer.loop_once()


    def run_policy_step(self, u, grip_cloth=-1, grip_basket=False):
        u_inds = self.plan.action_inds
        r_joints = u[u_inds[('baxter', 'rArmPose')]]
        l_joints = u[u_inds[('baxter', 'lArmPose')]]
        r_grip = u[u_inds[('baxter', 'rGripper')]]
        l_grip = u[u_inds[('baxter', 'lGripper')]]

        success = True

        if self.pos_model.data.ncon < N_CONTACT_LIMIT:
            self.pos_model.data.ctrl = np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1))
        else:
            # print 'Collision Limit Exceeded in Position Model.'
            # self.pos_model.data.ctrl = np.zeros((18,1))
            return False

        # xpos = self.pos_model.data.xpos.copy()
        # eq_active = self.pos_model.eq_active.copy()
        run_forward = False
        # if (np.all((xpos[self.basket_left_ind] - xpos[self.left_grip_l_ind])**2 < [0.0081, 0.0081, 0.0081]) and l_grip < const.GRIPPER_CLOSE_VALUE) and \
        #    (np.all((xpos[self.basket_right_ind] - xpos[self.right_grip_r_ind])**2 < [0.0081, 0.0081, 0.0081]) and r_grip < const.GRIPPER_CLOSE_VALUE):
        #    eq_active[self.equal_active_inds['left_handle', 'left_gripper']] = 1
        #    eq_active[self.equal_active_inds['right_handle', 'right_gripper']] = 1
        #    run_forward = True
        # else:
        #     eq_active[self.equal_active_inds['left_handle', 'left_gripper']] = 0
        #     eq_active[self.equal_active_inds['right_handle', 'right_gripper']] = 0

        # gripped_cloth = -1
        # for i in range(self.num_cloths):
        #     if not run_forward and (np.all((xpos[self.cloth_inds[i]] - xpos[self.left_grip_l_ind])**2 < [0.0081, 0.0081, 0.0081]) and l_grip < const.GRIPPER_CLOSE_VALUE):
        #         eq_active[self.equal_active_inds['cloth_{0}'.format(i), 'left_gripper']] = 1
        #         run_forward = True
        #         gripped_cloth = i
        #         break
        #     elif l_grip > const.GRIPPER_CLOSE_VALUE:
        #         eq_active[self.equal_active_inds['cloth_{0}'.format(i), 'left_gripper']] = 0

        # for i in range(self.num_cloths):
        #     if i != gripped_cloth:
        #         eq_active[self.equal_active_inds['cloth_{0}'.format(i), 'left_gripper']] = 0

        # self.pos_model.eq_active = eq_active

        self.pos_model.step()

        body_pos = self.pos_model.body_pos.copy()
        body_quat = self.pos_model.body_quat.copy()
        xpos = self.pos_model.data.xpos.copy()
        xquat = self.pos_model.data.xquat.copy()

        if ('baxter', 'pose') in u_inds:
            qpos = self.pos_model.data.qpos.copy()
            angle = U[u_inds['baxter', 'pose']]
            quat = np.array([np.cos(angle/2.0), 0, 0, np.sin(angle/2.0)])
            if np.any(quat != xquat[self.baxter_ind]):
                body_quat[self.baxter_ind] = quat
                self.pos_model.body_quat = body_quat
                self.pos_model.forward()

        holding_basket = False
        body_pos = self.pos_model.body_pos.copy()
        body_quat = self.pos_model.body_quat.copy()
        xpos = self.pos_model.data.xpos.copy()
        xquat = self.pos_model.data.xquat.copy()

        if np.all((xpos[self.basket_left_ind] - xpos[self.left_grip_l_ind])**2 < [0.0081, 0.0081, 0.0081]) and l_grip < const.GRIPPER_CLOSE_VALUE \
           and np.all((xpos[self.basket_right_ind] - xpos[self.right_grip_r_ind])**2 < [0.0081, 0.0081, 0.0081]) and r_grip < const.GRIPPER_CLOSE_VALUE:
            body_pos[self.basket_ind] = (xpos[self.left_grip_l_ind] + xpos[self.right_grip_r_ind]) / 2.0
            vec = xpos[self.left_grip_l_ind] - xpos[self.right_grip_r_ind]
            yaw = np.arccos(vec[0] / np.sqrt(vec[0]**2+vec[1]**2))
            body_quat[self.basket_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose([0, 0, 0], [yaw, 0, np.pi/2])[:3,:3])
            holding_basket = True
            run_forward = True

        grip_cloth = -1
        if not holding_basket:
            for i in range(self.num_cloths):
                if np.all((xpos[self.cloth_inds[i]] - xpos[self.left_grip_l_ind])**2 < [0.0036, 0.0036, 0.0016]) and l_grip < const.GRIPPER_CLOSE_VALUE:
                    body_pos[self.cloth_inds[i]] = (xpos[self.left_grip_l_ind] + xpos[self.left_grip_r_ind]) / 2.0
                    run_forward = True
                    break
            for i in range(self.num_cloths-1, -1, -1):
                if l_grip > const.GRIPPER_CLOSE_VALUE and np.all((xpos[self.cloth_inds[i]] - xpos[self.basket_ind])**2 < [0.06, 0.06, 0.04]) and xpos[self.cloth_inds[i]][2] > 0.65 + MUJOCO_MODEL_Z_OFFSET:
                    pos = xpos[self.cloth_inds[i]].copy()
                    pos[0] -= 0.03
                    pos[1] -= 0.04
                    pos[2] = 0.67 + MUJOCO_MODEL_Z_OFFSET
                    body_pos[self.cloth_inds[i]] = pos
                    run_forward = True
                    break

        if run_forward:
            self.pos_model.body_pos = body_pos
            self.pos_model.body_quat = body_quat
            self.pos_model.forward()

        return success


    # def _clip_joint_angles(self, X):
    #     DOF_limits = self.plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
    #     left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
    #     right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
    #     X[self.plan.state_inds[('baxter', 'lArmPose')]] = np.maximum(X[self.plan.state_inds[('baxter', 'lArmPose')]], left_DOF_limits[0])
    #     X[self.plan.state_inds[('baxter', 'lArmPose')]] = np.minimum(X[self.plan.state_inds[('baxter', 'lArmPose')]], left_DOF_limits[1])


    # def optimize_trajectories(self, alg, reuse=False):
    #     ps = PlanSerializer()
    #     pd = PlanDeserializer()

    #     if reuse:
    #         pass
    #     else:
    #         for m in range(0, alg.M):
    #             x0 = self.init_plan_states[m]
    #             init_act = self.plan.actions[x0[1][0]]
    #             final_act = self.plan.actions[x0[1][1]]
    #             init_t = init_act.active_timesteps[0]
    #             final_t = final_act.active_timesteps[1]

    #             utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #             utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #             for param in x0[2]:
    #                 self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

    #             old_params_free = {}
    #             for p in self.params:
    #                 if p.is_symbol():
    #                     if p not in init_act.params: continue
    #                     old_params_free[p] = p._free_attrs
    #                     p._free_attrs = {}
    #                     for attr in old_params_free[p].keys():
    #                         p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
    #                 else:
    #                     p_attrs = {}
    #                     old_params_free[p] = p_attrs
    #                     for attr in p._free_attrs:
    #                         p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
    #                         p._free_attrs[attr][:, init_t] = 0

    #             self.current_cond = m
    #             self.plan.params['baxter'].rArmPose[:,:] = 0
    #             # self.plan.params['baxter'].rGripper[:,:] = 0

    #             print '\n\n\n\nReoptimizing at condition {0}.\n'.format(m)
    #             success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

    #             while self.initial_opt and not success:
    #                 print "Solve failed."
    #                 self.replace_cond(m, self.num_cloths)
    #                 x0 = self.init_plan_states[m]
    #                 utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #                 utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #                 for param in x0[2]:
    #                     self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
    #                 success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

    #             if not self.initial_opt:
    #                 self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)
    #                 # self.solver.optimize_against_global(self.plan, cond=m)

    #             for p in self.params:
    #                 if p.is_symbol():
    #                     if p not in init_act.params: continue
    #                     p._free_attrs = old_params_free[p]
    #                 else:
    #                     for attr in p._free_attrs:
    #                         p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #             # if self.cond_global_pol_sample[m]:
    #             #     self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

    #             # if self.initial_opt:
    #             #     'Saving plan...\n'
    #             #     ps.write_plan_to_hdf5('plan_{0}_cloths_condition_{1}.hdf5'.format(self.num_cloths, m), self.plan)
    #             #     pickle.dump(x0, 'plan_{0}_cloths_condition_{1}_init.npy')

    #             # utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #             traj_distr = alg.cur[m].traj_distr
    #             k = np.zeros((traj_distr.T, traj_distr.dU))
    #             for t in range(init_t, final_t):
    #                 u = np.zeros((self.plan.dU))
    #                 utils.fill_vector(self.params, self.plan.action_inds, u, t+1)
    #                 k[(t-init_t)*utils.POLICY_STEPS_PER_SECOND:(t-init_t+1)*utils.POLICY_STEPS_PER_SECOND] = u
    #             traj_distr.k = k

    #     self.initial_opt = False


    # def init_trajectories(self, alg, use_single_cond=False):
    #     for m in range(0, alg.M):
    #         x0 = self.init_plan_states[m]
    #         init_act = self.plan.actions[x0[1][0]]
    #         final_act = self.plan.actions[x0[1][1]]
    #         init_t = init_act.active_timesteps[0]
    #         final_t = final_act.active_timesteps[1]

    #         utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #         utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #         for param in x0[2]:
    #             self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

    #         old_params_free = {}
    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 old_params_free[p] = p._free_attrs
    #                 p._free_attrs = {}
    #                 for attr in old_params_free[p].keys():
    #                     p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
    #             else:
    #                 p_attrs = {}
    #                 old_params_free[p] = p_attrs
    #                 for attr in p._free_attrs:
    #                     p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
    #                     p._free_attrs[attr][:, init_t] = 0

    #         self.current_cond = m
    #         # self.plan.params['baxter'].rArmPose[:,:] = 0
    #         # self.plan.params['baxter'].rGripper[:,:] = 0
    #         success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

    #         while self.initial_opt and not success:
    #             print "Solve failed."
    #             self.replace_cond(m, self.num_cloths)
    #             x0 = self.init_plan_states[m]
    #             utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #             utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #             for param in x0[2]:
    #                 self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
    #             success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 p._free_attrs = old_params_free[p]
    #             else:
    #                 for attr in p._free_attrs:
    #                     p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #         if not use_single_cond:
    #             traj_distr = alg.cur[m].traj_distr
    #             tgt_u = np.zeros((self.T, self.plan.dU))
    #             for t in range(init_t, final_t):
    #                 utils.fill_vector(self.params, self.plan.state_inds, self.saved_trajs[m, t], t)
    #                 tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,t+1].copy()
    #                 tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t+1].copy()
    #             traj_distr.k = tgt_u
    #         else:
    #             for c in range(0, alg.M):
    #                 traj_distr = alg.cur[c].traj_distr
    #                 tgt_u = np.zeros((self.T, self.plan.dU))
    #                 for t in range(init_t, final_t):
    #                     utils.fill_vector(self.params, self.plan.state_inds, self.saved_trajs[c, t], t)
    #                     tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,t+1].copy()
    #                     tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t+1].copy()
    #                 traj_distr.k = tgt_u
    #             break

    #     self.initial_opt = False


    # def update_trajectories(self, alg):
    #     for m in range(0, alg.M):
    #         x0 = self.init_plan_states[m]
    #         init_act = self.plan.actions[x0[1][0]]
    #         final_act = self.plan.actions[x0[1][1]]
    #         init_t = init_act.active_timesteps[0]
    #         final_t = final_act.active_timesteps[1]

    #         utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #         utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #         for param in x0[2]:
    #             self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

    #         old_params_free = {}
    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 old_params_free[p] = p._free_attrs
    #                 p._free_attrs = {}
    #                 for attr in old_params_free[p].keys():
    #                     p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
    #             else:
    #                 p_attrs = {}
    #                 old_params_free[p] = p_attrs
    #                 for attr in p._free_attrs:
    #                     p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
    #                     p._free_attrs[attr][:, init_t] = 0

    #         self.current_cond = m
    #         # self.plan.params['baxter'].rArmPose[:,:] = 0
    #         # self.plan.params['baxter'].rGripper[:,:] = 0

    #         tgt_x = self.saved_trajs[m]
    #         for t in range(init_t+1, final_t):
    #             utils.set_params_attrs(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t)

    #         self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 p._free_attrs = old_params_free[p]
    #             else:
    #                 for attr in p._free_attrs:
    #                     p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #         traj_distr = alg.cur[m].traj_distr
    #         tgt_u = np.zeros((self.T, self.plan.dU))
    #         for t in range(init_t, final_t):
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,t+1].copy()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t+1].copy()
    #         traj_distr.k = tgt_u

    #     self.initial_opt = False


    def init_cost_trajectories(self, alg, center=False, full_solve=True):
        for m in range(0, alg.M):
            x0 = self.init_plan_states[m]
            init_act = self.plan.actions[x0[1][0]]
            final_act = self.plan.actions[x0[1][1]]
            init_t = init_act.active_timesteps[0]
            final_t = final_act.active_timesteps[1]
            joints = x0[3] if len(x0) > 3 else []

            utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
            utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
            if len(joints):
                self.plan.params['baxter'].lArmPose[:, init_t] = joints[10:17]
                self.plan.params['baxter'].rArmPose[:, init_t] = joints[1:8]
                self.plan.params['baxter'].lGripper[:, init_t] = joints[17]
                self.plan.params['baxter'].rGripper[:, init_t] = joints[8]
            for param in x0[2]:
                self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

            old_params_free = {}
            for p in self.params:
                if p.is_symbol():
                    if p not in init_act.params: continue
                    old_params_free[p] = p._free_attrs
                    p._free_attrs = {}
                    for attr in old_params_free[p].keys():
                        p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                else:
                    p_attrs = {}
                    old_params_free[p] = p_attrs
                    for attr in p._free_attrs:
                        p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
                        p._free_attrs[attr][:, init_t] = 0

            self.current_cond = m
            if full_solve:
                success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1], n_resamples=3)
            else:
                success = True
                self.set_plan_from_cost_trajs(alg, init_t, final_t, m)

            while not success:
                print "Solve failed."
                for p in self.params:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        p._free_attrs = old_params_free[p]
                    else:
                        for attr in p._free_attrs:
                            p._free_attrs[attr][:, init_t] = old_params_free[p][attr]
                self.replace_cond(m, self.num_cloths)
                x0 = self.init_plan_states[m]
                init_act = self.plan.actions[x0[1][0]]
                final_act = self.plan.actions[x0[1][1]]
                init_t = init_act.active_timesteps[0]
                final_t = final_act.active_timesteps[1]
                joints = x0[3] if len(x0) > 3 else []

                utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
                utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
                if len(joints):
                    self.plan.params['baxter'].lArmPose[:, init_t] = joints[10:17]
                    self.plan.params['baxter'].rArmPose[:, init_t] = joints[1:8]
                    self.plan.params['baxter'].lGripper[:, init_t] = joints[17]
                    self.plan.params['baxter'].rGripper[:, init_t] = joints[8]
                for param in x0[2]:
                    self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

                old_params_free = {}
                for p in self.params:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        old_params_free[p] = p._free_attrs
                        p._free_attrs = {}
                        for attr in old_params_free[p].keys():
                            p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
                    else:
                        p_attrs = {}
                        old_params_free[p] = p_attrs
                        for attr in p._free_attrs:
                            p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
                            p._free_attrs[attr][:, init_t] = 0
                success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1], n_resamples=3)

            if not self.initial_opt:
                self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

            for p in self.params:
                if p.is_symbol():
                    if p not in init_act.params: continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

            self.set_cost_trajectories(alg, init_t, final_t, m, center=center)

        self.initial_opt = False

    def set_plan_from_cost_trajs(self, alg, init_t, final_t, m):
        tgt_x = alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state']
        if utils.POLICY_STEPS_PER_SECOND < 1:
            for t in range(0, final_t-init_t, (1/utils.POLICY_STEPS_PER_SECOND)):
                utils.set_params_attrs(self.params, self.plan.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
        else:
            for t in range(0, final_t-init_t):
                utils.set_params_attrs(self.params, self.plan.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)

    def set_cost_trajectories(self, alg, init_t, final_t, m, center=False):
        tgt_x = np.zeros((self.T, self.plan.symbolic_bound))
        tgt_u = np.zeros((self.T, self.plan.dU))

        if utils.POLICY_STEPS_PER_SECOND < 1:
            for t in range(0, final_t-init_t, int(1.0 / utils.POLICY_STEPS_PER_SECOND)):
                utils.fill_vector(self.params, self.plan.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,init_t+t+1].copy()
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'rArmPose')]] = self.plan.params['baxter'].rArmPose[:,init_t+t+1].copy()

                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.state_inds[('baxter', 'lGripper')]] = 0 if self.plan.params['baxter'].lGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.state_inds[('baxter', 'rGripper')]] = 0 if self.plan.params['baxter'].rGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
    
        else:
            for t in range(0, final_t-init_t):
                utils.fill_vector(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t+init_t)
                tgt_x[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND] = tgt_x[t*utils.POLICY_STEPS_PER_SECOND]
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,init_t+t+1].copy()
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'rArmPose')]] = self.plan.params['baxter'].rArmPose[:,init_t+t+1].copy()

                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = 0 if self.plan.params['baxter'].lGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'rGripper')]] = 0 if self.plan.params['baxter'].rGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
                        
        alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state'] = tgt_x
        alg.cost[m]._costs[1]._hyperparams['data_types'][utils.ACTION_ENUM]['target_state'] = tgt_u
        self.saved_trajs[m] = tgt_x

        if center:
            traj_distr = alg.cur[m].traj_distr
            traj_distr.k = tgt_u.copy()


    # def update_cost_trajectories(self, alg):
    #     for m in range(0, alg.M):
    #         x0 = self.init_plan_states[m]
    #         init_act = self.plan.actions[x0[1][0]]
    #         final_act = self.plan.actions[x0[1][1]]
    #         init_t = init_act.active_timesteps[0]
    #         final_t = final_act.active_timesteps[1]

    #         utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #         utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
    #         for param in x0[2]:
    #             self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)

    #         old_params_free = {}
    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 old_params_free[p] = p._free_attrs
    #                 p._free_attrs = {}
    #                 for attr in old_params_free[p].keys():
    #                     p._free_attrs[attr] = np.zeros(old_params_free[p][attr].shape)
    #             else:
    #                 p_attrs = {}
    #                 old_params_free[p] = p_attrs
    #                 for attr in p._free_attrs:
    #                     p_attrs[attr] = p._free_attrs[attr][:, init_t].copy()
    #                     p._free_attrs[attr][:, init_t] = 0

    #         self.current_cond = m
    #         # self.plan.params['baxter'].rArmPose[:,:] = 0
    #         # self.plan.params['baxter'].rGripper[:,:] = 0

    #         tgt_x = self.saved_trajs[m]
    #         for t in range(init_t+1, final_t):
    #             utils.set_params_attrs(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t-init_t)

    #         # self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 p._free_attrs = old_params_free[p]
    #             else:
    #                 for attr in p._free_attrs:
    #                     p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #         tgt_x = np.zeros((self.T, self.plan.symbolic_bound))
    #         tgt_u = np.zeros((self.T, self.plan.dU))
    #         for t in range(0, final_t-init_t):
    #             utils.fill_vector(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t+init_t)
    #             tgt_x[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND] = tgt_x[t*utils.POLICY_STEPS_PER_SECOND]
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'lArmPose')]] = self.plan.params['baxter'].lArmPose[:,init_t+t+1].copy()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, init_t+t+1].copy()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'rArmPose')]] = self.plan.params['baxter'].rArmPose[:,init_t+t+1].copy()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'rGripper')]] = self.plan.params['baxter'].rGripper[0, init_t+t+1].copy()
            
    #         alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state'] = tgt_x
    #         alg.cost[m]._costs[1]._hyperparams['data_types'][utils.ACTION_ENUM]['target_state'] = tgt_u
    #         self.saved_trajs[m] = tgt_x

    #     self.initial_opt = False


    def replace_cond(self, cond, num_cloths=1):
        print "Replacing Condition {0}.\n".format(cond)
        # x0s = get_randomized_initial_state_left(self.plan)
        x0s = get_random_initial_pick_place_state(self.plan, num_cloths)
        self.init_plan_states[cond] = x0s
        self.x0[cond] = x0s[0][:self.plan.symbolic_bound]
        self.global_policy_samples[cond] = []


    def replace_all_conds(self, num_cloths=2):
        for cond in range(len(self.x0)):
            self.replace_cond(cond, num_cloths)


    # def get_policy_avg_cost(self):
    #     cost = 0
    #     conds = 0
    #     for m in range(len(self.init_plan_states)):
    #         pol_sample = self.cond_global_pol_sample[m]
    #         if not pol_sample: continue
    #         x0 = self.init_plan_states[m]
    #         init_act = self.plan.actions[x0[1][0]]
    #         final_act = self.plan.actions[x0[1][1]]
    #         init_t = init_act.active_timesteps[0]
    #         final_t = final_act.active_timesteps[1]
    #         conds += 1.0
    #         for t in range(init_t, final_t):
    #             X = pol_sample.get_X((t-init_t)*utils.MUJOCO_STEPS_PER_SECOND)
    #             self._clip_joint_angles(X)
    #             utils.set_params_attrs(self.params, self.plan.state_inds, X, t)
    #         X = pol_sample.get_X((final_t-init_t)*utils.MUJOCO_STEPS_PER_SECOND-1)
    #         self._clip_joint_angles(X)
    #         utils.set_params_attrs(self.params, self.plan.state_inds, X, final_t)
    #         utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], t)
    #         cond_costs = utils.get_trajectory_cost(self.plan, init_t, final_t)[0]
    #         cost += np.sum(cond_costs)
    #     if conds < 1.0:
    #         return 1e5
    #     return float(cost) / (conds * self.T)
