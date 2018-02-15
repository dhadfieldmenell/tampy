""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

from mujoco_py import mjcore, mjconstants, mjviewer
from mujoco_py.mjtypes import *
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

# MUJOCO_MODEL_Z_OFFSET = -0.686
MUJOCO_MODEL_Z_OFFSET = -0.736

N_CONTACT_LIMIT = 12

left_lb = [-1.701, -2.146, -3.054, -0.049, -3.058, -1.570, -3.058]
left_ub = [1.701, 1.046, 3.054, 2.617, 3.058, 2.093, 3.058]

right_lb = [-1.701, -2.146, -3.054, -0.049, -3.058, -1.570, -3.058]
right_ub = [1.701, 1.046, 3.054, 2.617, 3.059, 2.093, 3.058]

def closest_arm_pose(arm_poses, cur_arm_pose):
    min_change = np.inf
    chosen_arm_pose = None
    cur_arm_pose = np.array(cur_arm_pose).flatten()
    for arm_pose in arm_poses:
        change = np.sum((np.array([1.75, 1.75, 2, 1.5, 2, 1, 1]) * (arm_pose - cur_arm_pose))**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

class LaundryWorldEEAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        
        self.plan = self._hyperparams['plan']
        self.solver = self._hyperparams['solver']
        self.init_plan_states = self._hyperparams['x0s']
        self.num_cloths = self._hyperparams['num_cloths']
        self.x0 = self._hyperparams['x0']
        self.sim = 'mujoco'
        self.viewer = None
        self.pos_model = self.setup_mujoco_model(self.plan, motor=False, view=True)

        self.symbols = filter(lambda p: p.is_symbol(), self.plan.params.values())
        self.params = filter(lambda p: not p.is_symbol(), self.plan.params.values())
        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
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

                # left_arm = model.data.qpos[10:17]
                # X[x_inds[('baxter', 'lArmPose')]] = left_arm.flatten()
                X[x_inds[('baxter', 'lGripper')]] = model.data.qpos[17, 0]

                # right_arm = model.data.qpos[2:9]
                # X[x_inds[('baxter', 'rArmPose')]] = right_arm.flatten()
                X[x_inds[('baxter', 'rGripper')]] = model.data.qpos[9, 0]

                # X[x_inds[('baxter', 'lArmPose__vel')]] = model.data.qvel[10:17].flatten()
                # ee_vels = np.zeros((6,))
                # ee_vels_c = ee_vels.ctypes.data_as(POINTER(c_double))
                # mjlib.mj_objectVelocity(self.pos_model.ptr, self.pos_model.data.ptr, mjconstants.mjOBJ_BODY, self.l_gripper_ind, ee_vels_c, 0)
                # X[x_inds[('baxter', 'ee_left_pos__vel')]] = ee_vels[:3]
                # X[x_inds[('baxter', 'ee_left_rot__vel')]] = ee_vels[3:]
                # X[x_inds[('baxter', 'lGripper__vel')]] = model.data.qvel[17]

                # ee_vels = np.zeros((6,))
                # ee_vels_c = ee_vels.ctypes.data_as(POINTER(c_double))
                # mjlib.mj_objectVelocity(self.pos_model.ptr, self.pos_model.data.ptr, mjconstants.mjOBJ_BODY, self.r_gripper_ind, ee_vels_c, 0)
                # X[x_inds[('baxter', 'ee_right_pos__vel')]] = ee_vels[:3]
                # X[x_inds[('baxter', 'ee_right_rot__vel')]] = ee_vels[3:]
                # X[x_inds[('baxter', 'rGripper__vel')]] = model.data.qvel[8]

                l_pos = self.pos_model.data.xpos[self.left_grip_l_ind].copy()
                l_pos[2] -= MUJOCO_MODEL_Z_OFFSET
                X[x_inds[('baxter', 'ee_left_pos')]] = l_pos
                r_pos = self.pos_model.data.xpos[self.right_grip_r_ind].copy()
                r_pos[2] -= MUJOCO_MODEL_Z_OFFSET
                X[x_inds[('baxter', 'ee_right_pos')]] = r_pos
                X[x_inds[('baxter', 'ee_left_rot')]] = self.pos_model.data.xquat[self.left_grip_l_ind]
                X[x_inds[('baxter', 'ee_right_rot')]] = self.pos_model.data.xquat[self.right_grip_r_ind]

        joints = model.data.qpos.copy()

        return X, np.r_[X[x_inds[('baxter', 'ee_left_pos')]], X[x_inds[('baxter', 'ee_right_pos')]], X[x_inds[('baxter', 'ee_left_rot')]], X[x_inds[('baxter', 'ee_right_rot')]]], joints

    
    def _get_obs(self, cond, t):
        o_t = np.zeros((self.plan.symbolic_bound))
        return o_t


    def sample(self, policy, condition, save_global=False, verbose=False, save=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.plan.params['table'].openrave_body.set_pose([5, 5, 5])
        self.plan.params['basket'].openrave_body.set_pose([-5, 5, 5])
        self.current_cond = condition
        x0 = self.init_plan_states[condition]
        sample = Sample(self)
        print 'Starting on-policy sample for condition {0}.'.format(condition)
        # if self.stochastic_conditions and save_global:
        #     self.replace_cond(condition)

        if noisy:
            noise = np.random.normal(0, 0.022, (self.T, self.dU))
            noise[:, self.plan.action_inds[('baxter', 'lGripper')]] *= 22
            noise[:, self.plan.action_inds[('baxter', 'rGripper')]] *= 22
            noise[:, self.plan.action_inds[('baxter', 'ee_left_rot')]] *= 0.5
            noise[:, self.plan.action_inds[('baxter', 'ee_right_rot')]] *= 0.5
        else:
            noise = np.zeros((self.T, self.dU))

        joints = x0[3] if len(x0) > 3 else []
        self._set_simulator_state(x0[0], self.plan, joints=joints)
        last_success_X = (x0[0], x0[3])
        last_left_ctrl = x0[3][10:17]
        last_right_ctrl = x0[3][1:8]
        jac = np.zeros((self.dU, 18))
        delta = 1
        for t in range(0, self.T, delta):
            X, ee_pos, joints = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
            U = policy.act(X.copy(), X.copy(), t, noise[t])
            steps = min(delta, self.T-t)
            for i in range(steps):
                sample.set(STATE_ENUM, X.copy(), t+i)
                sample.set(OBS_ENUM, X.copy(), t+i)
                sample.set(ACTION_ENUM, U.copy(), t+i)
                sample.set(NOISE_ENUM, noise[t], t+i)
                sample.set(EE_ENUM, ee_pos[:6], t+i)

            ee_left_pos = U[self.plan.action_inds[('baxter', 'ee_left_pos')]] + noise[t, self.plan.action_inds[('baxter', 'ee_left_pos')]]
            ee_left_rot = U[self.plan.action_inds[('baxter', 'ee_left_rot')]] + noise[t, self.plan.action_inds[('baxter', 'ee_left_rot')]]
            # ee_left_pos = np.maximum(ee_left_pos, [0, -0.2, 0.615])
            # ee_left_pos = np.minimum(ee_left_pos, [1.0, 1.0, 1.5])
            ee_right_pos = U[self.plan.action_inds[('baxter', 'ee_right_pos')]] + noise[t, self.plan.action_inds[('baxter', 'ee_right_pos')]]
            ee_right_rot = U[self.plan.action_inds[('baxter', 'ee_right_rot')]] + noise[t, self.plan.action_inds[('baxter', 'ee_right_rot')]]
            # ee_right_pos = np.maximum(ee_right_pos, [0, -1.0, 0.615])
            # ee_right_pos = np.minimum(ee_right_pos, [1.0, 0.2, 1.5])

            ee_left_rot = ee_left_rot / np.linalg.norm(ee_left_rot)
            ee_right_rot = ee_right_rot / np.linalg.norm(ee_right_rot)

            # ee_left_rot_euler = OpenRAVEBody._ypr_from_rot_matrix(openravepy.rotationMatrixFromQuat(ee_left_rot))
            # ee_right_rot_euler = OpenRAVEBody._ypr_from_rot_matrix(openravepy.rotationMatrixFromQuat(ee_right_rot))
            # target_left_poses = self.plan.params['baxter'].openrave_body.get_ik_from_pose(ee_left_pos, ee_left_rot_euler, "left_arm")
            # target_right_poses = self.plan.params['baxter'].openrave_body.get_ik_from_pose(ee_right_pos, ee_right_rot_euler, "right_arm")

            # if len(target_left_poses):
            #     left_ctrl_signal = closest_arm_pose(target_left_poses, joints[9:16])
            #     last_left_ctrl = left_ctrl_signal
            # else:
            #     target_left_poses = self.plan.params['baxter'].openrave_body.get_ik_from_pose(ee_left_pos, [0, np.pi/2, 0], "left_arm")
            #     if len(target_left_poses):
            #         left_ctrl_signal = closest_arm_pose(target_left_poses, joints[9:16])
            #         last_left_ctrl = left_ctrl_signal
            #     else:
            #         left_ctrl_signal = last_left_ctrl

            # if len(target_right_poses):
            #     right_ctrl_signal = closest_arm_pose(target_right_poses, joints[1:8])
            #     last_right_ctrl = right_ctrl_signal
            # else:
            #     right_ctrl_signal = last_right_ctrl
            # right_ctrl_signal = last_right_ctrl

            # trans = np.zeros((4,4))
            # trans[:3,:3] = openravepy.rotationMatrixFromQuat(ee_left_rot)
            # trans[:3,3] = ee_left_pos
            # left_ctrl_signal = self.plan.params['baxter'].openrave_body.env_body.GetManipulator("left_arm").FindIKSolution(trans)

            # trans = np.zeros((4,4))
            # trans[:3,:3] = openravepy.rotationMatrixFromQuat(ee_right_rot)
            # trans[:3,3] = ee_right_pos
            # right_ctrl_signal = self.plan.params['baxter'].openrave_body.env_body.GetManipulator("right_arm").FindIKSolution(trans)

            # r_grip = U[self.plan.action_inds['baxter', 'rGripper']]
            # l_grip = U[self.plan.action_inds['baxter', 'lGripper']]

            # ctrl_signal = np.r_[right_ctrl_signal, r_grip, -r_grip, left_ctrl_signal, l_grip, -l_grip]

            left_vec = ee_left_pos - ee_pos[:3]
            right_vec = ee_right_pos - ee_pos[3:6]
            left_rot_vec = ee_left_rot - ee_pos[6:10]
            right_rot_vec = ee_right_rot - ee_pos[10:14]

            # iteration = 0
            # while ((np.any(np.abs(np.r_[left_vec, right_vec/2]) > 0.05) or np.any(np.abs(np.r_[left_rot_vec, right_rot_vec]) > 0.1)) and iteration < 300):
            #     self.run_policy_step(self.pos_model.data.qpos[1:].flatten() + (ctrl_signal - self.pos_model.data.qpos[1:].flatten())*105)
            #     if not iteration:
            #         self.viewer.loop_once()
            #     iteration += 1
            #     xpos = self.pos_model.data.xpos
            #     xquat = self.pos_model.data.xquat
            #     left_vec = ee_left_pos - xpos[self.left_grip_l_ind]
            #     left_vec[2] += MUJOCO_MODEL_Z_OFFSET
            #     right_vec = ee_right_pos - xpos[self.right_grip_r_ind]
            #     right_vec[2] += MUJOCO_MODEL_Z_OFFSET
            #     left_rot_vec = ee_left_rot - xquat[self.left_grip_l_ind]
            #     right_rot_vec = ee_right_rot - xquat[self.right_grip_r_ind]

            iteration = 0
            while ((np.any(np.abs(np.r_[left_vec, right_vec/2]) > 0.025) or np.any(np.abs(np.r_[left_rot_vec, right_rot_vec]) > 0.05)) and iteration < 120*delta):
                joints[10:17, 0] = np.maximum(np.minimum(joints[10:17, 0], left_ub), left_lb)
                joints[1:8, 0] = np.maximum(np.minimum(joints[1:8, 0], right_ub), right_lb)
                self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': joints[10:17].flatten(), 'rArmPose': joints[1:8].flatten()})
                body = self.plan.params['baxter'].openrave_body.env_body
                l_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in range(2,9)]
                left_jac = np.array([np.cross(joint.GetAxis(), ee_pos[:3] - joint.GetAnchor()) for joint in l_arm_joints]).T.copy()
                r_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in range(10,17)]
                right_jac = np.array([np.cross(joint.GetAxis(), ee_pos[3:6] - joint.GetAnchor()) for joint in r_arm_joints]).T.copy()
                jac[self.plan.action_inds['baxter', 'ee_left_pos'], 9:16] = left_jac
                jac[self.plan.action_inds['baxter', 'ee_right_pos'], 0:7] = right_jac * 0.75
                # jac[self.plan.action_inds['baxter', 'ee_left_pos'], 10] *= 0.5
                # jac[self.plan.action_inds['baxter', 'ee_right_pos'], 1] *= 0.5

                # left_rot_jac = body.CalculateRotationJacobian(19, left_rot_vec)[:, 2:9]
                # right_rot_jac = body.CalculateRotationJacobian(43, right_rot_vec)[:, 10:17]
                left_rot_jac = body.GetManipulator("left_arm").CalculateRotationJacobian()
                right_rot_jac = body.GetManipulator("right_arm").CalculateRotationJacobian()
                jac[self.plan.action_inds['baxter', 'ee_left_rot'], 9:16] = left_rot_jac * 2.1
                jac[self.plan.action_inds['baxter', 'ee_right_rot'], 0:7] = right_rot_jac * 2.1


                jac[:, 0] *= 1.1
                jac[:, 1] *= 0.84
                jac[:, 2] *= 1.5
                jac[:, 3] *= 1.1
                jac[:, 4] *= 1.7
                jac[:, 5] *= 1.2
                # jac[:, 6] *= 0.2
                # jac[:, 7] *= 0.2
                # jac[:, 8] *= 0.2
                jac[:, 9] *= 1.1
                jac[:, 10] *= 0.84
                jac[:, 11] *= 1.5
                jac[:, 12] *= 1.1
                jac[:, 13] *= 1.7
                jac[:, 14] *= 1.2
                # jac[:, 15] *= 0.2
                # jac[:, 16] *= 0.2
                # jac[:, 17] *= 0.2

                jac[self.plan.action_inds['baxter', 'ee_left_pos'][2], :] *= 2.4

                jac = jac / np.linalg.norm(jac)

                u_vec = np.zeros((self.dU,))
                u_vec[self.plan.action_inds['baxter', 'ee_left_pos']] = left_vec
                u_vec[self.plan.action_inds['baxter', 'ee_right_pos']] = right_vec
                u_vec[self.plan.action_inds['baxter', 'ee_left_rot']] = left_rot_vec
                u_vec[self.plan.action_inds['baxter', 'ee_right_rot']] = right_rot_vec

                ctrl_signal = self.pos_model.data.qpos[1:].flatten() + np.dot(jac.T, u_vec).flatten() * 2.4
                ctrl_signal[7] = 0 if U[self.plan.action_inds['baxter', 'rGripper']] <= 0.5 else const.GRIPPER_OPEN_VALUE
                ctrl_signal[16] = 0 if U[self.plan.action_inds['baxter', 'lGripper']] <= 0.5 else const.GRIPPER_OPEN_VALUE

                # import ipdb; ipdb.set_trace()
                self.run_policy_step(ctrl_signal)
                    
                xpos = self.pos_model.data.xpos
                xquat = self.pos_model.data.xquat
                left_vec = ee_left_pos - xpos[self.left_grip_l_ind]
                left_vec[1] += 0.02
                left_vec[2] += (MUJOCO_MODEL_Z_OFFSET)
                right_vec = ee_right_pos - xpos[self.right_grip_r_ind]
                right_vec[2] += (MUJOCO_MODEL_Z_OFFSET)
                left_rot_vec = ee_left_rot - xquat[self.left_grip_l_ind]
                right_rot_vec = ee_right_rot - xquat[self.right_grip_r_ind]
                iteration += 1

                if not iteration % 50:
                    self.viewer.loop_once()
        # if save_global and success:
        #     self.cond_global_pol_sample[condition] = sample
        print 'Finished on-policy sample.\n'.format(condition)

        if save:
            self._samples[condition].append(sample)
        return sample


    def run_policy_step(self, u, grip_cloth=-1, grip_basket=False):
        u_inds = self.plan.action_inds
        r_joints = u[0:7]
        l_joints = u[9:16]
        r_grip = u[7]
        l_grip = u[16]

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

        if np.all((xpos[self.basket_left_ind] - xpos[self.left_grip_l_ind])**2 < [0.0081, 0.0081, 0.0081]) and l_grip < const.GRIPPER_CLOSE_VALUE \
           and np.all((xpos[self.basket_right_ind] - xpos[self.right_grip_r_ind])**2 < [0.0081, 0.0081, 0.0081]) and r_grip < const.GRIPPER_CLOSE_VALUE:
            body_pos[self.basket_ind] = (xpos[self.left_grip_l_ind] + xpos[self.right_grip_r_ind]) / 2.0
            vec = xpos[self.left_grip_l_ind] - xpos[self.right_grip_r_ind]
            yaw = np.arccos(vec[0] / np.sqrt(vec[0]**2+vec[1]**2))
            body_quat[self.basket_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose([0, 0, 0], [yaw, 0, np.pi/2])[:3,:3])
            run_forward = True

        grip_cloth = -1
        if not run_forward:
            for i in range(self.num_cloths):
                if grip_cloth == i or np.all((xpos[self.cloth_inds[i]] - xpos[self.left_grip_l_ind])**2 < [0.0036, 0.0036, 0.0081]) and l_grip < const.GRIPPER_CLOSE_VALUE and xpos[self.left_grip_l_ind][2] > 0.615:
                    body_pos[self.cloth_inds[i]] = (xpos[self.left_grip_l_ind] + xpos[self.left_grip_r_ind]) / 2.0
                    run_forward = True
                    break
            for i in range(self.num_cloths-1, -1, -1):
                if l_grip > const.GRIPPER_CLOSE_VALUE and np.all((xpos[self.cloth_inds[i]] - xpos[self.basket_ind])**2 < [0.06, 0.06, 0.04]) and xpos[self.cloth_inds[i]][2] > 0.65 + MUJOCO_MODEL_Z_OFFSET:
                    pos = xpos[self.cloth_inds[i]].copy()
                    pos[0] += 0.03
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
    #             self.plan.params['baxter'].rGripper[:,:] = 0
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

    #             for p in self.params:
    #                 if p.is_symbol():
    #                     if p not in init_act.params: continue
    #                     p._free_attrs = old_params_free[p]
    #                 else:
    #                     for attr in p._free_attrs:
    #                         p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #             # self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

    #             # if self.initial_opt:
    #             #     'Saving plan...\n'
    #             #     ps.write_plan_to_hdf5('plan_{0}_cloths_condition_{1}.hdf5'.format(self.num_cloths, m), self.plan)
    #             #     pickle.dump(x0, 'plan_{0}_cloths_condition_{1}_init.npy')

    #             # utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
    #             traj_distr = alg.cur[m].traj_distr
    #             k = np.zeros((traj_distr.T, traj_distr.dU))
    #             u = np.zeros((self.plan.dU))
    #             for t in range(init_t, final_t):
    #                 self.plan.params['baxter'].set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:,t+1]})
    #                 ee_trans = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
    #                 ee_pos = ee_trans[:3, 3]
    #                 ee_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_trans[:3,:3])
    #                 l_gripper = self.plan.params['baxter'].lGripper[0, t+1]

    #                 u[self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_pos
    #                 u[self.plan.action_inds[('baxter', 'ee_left_rot')]] = ee_rot
    #                 u[self.plan.action_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0,t+1]

    #                 k[(t-init_t)*utils.POLICY_STEPS_PER_SECOND:(t-init_t+1)*utils.POLICY_STEPS_PER_SECOND] = u
    #             traj_distr.k = k

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
                self.replace_cond(m)
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
            for t in range(0, final_t-init_t, int(1/utils.POLICY_STEPS_PER_SECOND)):
                utils.fill_vector(self.params, self.plan.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)                
                self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:, init_t+t+1], \
                                                                  'lGripper': self.plan.params['baxter'].lGripper[:, init_t+t+1], \
                                                                  'rArmPose': self.plan.params['baxter'].rArmPose[:, init_t+t+1], \
                                                                  'rGripper': self.plan.params['baxter'].rGripper[:, init_t+t+1]})
                self.plan.params['baxter'].openrave_body.set_pose({'pose': self.plan.params['baxter'].pose[:, init_t+t+1]})
                ee_pose = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransformPose()
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_pose[4:]
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'ee_left_rot')]] = ee_pose[:4]
                
                ee_pose = self.plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransformPose()
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'ee_right_pos')]] = ee_pose[4:]
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.action_inds[('baxter', 'ee_right_rot')]] = ee_pose[:4]

                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.state_inds[('baxter', 'lGripper')]] = 0 if self.plan.params['baxter'].lGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.state_inds[('baxter', 'rGripper')]] = 0 if self.plan.params['baxter'].rGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1

                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.plan.state_inds[('baxter', 'pose')]] = self.plan.params['baxter'].pose[:, init_t+t+1]
        else:
            for t in range(0, final_t-init_t):
                utils.fill_vector(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t+init_t)
                tgt_x[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND] = tgt_x[t*utils.POLICY_STEPS_PER_SECOND]
                
                self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:, init_t+t+1], \
                                                                  'lGripper': self.plan.params['baxter'].lGripper[:, init_t+t+1], \
                                                                  'rArmPose': self.plan.params['baxter'].rArmPose[:, init_t+t+1], \
                                                                  'rGripper': self.plan.params['baxter'].rGripper[:, init_t+t+1]})
                ee_pose = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransformPose()
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_pose[4:]
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_left_rot')]] = ee_pose[:4]
                
                ee_pose = self.plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransformPose()
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_right_pos')]] = ee_pose[4:]
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_right_rot')]] = ee_pose[:4]

                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = 0 if self.plan.params['baxter'].lGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'rGripper')]] = 0 if self.plan.params['baxter'].rGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1
            
        alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state'] = tgt_x
        alg.cost[m]._costs[1]._hyperparams['data_types'][utils.ACTION_ENUM]['target_state'] = tgt_u

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
    #         self.plan.params['baxter'].rArmPose[:,:] = 0
    #         self.plan.params['baxter'].rGripper[:,:] = 0

    #         self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

    #         for p in self.params:
    #             if p.is_symbol():
    #                 if p not in init_act.params: continue
    #                 p._free_attrs = old_params_free[p]
    #             else:
    #                 for attr in p._free_attrs:
    #                     p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

    #         tgt_x = np.zeros((self.T, self.plan.symbolic_bound))
    #         tgt_u = np.zeros((self.T, self.plan.dU))
    #         for t in range(init_t, final_t):
    #             utils.fill_vector(self.params, self.plan.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t)
    #             tgt_x[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND] = tgt_x[t*utils.POLICY_STEPS_PER_SECOND]
                
    #             self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:, t+1], 'lGripper': self.plan.params['baxter'].lGripper[:, t+1]})
    #             ee_trans = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_trans[:3,3]
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'ee_left_rot')]] = OpenRAVEBody._ypr_from_rot_matrix(ee_trans[:3,:3])
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t]

    #         alg.cost[m]._hyperparams['data_types'][utils.STATE_ENUM] = tgt_x
    #         alg.cost[m]._hyperparams['data_types'][utils.ACTION_ENUM] = tgt_u

    #     self.initial_opt = False


    # def init_trajectories(self, alg):
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
    #         self.plan.params['baxter'].rArmPose[:,:] = 0
    #         self.plan.params['baxter'].rGripper[:,:] = 0
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

    #         traj_distr = alg.cur[m].traj_distr
    #         tgt_u = np.zeros((self.T, self.plan.dU))
    #         for t in range(init_t, final_t):
    #             self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:, t+1], 'lGripper': self.plan.params['baxter'].lGripper[:, t+1]})
    #             ee_trans = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_trans[:3,3]
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'ee_left_rot')]] = OpenRAVEBody._ypr_from_rot_matrix(ee_trans[:3,:3])
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t]
    #         traj_distr.k = tgt_u

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
    #         self.plan.params['baxter'].rArmPose[:,:] = 0
    #         self.plan.params['baxter'].rGripper[:,:] = 0

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
    #             self.plan.params['baxter'].openrave_body.set_dof({'lArmPose': self.plan.params['baxter'].lArmPose[:, t+1], 'lGripper': self.plan.params['baxter'].lGripper[:, t+1]})
    #             ee_trans = self.plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransform()
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.action_inds[('baxter', 'ee_left_pos')]] = ee_trans[:3,3]
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'ee_left_rot')]] = OpenRAVEBody._ypr_from_rot_matrix(ee_trans[:3,:3])
    #             tgt_u[t*utils.POLICY_STEPS_PER_SECOND:t*utils.POLICY_STEPS_PER_SECOND+utils.POLICY_STEPS_PER_SECOND, self.plan.state_inds[('baxter', 'lGripper')]] = self.plan.params['baxter'].lGripper[0, t]
    #         traj_distr.k = tgt_u

    #     self.initial_opt = False

    def replace_cond(self, cond):
        print "Replacing Condition {0}.\n".format(cond)
        x0s = get_random_initial_pick_place_state(self.plan, self.num_cloths)
        self.init_plan_states[cond] = x0s
        self.x0[cond] = x0s[0][:self.plan.symbolic_bound]


    def replace_all_conds(self):
        for cond in range(len(self.x0)):
            self.replace_cond(cond)


    def get_policy_avg_cost(self):
        cost = 0
        conds = 0
        for m in range(len(self.init_plan_states)):
            pol_sample = self.cond_global_pol_sample[m]
            if not pol_sample: continue
            x0 = self.init_plan_states[m]
            init_act = self.plan.actions[x0[1][0]]
            final_act = self.plan.actions[x0[1][1]]
            init_t = init_act.active_timesteps[0]
            final_t = final_act.active_timesteps[1]
            conds += 1.0
            for t in range(init_t, final_t):
                X = pol_sample.get_X((t-init_t)*utils.MUJOCO_STEPS_PER_SECOND)
                self._clip_joint_angles(X)
                utils.set_params_attrs(self.params, self.plan.state_inds, X, t)
            X = pol_sample.get_X((final_t-init_t)*utils.MUJOCO_STEPS_PER_SECOND-1)
            self._clip_joint_angles(X)
            utils.set_params_attrs(self.params, self.plan.state_inds, X, final_t)
            utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], t)
            cond_costs = utils.get_trajectory_cost(self.plan, init_t, final_t)[0]
            cost += np.sum(cond_costs)
        if conds < 1.0:
            return 1e5
        return float(cost) / (conds * self.T)
