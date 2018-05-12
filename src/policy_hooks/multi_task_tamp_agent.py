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
from gps.sample.sample_list import SampleList

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

from policy_hooks.baxter_controller import BaxterMujocoController
from policy_hooks.cloth_world_policy_utils import *
from policy_hooks.policy_solver_utils import STATE_ENUM, OBS_ENUM, ACTION_ENUM, NOISE_ENUM, EE_ENUM, GRIPPER_ENUM, COLORS_ENUM, TRAJ_HIST_ENUM, TASK_ENUM
from policy_hooks.setup_mjc_model import setup_mjc_model
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

N_CONTACT_LIMIT = 16

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
        # Note: All plans should contain identical sets of parameters
        self.plans = self._hyperparams['plans']
        self.task_breaks = self._hyperparams['task_breaks']
        self.task_encoding = self._hyperparams['task_encoding']
        self.task_durations = self._hyperparams['task_durations']
        self.color_maps = self._hyperparams['color_maps']
        self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.num_cloths = self._hyperparams['num_cloths']
        self.x0 = self._hyperparams['x0']
        self.sim = 'mujoco'
        self.viewer = self._hyperparams['viewer']
        self.pos_model = self.setup_mujoco_model(self.plans[0], motor=False, view=True) if not self._hyperparams['model'] else self._hyperparams['model']

        self.symbols = [filter(lambda p: p.is_symbol(), self.plans[m].params.values()) for m in range(len(self.plans))]
        self.params = [filter(lambda p: not p.is_symbol(), self.plans[m].params.values()) for m in range(len(self.plans))]
        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        self.hist_len = self._hyperparams['hist_len']
        self.traj_hist = None
        self._reset_hist()

        self.optimal_state_traj = [[] for _ in range(len(self.plans))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans))]

        self.get_plan = self._hyperparams['get_plan']

        self.in_left_grip = -1
        self.in_right_grip = -1


    def get_samples(self, condition, task, start=0, end=None):
        if np.abs(start) >= len(self._samples[condition][task]):
            start = 0

        samples = {}
        if end is None:
            for sample in self._samples[condition][task][start:]:
                if sample.init_t not in samples:
                    samples[sample.init_t] = []
                samples[sample.init_t].append(sample)
        else:
            for sample in self._samples[condition][task][start:end]:
                if sample.init_t not in samples:
                    samples[sample.init_t] = []
                samples[sample.init_t].append(sample)

        for ts in samples:
            samples[ts] = SampleList(samples[ts])

        return samples

        
    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples[condition] = {task:[] for task in self.task_encoding.keys()}


    def _generate_xml(self, plan, cond=0, motor=False):
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
                radius = param.geom.radius * 2.5
                x, y, z = param.pose[:, active_ts[0]]
                color = self.color_maps[cond][param.name]
                cloth_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                # cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
                cloth_geom = xml.SubElement(cloth_body, 'geom', {'name': param.name, 'type':'sphere', 'size':"{}".format(radius), 'rgba':'{}'.format(color[1]), 'friction':'1 1 1'})
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
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_wrist'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_hand'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_base'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_l_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_l_finger'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_r_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_r_finger'})
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
                basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(roll, pitch, yaw)})
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
            self.cloth_inds.append(mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'cloth{0}'.format(i)))

        if not self.viewer:
            self.viewer = mjviewer.MjViewer()
            self.viewer.start()
            self.viewer.set_model(model)
            self.viewer.cam.distance = 3
            self.viewer.cam.azimuth = 180.0
            self.viewer.cam.elevation = -37.5
            self.viewer.loop_once()

        self.setup_obs_viewer(model)
        return model


    def replace_model(self, condition):
        self._generate_xml(self.plans[condition], condition)
        self.pos_model = mjcore.MjModel(ENV_XML)
        
        self.left_grip_l_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'left_gripper_l_finger_tip')
        self.left_grip_r_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'left_gripper_r_finger_tip')
        self.right_grip_l_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'right_gripper_l_finger_tip')
        self.right_grip_r_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'right_gripper_r_finger_tip')
        self.basket_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'basket')
        self.basket_left_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'basket_left_ind')
        self.basket_right_ind = mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'basket_right_ind')
        self.cloth_inds = []
        for i in range(self.num_cloths):
            self.cloth_inds.append(mjlib.mj_name2id(self.pos_model.ptr, mjconstants.mjOBJ_BODY, 'cloth{0}'.format(i)))

        if self.viewer:
            self.viewer.set_model(self.pos_model)
            self.viewer.loop_once()

        self.setup_obs_viewer(self.pos_model)


    def setup_obs_viewer(self, model):
        self.obs_viewer = mjviewer.MjViewer(False, utils.IM_W, utils.IM_H)
        self.obs_viewer.start()
        self.obs_viewer.set_model(model)
        self.obs_viewer.cam.distance = 2
        self.obs_viewer.cam.azimuth = 180.0
        self.obs_viewer.cam.elevation = -47.5
        self.viewer.loop_once()

    def _set_simulator_state(self, cond, plan, t, joints=[]):
        '''
            Set the simulator to the state of the specified condition, except for the robot
        '''
        model  = self.pos_model
        xpos = model.body_pos.copy()
        xquat = model.body_quat.copy()
        params = plan.params.values()

        for param in self.params[cond]:
            if param._type != 'Robot':
                param_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, param.name)
                if param_ind == -1: continue
                if (param.name, 'pose') in self.state_inds:
                    pos = param.pose[:,t]
                    xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET]) + np.array([0, 0, 0.025])
                if (param.name, 'rotation') in self.state_inds:
                    rot = param.rotation[:,t]
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, rot)[:3,:3])
                if param.name == 'basket' and (param.name, 'rotation') not in self.state_inds:
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, [0, 0, np.pi/2])[:3,:3])

        model.body_pos = xpos
        model.body_quat = xquat
        x_inds = self.state_inds
        l_joints = plan.params['baxter'].lArmPose[:,t]
        r_joints = plan.params['baxter'].rArmPose[:,t]
        l_grip = plan.params['baxter'].lGripper[:,t]
        r_grip = plan.params['baxter'].rGripper[:,t]
        model.data.qpos = self._baxter_to_mujoco(x_inds, l_joints, r_joints, l_grip, r_grip).reshape((19,1))
        model.forward()

    def _baxter_to_mujoco(self, x_inds, l_joints, r_joints, l_grip, r_grip):
        return np.r_[0, r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip]

    def _get_simulator_state(self, x_inds, cond, dX, motor=False):
        model = self.pos_model if not motor else self.motor_model
        X = np.zeros((dX,))

        for param in self.params[cond]:
            if param._type in ["Cloth", "Obstacle", "Basket"]:
                param_ind = model.body_names.index(param.name)
                if (param.name, "pose") in self.state_inds:
                    X[x_inds[param.name, 'pose']] = model.data.xpos[param_ind].flatten() - np.array([0,0, MUJOCO_MODEL_Z_OFFSET])
                if (param.name, "rotation") in self.state_inds:
                    quat = model.data.xquat[param_ind].flatten()
                    rotation = [np.arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2)), np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1])), \
                                np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))]

                    X[x_inds[param.name, 'rotation']] = rotation

            elif param._type == "Robot":
                robot_name = param.name

                left_arm = model.data.qpos[10:17]
                X[x_inds[('baxter', 'lArmPose')]] = left_arm.flatten()
                X[x_inds[('baxter', 'lGripper')]] = model.data.qpos[17, 0]

                right_arm = model.data.qpos[2:9]
                X[x_inds[('baxter', 'rArmPose')]] = right_arm.flatten()
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

                # l_pos = self.pos_model.data.xpos[self.left_grip_l_ind].copy()
                # l_pos[2] -= MUJOCO_MODEL_Z_OFFSET
                # r_pos = self.pos_model.data.xpos[self.right_grip_r_ind].copy()
                # r_pos[2] -= MUJOCO_MODEL_Z_OFFSET

        joints = model.data.qpos.copy()

        return X, joints


    def _reset_hist(self):
        self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist() if self.hist_len > 0 else None


    def sample(self, policy_map, condition, save_global=False, verbose=False, save=True, use_base_t=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.plans[condition].params['table'].openrave_body.set_pose([5, 5, 5])
        self.plans[condition].params['basket'].openrave_body.set_pose([-5, 5, 5])
        self.current_cond = condition
        # x0 = self.init_plan_states[condition]
        x0 = np.zeros((self.dX,))
        utils.fill_vector(self.params[condition], self.state_inds, x0, 0)                
        num_tasks = len(self.task_encoding.keys())
        cur_task_ind = 0
        next_t, task = self.task_breaks[condition][cur_task_ind]
        policy = policy_map[task]['policy']
        base_t = 0
        self.T = next_t
        sample = Sample(self)
        sample.init_t = 0
        print 'Starting on-policy sample for condition {0}.'.format(condition)
        # if self.stochastic_conditions and save_global:
        #     self.replace_cond(condition)

        color_vec = np.zeros((len(self.color_maps[condition].keys())))
        for cloth_name in self.color_maps[condition]:
            color_vec[int(cloth_name[-1])] = self.color_maps[condition][cloth_name][0] * 100

        attempts = 0
        success = False
        while not success and attempts < 3:
            self._set_simulator_state(condition, self.plans[condition], 0)
            last_successful_pos = self.pos_model.data.qpos.copy()
            # last_success_X = (x0[0], x0[3])
            # last_left_ctrl = x0[3][10:17]
            # last_right_ctrl = x0[3][1:8]

            if noisy:
                noise = generate_noise(self.T, self.dU, self._hyperparams)
            else:
                noise = np.zeros((self.T, self.dU))

            noise[:, self.action_inds['baxter', 'lGripper']] = 0
            noise[:, self.action_inds['baxter', 'rGripper']] = 0


            for t in range(0, (self.plans[condition].horizon-1)*utils.POLICY_STEPS_PER_SECOND):
                if t >= next_t:
                    if save:
                        self._samples[condition][task].append(sample)
                    cur_task_ind += 1
                    next_t, task = self.task_breaks[condition][cur_task_ind]
                    policy = policy_map[task]['policy']
                    self.T = next_t - t
                    sample = Sample(self)
                    sample.init_t = t

                base_t = sample.init_t

                X, joints = self._get_simulator_state(self.state_inds, condition, self.symbolic_bound)

                obs = []
                if OBS_ENUM in self._hyperparams['obs_include']:
                    im = self.get_obs()
                    obs = np.r_[obs, im]
                
                if STATE_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, X]

                if TRAJ_HIST_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, np.array(self.traj_hist).flatten()]

                if use_base_t:
                    U = policy.act(X.copy(), obs, t-base_t, noise[t-base_t])
                else:
                    U = policy.act(X.copy(), obs, t, noise[t-base_t])


                for i in range(1):
                    sample.set(STATE_ENUM, X.copy(), t-base_t+i)
                    if OBS_ENUM in self._hyperparams['obs_include']:
                        sample.set(OBS_ENUM, im.copy(), t-base_t+i)
                    sample.set(ACTION_ENUM, U.copy(), t-base_t+i)
                    sample.set(NOISE_ENUM, noise[t-base_t], t-base_t+i)
                    sample.set(COLORS_ENUM, color_vec.copy(), t-base_t+i)
                    sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t-base_t+i)
                    task_vec = np.zeros((num_tasks,))
                    task_vec[self.task_encoding[task]] = 1
                    sample.set(TASK_ENUM, task_vec, t-base_t+i)


                self.traj_hist.pop(0)
                self.traj_hist.append(U)

                left_vec = U[self.action_inds['baxter', 'lArmPose']] - X[self.state_inds['baxter', 'lArmPose']]
                right_vec = U[self.action_inds['baxter', 'rArmPose']] - X[self.state_inds['baxter', 'rArmPose']]

                iteration = 0
                while np.any(np.abs(np.r_[left_vec, right_vec]) > 0.02) and iteration < 200:
                    success = self.run_policy_step(U, last_successful_pos)
                    last_successful_pos[:] = self.pos_model.data.qpos.copy()[:]
                    iteration += 1

                self.viewer.loop_once()

                if not success:
                    attempts += 1
                    cur_task_ind = 0
                    next_t, task = self.task_breaks[condition][cur_task_ind]
                    policy = policy_map[task]['policy']
                    self.T = next_t
                    sample = Sample(self)
                    sample.init_t = 0

                    break
            print 'Finished on-policy sample.\n'.format(condition)

        if save:
            self._samples[condition][task].append(sample)
        return sample


    def sample_task(self, policy, condition, x0, task, save_global=False, verbose=False, save=True, use_base_t=True, noisy=True):
        '''
            Take a sample for a given condition
        '''
        self.plans[condition].params['table'].openrave_body.set_pose([5, 5, 5])
        self.plans[condition].params['basket'].openrave_body.set_pose([-5, 5, 5])
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(self.plans[condition].params[param], attr)[:,1] = x0[self.state_inds[param, attr]]
        num_tasks = len(self.task_encoding.keys())
        cur_task_ind = 0
        self.T = self.task_durations[task]
        base_t = 0
        sample = Sample(self)
        sample.init_t = 0
        print 'Starting on-policy sample for condition {0}.'.format(condition)
        # if self.stochastic_conditions and save_global:
        #     self.replace_cond(condition)

        color_vec = np.zeros((len(self.color_maps[condition].keys())))
        for cloth_name in self.color_maps[condition]:
            color_vec[int(cloth_name[-1])] = self.color_maps[condition][cloth_name][0] * 100

        attempts = 0
        success = False
        while not success and attempts < 3:
            self._set_simulator_state(condition, self.plans[condition], 1)
            last_successful_pos = self.pos_model.data.qpos.copy()
            # last_success_X = (x0[0], x0[3])
            # last_left_ctrl = x0[3][10:17]
            # last_right_ctrl = x0[3][1:8]

            if noisy:
                noise = generate_noise(self.T, self.dU, self._hyperparams)
            else:
                noise = np.zeros((self.T, self.dU))

            noise[:, self.action_inds['baxter', 'lGripper']] = 0
            noise[:, self.action_inds['baxter', 'rGripper']] = 0


            for t in range(0, self.T*utils.POLICY_STEPS_PER_SECOND):
                base_t = sample.init_t

                X, joints = self._get_simulator_state(self.state_inds, condition, self.symbolic_bound)

                obs = []
                if OBS_ENUM in self._hyperparams['obs_include']:
                    im = self.get_obs()
                    obs = np.r_[obs, im]
                
                if STATE_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, X]

                if TRAJ_HIST_ENUM in self._hyperparams['obs_include']:
                    obs = np.r_[obs, np.array(self.traj_hist).flatten()]

                if use_base_t:
                    U = policy.act(X.copy(), obs, t-base_t, noise[t-base_t])
                else:
                    U = policy.act(X.copy(), obs, t, noise[t-base_t])


                for i in range(1):
                    sample.set(STATE_ENUM, X.copy(), t-base_t+i)
                    if OBS_ENUM in self._hyperparams['obs_include']:
                        sample.set(OBS_ENUM, im.copy(), t-base_t+i)
                    sample.set(ACTION_ENUM, U.copy(), t-base_t+i)
                    sample.set(NOISE_ENUM, noise[t-base_t], t-base_t+i)
                    sample.set(COLORS_ENUM, color_vec.copy(), t-base_t+i)
                    sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t-base_t+i)
                    task_vec = np.zeros((num_tasks,))
                    task_vec[self.task_encoding[task]] = 1
                    sample.set(TASK_ENUM, task_vec, t-base_t+i)


                self.traj_hist.pop(0)
                self.traj_hist.append(U)

                left_vec = U[self.action_inds['baxter', 'lArmPose']] - X[self.state_inds['baxter', 'lArmPose']]
                right_vec = U[self.action_inds['baxter', 'rArmPose']] - X[self.state_inds['baxter', 'rArmPose']]

                iteration = 0
                while np.any(np.abs(np.r_[left_vec, right_vec]) > 0.02) and iteration < 200:
                    success = self.run_policy_step(U, last_successful_pos)
                    last_successful_pos[:] = self.pos_model.data.qpos.copy()[:]
                    iteration += 1

                self.viewer.loop_once()

                if not success:
                    attempts += 1
                    sample = Sample(self)
                    sample.init_t = 0

                    break
            print 'Finished on-policy sample.\n'.format(condition)

        if save:
            self._samples[condition][task].append(sample)
        return sample

    def run_policy_step(self, u, last_success, grip_cloth=-1, grip_basket=False):
        u_inds = self.action_inds
        r_joints = u[u_inds['baxter', 'rArmPose']]
        l_joints = u[u_inds['baxter', 'lArmPose']]
        r_grip = u[u_inds['baxter', 'rGripper']]
        l_grip = u[u_inds['baxter', 'lGripper']]

        success = True

        if self.pos_model.data.ncon < N_CONTACT_LIMIT:
            self.pos_model.data.ctrl = np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1))
        else:
            # print 'Collision Limit Exceeded in Position Model.'
            # self.pos_model.data.ctrl = np.zeros((18,1))
            self.pos_model.data.qpos = last_success
            self.pos_model.forward()
            return True

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
        if self.in_left_grip < 0:
            cur_left_dist = -1
            for i in range(self.num_cloths):
                if xpos[self.cloth_inds[i]][2] > 0.67 and np.all((xpos[self.cloth_inds[i]] - xpos[self.left_grip_r_ind])**2 < [0.01, 0.01, 0.01]) and l_grip < const.GRIPPER_CLOSE_VALUE:
                    new_dist = np.sum((xpos[self.cloth_inds[i]] - xpos[self.left_grip_r_ind])**2)
                    if cur_left_dist < 0 or new_dist < cur_left_dist:
                        self.in_left_grip = i
                        cur_left_dist = new_dist

                if grip_cloth == i or np.all((xpos[self.cloth_inds[i]] - xpos[self.left_grip_r_ind])**2 < [0.0081, 0.0081, 0.005]) and l_grip < const.GRIPPER_CLOSE_VALUE:
                    new_dist = np.sum((xpos[self.cloth_inds[i]] - xpos[self.left_grip_r_ind])**2)
                    if cur_left_dist < 0 or new_dist < cur_left_dist:
                        self.in_left_grip = i
                        cur_left_dist = new_dist

        if self.in_right_grip < 0:
            cur_right_dist = -1
            for i in range(self.num_cloths):
                if xpos[self.cloth_inds[i]][2] > 0.67 and np.all((xpos[self.cloth_inds[i]] - xpos[self.right_grip_l_ind])**2 < [0.01, 0.01, 0.01]) and r_grip < const.GRIPPER_CLOSE_VALUE:
                    new_dist = np.sum((xpos[self.cloth_inds[i]] - xpos[self.right_grip_l_ind])**2)
                    if cur_right_dist < 0 or new_dist < cur_right_dist:
                        self.in_right_grip = i
                        cur_right_dist = new_dist

                if grip_cloth == i or np.all((xpos[self.cloth_inds[i]] - xpos[self.right_grip_l_ind])**2 < [0.0081, 0.0081, 0.005]) and r_grip < const.GRIPPER_CLOSE_VALUE:
                    new_dist = np.sum((xpos[self.cloth_inds[i]] - xpos[self.right_grip_l_ind])**2)
                    if cur_right_dist < 0 or new_dist < cur_right_dist:
                        self.in_right_grip = i
                        cur_right_dist = new_dist

        if l_grip > const.GRIPPER_CLOSE_VALUE:
            self.in_left_grip = -1

        if r_grip > const.GRIPPER_CLOSE_VALUE:
            self.in_right_grip = -1

        if self.in_left_grip > -1:
            run_forward = True
            body_pos[self.cloth_inds[self.in_left_grip]] = (xpos[self.left_grip_l_ind] + xpos[self.left_grip_r_ind]) / 2.0
            body_pos[self.cloth_inds[self.in_left_grip]][2] = max(body_pos[self.cloth_inds[self.in_left_grip]][2], 0.655 + MUJOCO_MODEL_Z_OFFSET)

        if self.in_right_grip > -1:
            run_forward = True
            body_pos[self.cloth_inds[self.in_right_grip]] = (xpos[self.right_grip_l_ind] + xpos[self.right_grip_r_ind]) / 2.0
            body_pos[self.cloth_inds[self.in_right_grip]][2] = max(body_pos[self.cloth_inds[self.in_right_grip]][2], 0.655 + MUJOCO_MODEL_Z_OFFSET)

            # for i in range(self.num_cloths-1, -1, -1):
            #     if l_grip > const.GRIPPER_CLOSE_VALUE and np.all((xpos[self.cloth_inds[i]] - xpos[self.basket_ind])**2 < [0.06, 0.06, 0.04]) and xpos[self.cloth_inds[i]][2] > 0.65 + MUJOCO_MODEL_Z_OFFSET:
            #         pos = xpos[self.cloth_inds[i]].copy()
            #         pos[0] += 0.03
            #         pos[1] -= 0.04
            #         pos[2] = 0.67 + MUJOCO_MODEL_Z_OFFSET
            #         body_pos[self.cloth_inds[i]] = pos
            #         run_forward = True
            #         break

        if run_forward:
            self.pos_model.body_pos = body_pos
            self.pos_model.body_quat = body_quat
            self.pos_model.forward()

        return True


    def get_obs(self):
        rawImData = self.obs_viewer.get_image()
        byteStr = rawImData[0]
        width = rawImData[1]
        height = rawImData[2]
        channels = 3
        imArr = np.fromstring(byteStr, np.uint8)
        imArr = imArr.reshape([height, width, channels])
        imArr = np.flipud(imArr)
        obs_h = self._hyperparams['image_height']
        obs_w = self._hyperparams['image_width']
        imArr = imArr.reshape((width*height*channels))
        return imArr


    def init_cost_trajectories(self, alg_map, center=False, full_solve=True):
        for m in range(0, len(self.plans)):
            old_params_free = {}
            for p in self.params[m]:
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
                        p_attrs[attr] = p._free_attrs[attr][:, 0].copy()
                        p._free_attrs[attr][:, 0] = 0

            self.current_cond = m
            if full_solve:
                success = self.solver._backtrack_solve(self.plans[m], n_resamples=3)
            else:
                success = True
                self.set_plan_from_cost_trajs(alg_map.values()[0], 0, self.plans[m].horizon, m)

            while not success:
                print "Solve failed."
                for p in self.params[m]:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        p._free_attrs = old_params_free[p]
                    else:
                        for attr in p._free_attrs:
                            p._free_attrs[attr][:, 0] = old_params_free[p][attr]
                self.replace_cond(m)

                old_params_free = {}
                for p in self.params[m]:
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
                            p_attrs[attr] = p._free_attrs[attr][:, 0].copy()
                            p._free_attrs[attr][:, 0] = 0

                success = self.solver._backtrack_solve(self.plans[m], n_resamples=3)

            for p in self.params[m]:
                if p.is_symbol():
                    if p not in init_act.params: continue
                    p._free_attrs = old_params_free[p]
                else:
                    for attr in p._free_attrs:
                        p._free_attrs[attr][:, 0] = old_params_free[p][attr]

            self.set_cost_trajectories(0, self.plans[m].horizon-1, m, alg_map.values(), center=center)
            for alg in alg_map.values():
                alg.task_breaks = self.task_breaks

        self.initial_opt = False


    def set_plan_from_cost_trajs(self, alg, init_t, final_t, m):
        tgt_x = alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state']
        if utils.POLICY_STEPS_PER_SECOND < 1:
            for t in range(0, final_t-init_t, (1/utils.POLICY_STEPS_PER_SECOND)):
                utils.set_params_attrs(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
        else:
            for t in range(0, final_t-init_t):
                utils.set_params_attrs(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)


    def set_cost_trajectories(self, init_t, final_t, m, algs, center=False):
        tgt_x = np.zeros((final_t-init_t, self.symbolic_bound))
        tgt_u = np.zeros((final_t-init_t, self.dU))

        if utils.POLICY_STEPS_PER_SECOND < 1:
            for t in range(0, final_t-init_t, int(1/utils.POLICY_STEPS_PER_SECOND)):
                utils.fill_vector(self.params[m], self.state_inds, tgt_x[int(t*utils.POLICY_STEPS_PER_SECOND)], t+init_t)
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.action_inds[('baxter', 'lArmPose')]] = self.plans[m].params['baxter'].lArmPose[:, init_t+t+1]
                
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.action_inds[('baxter', 'rArmPose')]] = self.plans[m].params['baxter'].rArmPose[:, init_t+t+1]

                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.action_inds[('baxter', 'lGripper')]] = 0 if self.plans[m].params['baxter'].lGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.action_inds[('baxter', 'rGripper')]] = 0 if self.plans[m].params['baxter'].rGripper[0, init_t+t] <= const.GRIPPER_CLOSE_VALUE else 1

                # tgt_u[int(t*utils.POLICY_STEPS_PER_SECOND), self.action_inds[('baxter', 'pose')]] = self.plans[m].params['baxter'].pose[:, init_t+t+1]
        else:
            for t in range(0, final_t-init_t):
                utils.fill_vector(self.params[m], self.state_inds, tgt_x[t*utils.POLICY_STEPS_PER_SECOND], t+init_t)
                tgt_x[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND] = tgt_x[t*utils.POLICY_STEPS_PER_SECOND]
                
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.action_inds[('baxter', 'lArmPose')]] = self.plans[m].params['baxter'].lArmPose[:, init_t+t+1]
                
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.action_inds[('baxter', 'rArmPose')]] = self.plans[m].params['baxter'].rArmPose[:, init_t+t+1]

                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.action_inds[('baxter', 'lGripper')]] = 0 if self.plans[m].params['baxter'].lGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
                tgt_u[t*utils.POLICY_STEPS_PER_SECOND:(t+1)*utils.POLICY_STEPS_PER_SECOND, self.action_inds[('baxter', 'rGripper')]] = 0 if self.plans[m].params['baxter'].rGripper[0, init_t+t+1] <= const.GRIPPER_CLOSE_VALUE else 1
            
        self.optimal_act_traj[m] = tgt_u
        self.optimal_state_traj[m] = tgt_x

        for alg in algs:
            alg.cost[m]._costs[0]._hyperparams['data_types'][utils.STATE_ENUM]['target_state'] = tgt_x.copy()
            alg.cost[m]._costs[1]._hyperparams['data_types'][utils.ACTION_ENUM]['target_state'] = tgt_u.copy()

        if center:
            for alg in algs:
                for ts in alg.cur[m]:
                    alg.cur[m][ts].traj_distr.k = self.optimal_act_traj[m][ts:ts+alg.T]


    def sample_optimal_trajectories(self):
        class optimal_pol:
            def __init__(self, act_f):
                self.act = act_f

        def get_policy_map(m):
            policy_map = {}
            for task in self.task_encoding.keys():
                policy_map[task] = {}
                policy_map[task]['policy'] = optimal_pol(lambda X, O, t, noise: self.optimal_act_traj[m][t].copy())

            return policy_map


        for m in range(len(self.plans)):
            self.replace_model(m)
            self.sample(get_policy_map(m), m, save=True, use_base_t=False, noisy=False)


    # def set_alg_conditions(self, alg):

    #     alg.cur = [{} for _ in range(alg.M)]
    #     alg.prev = [{} for _ in range(alg.M)]

    #     for m in range(len(self.plans)):
    #         plan = self.plans[m]
    #         task_breaks = self.task_breaks[m]
    #         cur_t = 0
    #         for next_t, task in task_breaks:
    #             if task == alg.task:
    #                 alg.cur[m][cur_t] = IterationData()
    #                 alg.prev[m][cur_t] = IterationData()
    #                 alg.cur[m][cur_t] = TrajectoryInfo()
    #                 if alg._hyperparams['fit_dynamics']:
    #                     dynamics = alg._hyperparams['dynamics']
    #                     alg.cur[m][cur_t].traj_info.dynamics = dynamics['type'](dynamics)

    #                 init_traj_distr = extract_condition(
    #                     alg._hyperparams['init_traj_distr'], alg._cond_idx[m]
    #                 )

    #                 alg.cur[m][cur_t].traj_distr = init_traj_distr['type'](init_traj_distr)


    def replace_cond(self, cond):
        print "Replacing Condition {0}.\n".format(cond)
        plan, task_breaks, color_map = self.get_plan(self.num_cloths)
        self.plans[cond].env.Destroy()
        self.plans[cond] = plan
        self.params[cond] = filter(lambda p: not p.is_symbol(), plan.params.values())
        self.symbols[cond] = filter(lambda p: p.is_symbol(), plan.params.values())
        self.task_breaks[cond] = task_breaks
        self.color_maps[cond] = color_map
        x = np.zeros((self.symbolic_bound,))
        utils.fill_vector(self.params[cond], self.state_inds, x, 0)                
        self.x0[cond] = x

    def replace_all_conds(self):
        for cond in range(len(self.plans)):
            self.replace_cond(cond)
