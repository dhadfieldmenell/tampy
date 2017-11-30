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

# MUJOCO_MODEL_Z_OFFSET = -0.686
MUJOCO_MODEL_Z_OFFSET = -0.706

N_CONTACT_LIMIT = 14

left_lb = [-1.70167994, -2.147, -3.05417994, -0.05, -3.059, -1.57079633, -3.059]
left_ub = [1.70167994, 1.047, 3.05417994, 2.618, 3.059, 2.094, 3.059]


class LaundryWorldClothLeftAgent(Agent):
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
                # Exclude collisions between the left hand and the cloth to help prevent exceeding the contact limit
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_wrist'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_hand'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_base'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger_tip'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger'})
                xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'basket'})
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
        
        self.l_gripper_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, 'left_gripper_l_finger_tip')
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
        return np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[x_inds['baxter', 'lArmPose']], x[x_inds['baxter', 'lGripper']], -x[x_inds['baxter', 'lGripper']]]


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
                X[x_inds[('baxter', 'lArmPose')]] = left_arm.flatten()
                X[x_inds[('baxter', 'lGripper')]] = model.data.qpos[17, 0]

                X[x_inds[('baxter', 'lArmPose__vel')]] = model.data.qvel[10:17].flatten()
                X[x_inds[('baxter', 'lGripper__vel')]] = model.data.qvel[17]

        return X

    
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
            if self.stochastic_conditions and save_global:
                self.replace_cond(condition)

            if noisy:
                noise = np.random.uniform(-1, 1, (self.T, self.dU))
                noise[:, self.plan.action_inds[('baxter', 'lGripper')]] *= 0
            else:
                noise = np.zeros((self.T, self.dU))

            self._set_simulator_state(x0[0], self.plan)
            last_success_X = x0[0]
            for t in range(self.T):
                X = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
                U = policy.act(X, X, t, noise[t])
                sample.set(STATE_ENUM, X, t)
                sample.set(OBS_ENUM, X, t)
                sample.set(ACTION_ENUM, U, t)
                sample.set(NOISE_ENUM, noise[t], t)
                success = self.run_policy_step(U)
                if success:
                    last_success_X = X
                else:
                    break
                    self._set_simulator_state(last_success_X, self.plan)
                if not t % 100 and self.viewer:
                    self.viewer.loop_once()
            if save_global and success:
                self.cond_global_pol_sample[condition] = sample
            print 'Finished on-policy sample.\n'.format(condition)
        else:
            success = self.sample_joint_trajectory(condition, sample)

        if save:
            self._samples[condition].append(sample)
        return sample


    def sample_joint_trajectory(self, condition, sample, bootstrap=False):
        # Initialize the plan for the given condition
        x0 = self.init_plan_states[condition]

        first_act = self.plan.actions[x0[1][0]]
        last_act = self.plan.actions[x0[1][1]]
        init_t = first_act.active_timesteps[0]
        final_t = last_act.active_timesteps[1]

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

        self._set_simulator_state(x0[0], self.plan)
        print "Reset Mujoco Sim to Condition {0}".format(condition)
        for ts in range(init_t, final_t):
            U = np.zeros(self.plan.dU)
            utils.fill_vector(self.params, self.plan.action_inds, U, ts+1)
            success = self.run_traj_step(U, self.plan.action_inds, sample, ts)
            if not success:
                print 'Collision Limit Exceeded'

        return success


    def run_traj_step(self, u, u_inds, sample, plan_t=0, timelimit=1.0):
        next_right_pose = np.zeros((7,))
        next_left_pose = u[u_inds[('baxter', 'lArmPose')]]
        r_grip = 0
        l_grip = u[u_inds[('baxter', 'lGripper')]]

        if l_grip > const.GRIPPER_CLOSE_VALUE:
            l_grip = const.GRIPPER_OPEN_VALUE
        else:
            l_grip = 0

        steps = int(timelimit * utils.MUJOCO_STEPS_PER_SECOND)

        self.pos_model.data.ctrl = np.r_[next_right_pose, r_grip, -r_grip, next_left_pose, l_grip, -l_grip].reshape((18, 1))
        self.avg_vels = np.zeros((18,))
        success = True
        X = self._get_simulator_state(self.plan.state_inds, self.plan.symbolic_bound)
        last_success_X = X
        qpos_0 = self.pos_model.data.qpos.flatten()

        for t in range(0, steps):
            run_forward = False
            for i in range(self.num_cloths):
                if np.sum((xpos[self.cloth_inds[i]] - xpos[self.l_gripper_ind])**2) < .0016 and self.pos_model.data.qpos[17] < const.GRIPPER_CLOSE_VALUE:
                    xpos[self.cloth_inds[i]] = xpos[self.l_gripper_ind]
                    run_forward = True
                    break

            if run_forward: self.pos_model.forward()

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
            U = np.zeros((self.plan.dU,))
            U[u_inds[('baxter', 'lArmPose')]] = next_left_pose
            U[u_inds[('baxter', 'lGripper')]] = l_grip
            sample.set(ACTION_ENUM, U, plan_t*steps+t)
            sample.set(NOISE_ENUM, np.zeros((self.plan.dU,)), plan_t*steps+t)
        if self.viewer:
            self.viewer.loop_once()

        return success


    def run_policy_step(self, u):
        u_inds = self.plan.action_inds
        r_joints = [0, 0, 0, 0, 0, 0, 0]
        l_joints = u[u_inds[('baxter', 'lArmPose')]]
        l_joints = np.maximum(l_joints, left_lb)
        l_joints = np.minimum(l_joints, left_ub)
        r_grip = 0
        l_grip = u[u_inds[('baxter', 'lGripper')]]

        if r_grip <= const.GRIPPER_CLOSE_VALUE:
            r_grip = 0

        if l_grip <= const.GRIPPER_CLOSE_VALUE:
            l_grip = 0

        success = True

        if self.pos_model.data.ncon < N_CONTACT_LIMIT:
            self.pos_model.data.ctrl = np.r_[r_joints, r_grip, -r_grip, l_joints, l_grip, -l_grip].reshape((18, 1))
        else:
            print 'Collision Limit Exceeded in Position Model.'
            self.pos_model.data.ctrl = np.zeros((18,1))
            success = False
        self.pos_model.step()

        body_pos = self.pos_model.body_pos.copy()
        xpos = self.pos_model.data.xpos.copy()
        run_forward = False
        for i in range(self.num_cloths):
            if np.all((xpos[self.cloth_inds[i]] - xpos[self.l_gripper_ind])**2 < [0.0025, 0.0025, 0.0009]) and self.pos_model.data.ctrl[16] < const.GRIPPER_CLOSE_VALUE:
                body_pos[self.cloth_inds[i]] = xpos[self.l_gripper_ind]
                run_forward = True
                break
        if run_forward:
            self.pos_model.body_pos = body_pos
            self.pos_model.forward()

        return success


    def _clip_joint_angles(self, X):
        DOF_limits = self.plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
        X[self.plan.state_inds[('baxter', 'lArmPose')]] = np.maximum(X[self.plan.state_inds[('baxter', 'lArmPose')]], left_DOF_limits[0])
        X[self.plan.state_inds[('baxter', 'lArmPose')]] = np.minimum(X[self.plan.state_inds[('baxter', 'lArmPose')]], left_DOF_limits[1])


    def optimize_trajectories(self, alg, reuse=False):
        ps = PlanSerializer()
        pd = PlanDeserializer()

        if reuse:
            pass
        else:
            for m in range(0, alg.M):
                x0 = self.init_plan_states[m]
                init_act = self.plan.actions[x0[1][0]]
                final_act = self.plan.actions[x0[1][1]]
                init_t = init_act.active_timesteps[0]
                final_t = final_act.active_timesteps[1]

                utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
                utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
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
                self.plan.params['baxter'].rArmPose[:,:] = 0
                self.plan.params['baxter'].rGripper[:,:] = 0
                success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

                while self.initial_opt and not success:
                    print "Solve failed."
                    self.replace_cond(m, self.num_cloths)
                    x0 = self.init_plan_states[m]
                    utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
                    utils.set_params_attrs(self.symbols, self.plan.state_inds, x0[0], init_t)
                    for param in x0[2]:
                        self.plan.params[param].pose[:,:] = x0[0][self.plan.state_inds[(param, 'pose')]].reshape(3,1)
                    success = self.solver._backtrack_solve(self.plan, anum=x0[1][0], amax=x0[1][1])

                for p in self.params:
                    if p.is_symbol():
                        if p not in init_act.params: continue
                        p._free_attrs = old_params_free[p]
                    else:
                        for attr in p._free_attrs:
                            p._free_attrs[attr][:, init_t] = old_params_free[p][attr]

                if self.cond_global_pol_sample[m]:
                    self.solver.optimize_against_global(self.plan, x0[1][0], x0[1][1], m)

                # if self.initial_opt:
                #     'Saving plan...\n'
                #     ps.write_plan_to_hdf5('plan_{0}_cloths_condition_{1}.hdf5'.format(self.num_cloths, m), self.plan)
                #     pickle.dump(x0, 'plan_{0}_cloths_condition_{1}_init.npy')

                # utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], init_t)
                traj_distr = alg.cur[m].traj_distr
                k = np.zeros((traj_distr.T, traj_distr.dU))
                for t in range(init_t, final_t):
                    u = np.zeros((self.plan.dU))
                    utils.fill_vector(self.params, self.plan.action_inds, u, t+1)
                    k[(t-init_t)*utils.MUJOCO_STEPS_PER_SECOND:(t-init_t+1)*utils.MUJOCO_STEPS_PER_SECOND] = u
                traj_distr.k = k

        self.initial_opt = False


    def replace_cond(self, cond):
        print "Replacing Condition {0}.\n".format(cond)
        x0s = get_randomized_initial_state_left(self.plan)
        self.init_plan_states[cond] = x0s
        self.x0[cond] = x0s[0][:self.plan.symbolic_bound]
        self.cond_global_pol_sample[cond] = None


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
