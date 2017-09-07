""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import xml.etree.ElementTree as xml

from openravepy import RaveCreatePhysicsEngine

from mujoco_py import mjcore

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
import policy_hooks.policy_solver_utils as utils

'''
Mujoco specific
'''
cloth_xml = '\n<body name={0} pos="{1} {2} {3}" euler = "0 0 0">\n    <geom name="cloth" type="cylinder" size="{4} {5}"/>\n</body>\n'
table_xml = '\n<body name={0} pos="{1} {2} {3}" euler = "0 0 0>\n     <geom name="table" type="box" size="{4} {5} {6}"/>\n</body>\n'
options_xml = '<option timestep="0.01" iterations="{0}" integrator="RK4" />'

BASE_POS_XML = '../models/baxter/baxter_mujoco_pos.xml'
BASE_MOTOR_XML = '../models/baxter/baxter_mujoco.xml'
ENV_XML = 'current_env.xml'

MUJOCO_JOINT_ORDER = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_e2', 'left_w0', 'left_w1', 'left_w2', 'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint'\
                                              'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint']

# TODO: Split this into two agents for the different simulators (Mujooc & Bullet).
# TODO: Handle gripper signals. I think we should do a binary approach (open or closed) due the its simpler nature.
class LaundryWorldAgent(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        self.plans = self._hyperparams['plans']
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.solver = self._hyperparams['solver']
        self.x0 = self._hyperparams['x0']
        self.sim = hyperparams['sim']
        if self.sim == 'mujoco'
            self.pos_model = self._setup_mujoco_model(self.plans[0], motor=False)
            self.motor_model = self._setup_mujoco_model(self.plans[0])
            self.viewer = None
        elif self.sim == 'bullet':
            self.physics = {}

        Agent.__init__(self, config)

    def _generate_xml(self, plan, motor=True):
        base_xml = xml.parse(BASE_MOTOR_XML) if motor else xml.parse(BASE_POS_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        root.append(xml.fromstring(options_xml.format(active_ts[1]-active_ts[0]+1)))
        for param in plan.params:
            if param.is_symbol(): continue
            if param.geom._type == 'can':
                height = param.geom.height
                radius = param.geom.radius
                x, y, z = param.pose[:, active_ts[0]]
                body = xml.fromstring(cloth_xml.format(param.name, x, y, z, height, radius))
                worldbody.append(body)
            elif param.geom._type == 'box':
                length = param.geom.table_dim[0]
                width = param.geom.table_dim[1]
                thickness = param.geom.thickness
                x, y, z = param.pose[:, active_ts[0]]
                body = xml.fromstring(table_xml.format(param.name, x, y, z, length, width, thickness))
                worldbody.append(body)
        base_xml.write(ENV_XML)

    def _setup_mujoco_model(self, plan, pos=True, view=False):
        self._generate_xml(plan, pos)
        model = mjcore.MjModel(ENV_XML)
        if view:
            self.viewer = mjviewer.MjViewer()
            self.viewer.start()
            self.viewer.set_model(model)
            self.viewer.loop_once()
        return model

    def _run_trajectory(self, condition):
        plan = self.plans[condition]
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1
        start_t = active_ts[0]
        env  = plan.env
        params = set()
        for action in plan.actions:
            params.update(action.params)
        params = list(params)

        if self.sim == 'mujoco':
            pass
        elif self.sim == 'bullet':
            for param in params:
                if not param.is_symbol():
                    param_body = param.openrave_body.set_pose(param.pose[:, start_t], param.rotation[:, start_t])
                    dof_map = {attr[0]:getattr(param, attr[0])[:, start_t] for attr in const.ATTR_MAP[param._type]}
                    param_body = param.openrave_body.set_dof(dof_map)

            # if env not in self.physics:
            #     self.physics[env] = RaveCreatePhysicsEngine(env, 'bullet')
            #     self.physics[env].SetGravity((0, 0, -0.981))
            #     env.SetPhysicsEngine(self.physics[env])
            
            # action_attrs = self.action_inds.keys()

            # positions = np.zeros((self.dU, T))
            # velocities = np.zeros((self.dU, T))
            # acceleraions = np.zeros((self.dU, T))

            # t = active_ts[0]
            # utils.positions[:,0]

            # physics = self.physics[env]
            # baxter.openrave_body.env_body.GetLinks()[0].SetStatic(True)
            # env.StopSimulation()
            # env.StartSimulation(timestep=0.001)


            # baxter = plan.params['baxter']
            # baxter.lArmPose
            # baxter.lGripper
            # baxter.rArmPose
            # baxter.rGripper

            # env.StopSimulation()

    def _run_policy(self, cond, policy, noise):
        plan = self.plans[cond]
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        x0 = self.x0[conds]
        self._set_simulator(x0, plan, active_ts[0])
        for t in range(active_ts[0], active_ts[1]+1)
            obs = self._get_obs()
            u = policy.act(x0, obs, noise[:, t-active_ts[0]], t-active_ts[0])
            mj_u = np.zeros((18,1))
            mj_u[:8] = u[:8].reshape(-1, 1)
            mj_u[8] = -u[7]
            mj_u[9:16] = u[8:].reshape(-1, 1)
            mj_u[17] = -u[15]
            real_t = plan.params['baxter'].time[:,t]
            self.motor_model.data.ctrl = mj_u
            start_t = self.motor_model.data.time
            cur_t = start_t
            while cur_t < start_t + real_t:
                self.motor_model.step()
            utils.set_param_attrs(self.action_inds.values(), self.action_inds, np.delete(self.motor_model.data.qpos, [0, 9, 18]), t)
            # TODO: Fill state values from Mujoco instead of action


    def _set_simulator(self, x, plan, t):
        pass

    def _get_obs(self):
        return []

    def _get_sim_state(self):
        return []


    def _inverse_dynamics(self, plan):
        '''
            Compute the joint forces necessary to achieve the provided trajectory. Currently assumes negligible mass of any held objects.
        '''
        vel, acc = utils.map_trajecotory_to_vel_acc(plan, self.dU, self.action_inds)
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1
        U = np.zeros((self.dU, T))
        if self.simulator == 'mujoco':
            for t in range(active_ts[0], active_ts[1]+1):
                x_t = np.zeros((self.dX))
                utils.fill_vector(self.state_inds.values(), self.state_inds, x_t, t)
                self._set_simulator(x_t, plan, t)
                self.model.data.qpos = self._baxter_to_mujoco(plan, t)
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
                qfrc = self.model.data.qfrc_inverse
                U[:, t-active_ts[0]] = np.delete(qfrc, [0, 9, 18]).flatten()
        elif self.simulator == 'bullet':
            # TODO: Fill this in using the bullet physics simulator & openrave inverse dynamics
            pass

        return U


    def _baxter_to_mujoco(self, plan, t):
        baxter = plan.params['baxter']
        return np.r_[0, baxter.lArmPos[:,t], baxter.lGripper[:,t], -baxter.lGripper[:,t], baxter.rArmPose[:,t], baxter.rGripper[:,t], -baxter.rGripper[:,t]]

    # TODO: Fill this in with proper behavior
    def sample(self, policy, condition, verbose=False, save=True, noisy=False):
        sample = Sample(self)
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        utils.reset_plan(self.plans[condition], self.state_inds, self.x0[condition])
        #TODO: Enforce this sample is close to the global policy
        self.solver.solve(self.plans[condition], n_resamples=5, active_ts=self.action.active_timesteps force_init=True)
        utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[0, :], 0, self.dX)
        active_ts = self.action.active_ts
        for t in range(active_ts[0], active_ts[1]+1):
            utils.fill_trajectory_ts_from_policy(policy, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0]], t, self.dX)
            utils.fill_sample_ts_from_trajectory(sample, self.action, self.state_inds, self.action_inds, noise[t-active_ts[0], :], t, self.dX)
        if save:
            self._samples[condition].append(sample)
        return sample
