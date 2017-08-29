import numpy as np
from gps.agent.agent_utils import generate_noise
from gps.agent.agent import Agent
from gps.algorithm.algorithm_utils import IterationData
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, NOISE
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython


import core.util_classes.baxter_constants as const
import pma.policy_hyperparams as hyperparams
import pma.policy_solver_utils as utils
from  pma.robot_ll_solver import RobotLLSolver

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 64

config = hyperparams.config

class DummyAgent(object):
    def __init__(self, T, dU, dX, dO, dM, sensor_dims):
        self.T = T
        self.dU = dU
        self.dX = dX
        self.dO = dO
        self.dM = dM
        self.dt = 1
        self.x_data_types = [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] #Using end effector points here in place of basket pose
        self.sensor_dims = sensor_dims
        self.obs_data_types = []
        self._x_data_idx = {}
        idx = 0
        for d in self.x_data_types:
            self._x_data_idx[d] = list(range(idx, idx+sensor_dims[d]))
            idx += sensor_dims[d]

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert


class DummyAlgorithm(object):
    def __init__(self, hyperparams):
        self.M = hyperparams['conditions']
        self.cur = [IterationData() for _ in range(self.M)]

class BaxterPolicySolver(RobotLLSolver):

    def __init__(self, early_converge=False, transfer_norm='min-vel'):
        # self.ref_images = np.load('controllerImages.npy')
        # self.ref_labels = np.load('controllerLabels.npy')
        self.agent = None
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)

    # TODO: Add hooks into the GPS policy optimizers
    # TODO: Add hooks for online policy learning
    # TODO: Add more robust description of state
    def train_action_policy(self, action, n_samples=5, iterations=5, active_ts=None, num_conditions=1, callback=None, n_resamples=5, verbose=False):
        '''
        Integrates the GPS code base with the TAMPy codebase to create a robust
        system for combining motion planning with policy learning
        '''
        config = baxter_config
        dX, state_inds, dU, action_inds = get_action_description(action)
        active_ts = action.active_timesteps
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            utils.STATE_ENUM: dX,
            utils.ACtiON_ENUM: dU
        }

        config['agent'] = {
            'type': TAMPAgent,
            'x0': x0,
            'T': T,
            'dT': 1,
            'sensor_dims': SENSOR_DIMS,
            'state_include': [utils.STATE_ENUM],
            'obs_include': [],
            'conditions': num_conditions,
            'action': action,
            'state_inds': state_inds,
            'action_inds': action_inds
        }

        self.agent = DummyAgent(active_ts[1] - active_ts[0] + 1)
        dummy_hyperparams = {
            'conditions': num_conditions,
            'x0': self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps),
            'dU': agent.dU,
            'dX': agent.dX,
            'dO': agent.dO,
            'dM': agent.dM,
            'dQ': 14,
            'T': agent.T,
            'dt': agent.dt,
            'state_include': agent.x_data_types,
            'obs_include': agent.obs_data_types,
            'smooth_noise': True,
            'smooth_noise_var': 1.0,
            'smooth_noise_renormalize': True
        }
        alg = DummyAlgorithm(dummy_hyperparams)
        traj_opt = TrajOptPI2(dummy_hyperparams)

        local_policies = []

        # TODO: Move the block specific to centering the gripper to a separate function
        for i in range(num_conditions):
            self._reset_plan(plan, dummy_hyperparams['x0'][i], active_ts, i)
            alg.cur[i].traj_distr = init_pd(dummy_hyperparams)
            samples = []
            sample_costs = np.zeros((n_samples, active_ts[1]-active_ts[0]+1))
            for j in range(n_samples):
                self.solve(plan, callback, n_resamples, active_ts=active_ts, verbose=verbose, force_init=True)
                samples.append(self._traj_to_sample(plan, agent, active_ts))
                sample_costs[j] = self._get_traj_cost(plan, active_ts)
            alg.cur[i].sample_list = SampleList(samples)
            alg.cur[i].traj_distr = traj_opt.update(i, alg, costs=sample_costs)[0]
            local_policies.append(alg.cur[i].traj_distr)

        for _ in range(1, iterations):
            for i in range(num_conditions):
                self._reset_plan(plan, dummy_hyperparams['x0'][i], active_ts, i)
                samples = []
                sample_costs = np.ndarray((n_samples, active_ts[1]-active_ts[0]+1))
                for j in range(n_samples):
                    joint_values = self._sample_to_traj(self._sample_policy(local_policies[i], agent, plan, active_ts, dummy_hyperparams))
                    plan.params['baxter'].lArmPose[:,active_ts[0]:active_ts[1]+1] = joint_values[:7]
                    plan.params['baxter'].rArmPose[:,active_ts[0]:active_ts[1]+1] = joint_values[7:14]                
                    # self.solve(plan, callback, n_resamples, active_ts=active_ts, verbose=verbose, force_init=True)
                    samples.append(self._traj_to_sample(plan, agent, active_ts))
                    sample_costs[j] = self._get_traj_cost(plan, active_ts)
                alg.cur[i].sample_list = SampleList(samples)
                alg.cur[i].traj_distr = traj_opt.update(i, alg, costs=sample_costs)[0]
                local_policies[i] = alg.cur[i].traj_distr
        delattr(plan, 'target_arm_poses')
        return policy

    def train_pi2_policy(self, plan, policy=None, n_samples=10, iterations=10, num_conditions=1, active_ts=None, callback=None, n_resamples=0, verbose=False):
        '''
        Use the PI2 trajectory optimizer from the GPS code base to generate
        policies for the plan, using the RobotLLSolver to create samples
        '''

        if not active_ts:
                active_ts = (0, plan.horizon-1)
        action = filter(lambda a: a.active_timesteps[0] >= active_ts[0], plan.actions)[0]
        action_type = action.name
        sensor_dims = {
            JOINT_ANGLES: 14,
            JOINT_VELOCITIES: 14,
            END_EFFECTOR_POINTS: 3,
            END_EFFECTOR_POINT_VELOCITIES: 3
        }
        agent = DummyAgent(active_ts[1] - active_ts[0] , 14, 34, 0, 1, sensor_dims)
        self.agent = agent
        self.dummy_hyperparams = {
            'conditions': num_conditions,
            'x0': self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps),
            'dU': agent.dU,
            'dX': agent.dX,
            'dO': agent.dO,
            'dM': agent.dM,
            'dQ': 14,
            'T': agent.T,
            'dt': agent.dt,
            'state_include': agent.x_data_types,
            'obs_include': agent.obs_data_types,
            'smooth_noise': False,
            'smooth_noise_var': 0.0,
            'smooth_noise_renormalize': False,
            # 'pos_gains':  np.array([0.3, 0.5, 0.65, 0.75, 1.5, 1.65, 1.5, 0.3, 0.5, 0.65, 0.75, 1.5, 1.65, 1.5]).reshape((14,1))*1e-6,
            'pos_gains':  np.array([0.25, 0.4, 0.65, 0.75, 1.25, 1.35, 1.275, 0.25, 0.4, 0.65, 0.75, 1.25, 1.35, 1.275]).reshape((14,1))*7.5e-7,
            'init_gains':  np.array([2.25, 1, 1, 1, 0.5, 0.5, 0.5, 2.25, 1, 1, 1, 0.5, 0.5, 0.5])*1e3,
            'init_acc': np.zeros((14,)),
            'init_var': 0.0001,
            'stiffness': 0.5,
            'stiffness_vel': 0.5,
            'final_weight': 50.0,
            'type': DynamicsLRPrior,
            'regularization': 1e-6,
            'prior': {
                'type': DynamicsPriorGMM,
                'max_clusters': 20,
                'min_samples_per_cluster': 40,
                'max_samples': 20,
            }
        }
        dummy_hyperparams = self.dummy_hyperparams
        alg = DummyAlgorithm(dummy_hyperparams)
        traj_opt = TrajOptPI2(dummy_hyperparams)

        # TODO: Move the block specific to centering the gripper to a separate function
        if policy:
            alg.cur[0].traj_distr = policy
        else:
            alg.cur[0].traj_distr = init_pd(dummy_hyperparams)

        print 'ITERATION: 0\n'
        samples = []
        sample_costs = np.zeros((n_samples*num_conditions, active_ts[1]-active_ts[0]))
        for i in range(num_conditions):
            self._reset_plan(plan, active_ts, i)
            for j in range(n_samples):
                # self._solve_opt_prob(plan, priority=-2, callback=callback, active_ts=active_ts, verbose=verbose)
                self.solve(plan, callback, n_resamples, active_ts=active_ts, verbose=verbose, force_init=True)
                if j < n_samples / 2:
                    noise = generate_noise(agent.T, agent.dU, self.dummy_hyperparams)
                    cur_noise = np.zeros((agent.dU))
                    for k in range(1, agent.T):
                        cur_noise += alg.cur[0].traj_distr.chol_pol_covar[k-1].T.dot(noise[k-1,:])
                        plan.params['baxter'].lArmPose[:, active_ts[0]+k] += cur_noise[:7]
                        plan.params['baxter'].rArmPose[:, active_ts[0]+k] += cur_noise[7:]
                else:
                    noise = np.zeros((agent.T, agent.dU))
                samples.append(self._traj_to_sample(plan, noise, agent, active_ts))
                sample_costs[j] = self._get_traj_cost(plan, active_ts) * 100
            # dummy_hyperparams['x0'] = self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps)

        alg.cur[0].sample_list = SampleList(samples)
        alg.cur[0].traj_distr = traj_opt.update(0, alg, use_lqr_actions=False, costs=sample_costs)[0]
        policy = alg.cur[0].traj_distr

        lowest_max_cost = 1e20
        iterations_since_descent = 0

        for iteration in range(1, iterations):
            print 'ITERATION: {}\n'.format(iteration)
            samples = []
            sample_costs = np.ndarray((n_samples*num_conditions, active_ts[1]-active_ts[0]))
            # dummy_hyperparams['x0'] = self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps)
            for i in range(num_conditions):
                self._reset_plan(plan, active_ts, i)
                for j in range(n_samples):
                    # TODO: Figure out why sample sometimes gets nans in the state
                    sample = self._sample_policy(policy, agent, plan, active_ts, dummy_hyperparams)
                    joint_values = self._sample_to_traj(sample)
                    plan.params['baxter'].lArmPose[:,active_ts[0]+1:active_ts[1]+1] = joint_values[:7]
                    plan.params['baxter'].rArmPose[:,active_ts[0]+1:active_ts[1]+1] = joint_values[7:14]
                    # self._solve_opt_prob(plan, priority=3, callback=callback,  active_ts=active_ts, verbose=verbose)
                    samples.append(sample)
                    # self._clip_joint_angles(plan, active_ts)
                    sample_costs[j] = self._get_traj_cost(plan, active_ts) * 100
            max_cost = sample_costs.max()
            print max_cost
            if max_cost == 0:
                break
            if max_cost < lowest_max_cost:
                lowest_max_cost = max_cost
                iterations_since_descent = 0
            else:
                iterations_since_descent += 1
                if iterations_since_descent > 5:
                    return alg.cur[0].traj_distr

            alg.cur[0].sample_list = SampleList(samples)
            alg.cur[0].traj_distr = traj_opt.update(0, alg, use_lqr_actions=False, costs=sample_costs)[0]
            policy = alg.cur[0].traj_distr

        return policy


    def train_pi2_grasp_policy(self, plan, policy=None, n_samples=10, iterations=10, num_conditions=1, active_ts=None, callback=None, n_resamples=5, verbose=False):
        '''
        Use the PI2 trajectory optimizer from the GPS code base to generate
        policies for the plan, using the RobotLLSolver to create samples
        '''

        if not active_ts:
                active_ts = (0, plan.horizon-1)
                act_n = 0
        action = filter(lambda a: a.active_timesteps[0] >= active_ts[0], plan.actions)[0]
        action_type = action.name
        sensor_dims = {
            JOINT_ANGLES: 14,
            JOINT_VELOCITIES: 14,
            END_EFFECTOR_POINTS: 4,
            END_EFFECTOR_POINT_VELOCITIES: 4
        }
        agent = DummyAgent(active_ts[1] - active_ts[0], 14, 36, 0, 0, sensor_dims)
        self.agent = agent
        self.dummy_hyperparams = {
            'conditions': num_conditions,
            'x0': self._get_random_initial_states(plan, 'basket_grasp', num_conditions, active_ts),
            'dU': agent.dU,
            'dX': agent.dX,
            'dO': agent.dO,
            'dM': agent.dM,
            'dQ': 14,
            'T': agent.T,
            'dt': agent.dt,
            'state_include': agent.x_data_types,
            'obs_include': agent.obs_data_types,
            'smooth_noise': False,
            'smooth_noise_var': 0.0001,
            'smooth_noise_renormalize': False,
            # 'pos_gains':  np.array([0.3, 0.5, 0.65, 0.75, 1.5, 1.65, 1.5, 0.3, 0.5, 0.65, 0.75, 1.5, 1.65, 1.5]).reshape((14,1))*1e-6,
            'pos_gains': np.array([0.25, 0.4, 0.65, 0.75, 1.25, 1.35, 1.275, 0.25, 0.4, 0.65, 0.75, 1.25, 1.35, 1.275]).reshape((14,1))*1e-7,
            'init_gains': np.array([.25, .5, .5, .75, 1, 1, 1, .25, .5, .5, .75, 1, 1, 1])*1e-2,
            'init_acc': np.zeros((18,)),
            'init_var': 0.00001,
            'stiffness': 5.0,
            'stiffness_vel': 5.0,
            'final_weight': 50.0,
            'type': DynamicsLRPrior,
            'regularization': 1e-6,
            'prior': {
                'type': DynamicsPriorGMM,
                'max_clusters': 20,
                'min_samples_per_cluster': 40,
                'max_samples': 20,
            }
        }
        dummy_hyperparams = self.dummy_hyperparams
        alg = DummyAlgorithm(dummy_hyperparams)
        traj_opt = TrajOptPI2(dummy_hyperparams)

        # TODO: Move the block specific to centering the gripper to a separate function
        if policy:
            alg.cur[0].traj_distr = policy
        else:
            alg.cur[0].traj_distr = init_pd(dummy_hyperparams)

        print 'ITERATION: 0\n'
        samples = []
        sample_costs = np.zeros((n_samples*num_conditions, active_ts[1]-active_ts[0]))
        for i in range(num_conditions):
            self._reset_plan(plan, active_ts, i, 'basket_grasp')
            for j in range(n_samples):
                # self._solve_opt_prob(plan, priority=-2, active_ts=active_ts, verbose=verbose)
                # import ipdb; ipdb.set_trace()
                self.solve(plan, callback=callback, verbose=verbose, n_resamples=5, force_init=True)
                plan.basket_trajs.append((plan.params['basket'].pose[:, active_ts[0]:active_ts[1]+1], plan.params['basket'].rotation[0, active_ts[0]:active_ts[1]+1]))
                if j < n_samples / 2:
                    noise = generate_noise(agent.T, agent.dU, self.dummy_hyperparams)
                    cur_noise = np.zeros((agent.dU))
                    for k in range(1, agent.T):
                        cur_noise += alg.cur[0].traj_distr.chol_pol_covar[k-1].T.dot(noise[k-1,:])
                        plan.params['baxter'].lArmPose[:, active_ts[0]+k] += cur_noise[:7]
                        plan.params['baxter'].rArmPose[:, active_ts[0]+k] += cur_noise[7:14]
                else:
                    noise = np.zeros((agent.T, agent.dU))
                samples.append(self._traj_to_sample(plan, noise, agent, active_ts, 'basket_grasp'))
                sample_costs[j] = self._get_traj_cost(plan, active_ts) * 1e0
            # dummy_hyperparams['x0'] = self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps)

        alg.cur[0].sample_list = SampleList(samples)
        alg.cur[0].traj_distr = traj_opt.update(0, alg, use_lqr_actions=False, costs=sample_costs)[0]
        policy = alg.cur[0].traj_distr

        lowest_max_cost = 1e20
        iterations_since_descent = 0

        for iteration in range(1, iterations):
            print 'ITERATION: {}\n'.format(iteration)
            samples = []
            sample_costs = np.ndarray((n_samples*num_conditions, active_ts[1]-active_ts[0]))
            # dummy_hyperparams['x0'] = self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps)
            for i in range(num_conditions):
                self._reset_plan(plan, active_ts, i, 'basket_grasp')
                for j in range(n_samples):
                    # TODO: Figure out why sample sometimes gets nans in the state
                    sample = self._sample_policy(policy, agent, plan, active_ts, dummy_hyperparams, action='basket_grasp')
                    traj_values = self._sample_to_traj(sample, 'basket_grasp')
                    plan.params['baxter'].lArmPose[:,active_ts[0]+1:active_ts[1]+1] = traj_values[:7]
                    plan.params['baxter'].rArmPose[:,active_ts[0]+1:active_ts[1]+1] = traj_values[7:14]
                    plan.params['robot_end_pose'].lArmPose[:,0] = traj_values[:7, -1]
                    plan.params['robot_end_pose'].rArmPose[:,0] = traj_values[7:14, -1]
                    # self._solve_opt_prob(plan, priority=3, callback=callback,  active_ts=active_ts, verbose=verbose)
                    samples.append(sample)
                    # self._clip_joint_angles(plan, active_ts)
                    sample_costs[j] = self._get_traj_cost(plan, active_ts) * 1e0
            max_cost = sample_costs.max()
            print max_cost
            if max_cost == 0:
                break
            if max_cost < lowest_max_cost:
                lowest_max_cost = max_cost
                iterations_since_descent = 0
            else:
                iterations_since_descent += 1
                if iterations_since_descent > 1000:
                    return alg.cur[0].traj_distr

            alg.cur[0].sample_list = SampleList(samples)
            alg.cur[0].traj_distr = traj_opt.update(0, alg, use_lqr_actions=False, costs=sample_costs)[0]
            policy = alg.cur[0].traj_distr

        delattr(plan, 'target_arm_poses')
        return policy

    def _get_random_initial_states(self, plan, action_type, num_conditions, active_ts):
        # In this case moveto is for centering the gripper over the basket handle
        if action_type == 'moveto':
            x0 = []
            plan.target_arm_poses = []
            plan.start_arm_poses = []
            plan.basket_poses = []
            for i in range(num_conditions):
                state, iter_count = [], 0
                while not len(state):
                    state = self._generate_random_basket_center_config(plan, active_ts)
                    iter_count += 1
                    if iter_count > 100: raise Exception('Cannot find initial state.')
                x0_next, target_left, target_right = state
                x0.append(x0_next)
                plan.target_arm_poses.append((target_left, target_right))
                plan.start_arm_poses.append((x0_next[:7], x0_next[7:14]))
                plan.basket_poses.append(x0_next[-6:-3])
            print 'Generated random initial states for policy learning.\n'
            return x0[0]
        elif action_type == 'basket_grasp':
            x0 = []
            plan.target_arm_poses = []
            plan.start_arm_poses = []
            plan.basket_poses = []
            plan.basket_trajs = []
            for i in range(num_conditions):
                state, iter_count = [], 0
                while not len(state):
                    state = self._generate_random_basket_grasp_config(plan, active_ts)
                    iter_count += 1
                    if iter_count > 100: raise Exception('Cannot find initial state.')
                x0_next, target_left, target_right = state
                x0.append(x0_next)
                plan.start_arm_poses.append((x0_next[:7], x0_next[7:14]))
                plan.basket_poses.append(x0_next[-8:-4])
            print 'Generated random initial states for policy learning.\n'
            return x0[0]
        else:
            raise NotImplementedError

    def _reset_plan(self, plan, active_ts=None, i=0, action='MOVETO'):
        if not active_ts:
            active_ts = (0, plan.horizon-1)
        if action == 'MOVETO':
            plan.params['robot_end_pose'].lArmPose[:,0] = plan.target_arm_poses[i][0]
            plan.params['robot_end_pose'].rArmPose[:,0] = plan.target_arm_poses[i][1]
            plan.params['basket'].pose[:2, active_ts[0]:active_ts[1]+1] = plan.basket_poses[i][:2].reshape((2,1))
            plan.params['basket'].rotation[0, active_ts[0]:active_ts[1]+1] = plan.basket_poses[i][2]
            plan.params['robot_init_pose'].lArmPose[:,0] = plan.start_arm_poses[i][0]
            plan.params['robot_init_pose'].rArmPose[:,0] = plan.start_arm_poses[i][1]
            plan.params['baxter'].lArmPose[:,active_ts[0]] = plan.start_arm_poses[i][0]
            plan.params['baxter'].rArmPose[:,active_ts[0]] = plan.start_arm_poses[i][1]
        if action == 'basket_grasp':
            # plan.params['robot_end_pose'].lArmPose[:,0] = plan.target_arm_poses[i][0]
            # plan.params['robot_end_pose'].rArmPose[:,0] = plan.target_arm_poses[i][1]
            plan.params['basket'].pose[:, active_ts[0]] = plan.basket_poses[i][:3]
            plan.params['basket'].rotation[0, active_ts[0]] = plan.basket_poses[i][3]
            plan.params['init_target'].value[:, active_ts[0]] = plan.basket_poses[i][:3]
            plan.params['init_target'].rotation[0, active_ts[0]] = plan.basket_poses[i][3]
            plan.params['robot_init_pose'].lArmPose[:,0] = plan.start_arm_poses[i][0]
            plan.params['robot_init_pose'].rArmPose[:,0] = plan.start_arm_poses[i][1]
            plan.params['baxter'].lArmPose[:,active_ts[0]] = plan.start_arm_poses[i][0]
            plan.params['baxter'].rArmPose[:,active_ts[0]] = plan.start_arm_poses[i][1]
            if len(plan.basket_trajs):
                plan.params['basket'].pose[:,active_ts[0]:active_ts[1]+1] = plan.basket_trajs[i][0]
                plan.params['basket'].rotation[0,active_ts[0]:active_ts[1]+1] = plan.basket_trajs[i][1]

    def _clip_joint_angles(self, plan, active_ts):
        DOF_limits = plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])
        for t in range(active_ts[0], active_ts[1]):
            lArmPose = plan.params['baxter'].lArmPose[:,t]
            rArmPose = plan.params['baxter'].rArmPose[:,t]
            for i in range(7):
                if lArmPose[i] < left_DOF_limits[0][i]:
                    lArmPose[i] = left_DOF_limits[0][i]
                if lArmPose[i] > left_DOF_limits[1][i]:
                    lArmPose[i] = left_DOF_limits[1][i]
                if rArmPose[i] < right_DOF_limits[0][i]:
                    rArmPose[i] = right_DOF_limits[0][i]
                if rArmPose[i] > right_DOF_limits[1][i]:
                    rArmPose[i] = right_DOF_limits[1][i]

    def _traj_to_sample(self, plan, noise, agent, active_ts, action='MOVETO'):
        sample = Sample(agent)
        if action == 'MOVETO':
            for t in range(active_ts[0], active_ts[1]):
                sample.set(END_EFFECTOR_POINTS, np.r_[plan.params['basket'].pose[:2, 0], plan.params['basket'].rotation[0,0]], t-active_ts[0])
                sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros((3,)), t-active_ts[0])
                sample.set(JOINT_ANGLES, np.r_[plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
                sample.set(JOINT_VELOCITIES, np.r_[(plan.params['baxter'].lArmPose[:,t+1]-plan.params['baxter'].lArmPose[:,t]), (plan.params['baxter'].rArmPose[:,t+1]-plan.params['baxter'].rArmPose[:,t])] / agent.dt, t-active_ts[0])
                sample.set(ACTION, np.r_[plan.params['baxter'].lArmPose[:,t+1]-plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t+1]-plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
                sample.set(NOISE, noise[t-active_ts[0], :], t-active_ts[0])
        elif action == 'basket_grasp':
            for t in range(active_ts[0], active_ts[1]):
                sample.set(END_EFFECTOR_POINTS, np.r_[plan.params['basket'].pose[:, t], plan.params['basket'].rotation[0,t]], t-active_ts[0])
                sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros((4,)), t-active_ts[0])
                sample.set(JOINT_ANGLES, np.r_[plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
                sample.set(JOINT_VELOCITIES, np.zeros((14,)), t-active_ts[0])
                sample.set(ACTION, np.r_[plan.params['baxter'].lArmPose[:,t+1]-plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t+1]-plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
                sample.set(NOISE, noise[t-active_ts[0], :], t-active_ts[0])
        return sample

    def _sample_to_traj(self, sample, action='MOVETO'):
        if action == 'MOVETO':
            joint_values = np.zeros((14, sample.T))
            X_t = sample.get_X(t=0)[:14]
            for t in range(sample.T):
                U_t = sample.get(ACTION, t=t)
                joint_values[:, t] = X_t + U_t
                X_t += U_t
            return joint_values
        elif action == 'basket_grasp':
            # Ordered as lArmPose, rArmPose, basket pose, basket rot
            traj_values = np.zeros((14, sample.T))
            X_t = sample.get_X(t=0)[:14]
            for t in range(sample.T):
                U_t = sample.get(ACTION, t=t)
                traj_values[:, t] = X_t + U_t
                X_t += U_t
            return traj_values

    def _get_traj_cost(self, plan, active_ts):
        '''
        Get a vector of the costs for each timestep in the plan
        '''
        act_num = 0
        cur_action = plan.actions[act_num] # filter(lambda a: a.active_timesteps[0] <= 0, plan.actions)[0]
        costs = np.zeros((active_ts[1]-active_ts[0]))
        # TODO: Find bug in action.get_failed_preds
        failed_preds = cur_action.get_failed_preds(tol=1e-3)
        for ts in range(active_ts[0], active_ts[1]+1):
            timestep_cost = 0
            if ts > cur_action.active_timesteps[1]:
                act_num += 1
                cur_action = act_num < len(plan.actions) and plan.actions[act_num] # filter(lambda a: a.active_timesteps[0] == ts, plan.actions)
                if not cur_action:
                    continue
                failed_preds = cur_action.get_failed_preds(tol=1e-3)
            failed = filter(lambda pred: pred[2] == ts, failed_preds)
            failed = map(lambda pred: pred[1], failed)
            for p in cur_action.preds:
                if p['pred'] not in failed: continue
                negated = p['negated']
                pred = p['pred']
                param_vector = pred.get_param_vector(ts)
                timestep_cost += np.sum(np.abs(pred.get_expr(negated=negated).expr.eval(param_vector)))
            if ts == 0:
                costs[ts] += timestep_cost
            else:
                costs[ts-1] += timestep_cost
        if costs.max() < 1e-3:
            return np.zeros(costs.shape)
        print plan.get_failed_preds(tol=1e-3)
        # costs += 1e1 * np.sum(np.abs(plan.params['baxter'].lArmPose[:,-1]-plan.params['robot_end_pose'].lArmPose[:,0]) + np.abs(plan.params['baxter'].rArmPose[:,-1]-plan.params['robot_end_pose'].rArmPose[:,-1]))# costs[-1]
        return costs

    def _sample_policy(self, policy, agent, plan, active_ts, hyperparams, use_noise=True, action='MOVETO'):
        new_sample = Sample(agent)
        if use_noise:
            noise = generate_noise(agent.T, agent.dU, hyperparams)
        else:
            noise = np.zeros((agent.T, agent.dU))
        U = np.zeros([agent.T, agent.dU])
        if action == 'MOVETO':
            X_t = np.r_[plan.params['baxter'].lArmPose[:,active_ts[0]], plan.params['baxter'].rArmPose[:,active_ts[0]], np.zeros((14,)), plan.params['basket'].pose[:2, 0], plan.params['basket'].rotation[0,0], np.zeros((3,))]
            cur_U = np.zeros((14,))
            for t in range(active_ts[1] - active_ts[0]):
                new_sample.set(JOINT_ANGLES, X_t[:14], t-active_ts[0])
                if np.any(map(np.isnan, X_t)):
                    import ipdb; ipdb.set_trace()
                obs_t = [] # new_sample.get_obs(t=t)
                cur_U = policy.act(X_t, obs_t, t, noise[t-active_ts[0], :])
                U[t, :] = cur_U
                # TODO: Make this general to sampling any state
                new_sample.set(JOINT_VELOCITIES, cur_U / agent.dt, t-active_ts[0])
                new_sample.set(END_EFFECTOR_POINTS, np.r_[plan.params['basket'].pose[:2, 0], plan.params['basket'].rotation[0,0]], t-active_ts[0])
                new_sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros((3,)), t-active_ts[0])
                new_sample.set(ACTION, cur_U, t-active_ts[0])
                new_sample.set(NOISE, noise[t-active_ts[0], :], t-active_ts[0])
                X_t = new_sample.get_X(t=t-active_ts[0])
                X_t[:14] += cur_U
        elif action == 'basket_grasp':
            X_t = np.r_[plan.params['baxter'].lArmPose[:,active_ts[0]], plan.params['baxter'].rArmPose[:,active_ts[0]], np.zeros((14,)), plan.params['basket'].pose[:,0], plan.params['basket'].rotation[0,0], np.zeros((4,))]
            cur_U = np.zeros((14,))
            for t in range(active_ts[1] - active_ts[0]):
                new_sample.set(JOINT_ANGLES, X_t[:14], t-active_ts[0])
                new_sample.set(END_EFFECTOR_POINTS, X_t[-8:-4], t-active_ts[0])
                new_sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros((4,)), t-active_ts[0])
                if np.any(map(np.isnan, X_t)):
                    import ipdb; ipdb.set_trace()
                obs_t = [] # new_sample.get_obs(t=t)
                cur_U = policy.act(X_t, obs_t, t, noise[t-active_ts[0], :])
                U[t, :] = cur_U
                # TODO: Make this general to sampling any state
                new_sample.set(JOINT_VELOCITIES, np.zeros((14,)), t-active_ts[0])
                new_sample.set(ACTION, cur_U, t-active_ts[0])
                new_sample.set(NOISE, noise[t-active_ts[0], :], t-active_ts[0])
                X_t = new_sample.get_X(t=t-active_ts[0])
                X_t[:14] += cur_U[:14]
        return new_sample

    def _generate_random_basket_grasp_config(self, plan, active_ts=None):
        if not active_ts:
            active_ts = (0, plan.horizon-1)

        # TODO: Make these values general
        basket_x_pose = np.random.choice(range(-10,0))*.01+0.75
        basket_y_pose = np.random.choice(range(-30,30))*.01+0.02
        basket_z_pose = 0.81
        basket_rot = np.random.choice([np.pi/3, 3*np.pi/8, 5*np.pi/12, 11*np.pi/24, np.pi/2, 13*np.pi/24, 7*np.pi/12, 5*np.pi/8, 2*np.pi/3])

        belief_x = basket_x_pose + np.random.choice(range(-10, 10))*.02
        belief_y = basket_y_pose + np.random.choice(range(-10, 10))*.02
        belief_rot = basket_rot + np.random.choice([-np.pi/8, -np.pi/12, -np.pi/24, 0, np.pi/24, np.pi/12, np.pi/8])

        robot = plan.params['baxter']

        gripper_z = plan.params['basket'].pose[2, 0] + .125

        ee_left_x = belief_x + const.BASKET_OFFSET*np.cos(belief_rot)
        ee_left_y = belief_y + const.BASKET_OFFSET*np.sin(belief_rot)

        # Start with grippers placed as if basket at center of table
        # lArmPose = [[-0.6, -1.20315112, -0.116344, 1.3919732, 0.0424701, 1.384408, 0.06889261]]
        lArmPose = robot.openrave_body.get_ik_from_pose([ee_left_x, ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(lArmPose):
            lArmPose = lArmPose[0]
        else:
            return []

        ee_right_x = belief_x - const.BASKET_OFFSET*np.cos(belief_rot)
        ee_right_y = belief_y - const.BASKET_OFFSET*np.sin(belief_rot)


        # Start with grippers placed as if basket at center of table
        # rArmPose = [[0.6, -1.20315112, 0.116344, 1.3919732, -0.0424701, 1.384408, -0.06889261]]
        rArmPose = robot.openrave_body.get_ik_from_pose([ee_right_x, ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(rArmPose):
            rArmPose = rArmPose[0]
        else:
            return []

        target_ee_left_x = basket_x_pose + const.BASKET_OFFSET*np.cos(basket_rot)
        target_ee_left_y = basket_y_pose + const.BASKET_OFFSET*np.sin(basket_rot)

        target_lArmPose = robot.openrave_body.get_ik_from_pose([target_ee_left_x, target_ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(target_lArmPose):
            target_lArmPose = target_lArmPose[0]
        else:
            return []

        target_ee_right_x = basket_x_pose - const.BASKET_OFFSET*np.cos(basket_rot)
        target_ee_right_y = basket_y_pose - const.BASKET_OFFSET*np.sin(basket_rot)

        target_rArmPose = robot.openrave_body.get_ik_from_pose([target_ee_right_x, target_ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(target_rArmPose):
            target_rArmPose = target_rArmPose[0]
        else:
            return []

        # plan.params['robot_end_pose'].lArmPose[:,0] = target_lArmPose
        # plan.params['robot_end_pose'].rArmPose[:,0] = target_rArmPose
        plan.params['basket'].pose[:, active_ts[0]] = [basket_x_pose, basket_y_pose, 0.81]
        plan.params['basket'].rotation[0, active_ts[0]] = basket_rot
        plan.params['init_target'].value[:, 0] = [basket_x_pose, basket_y_pose, 0.81]
        plan.params['init_target'].rotation[0, 0] = basket_rot
        plan.params['robot_init_pose'].lArmPose[:,0] = lArmPose.copy()
        plan.params['robot_init_pose'].rArmPose[:,0] = rArmPose.copy()
        plan.params['baxter'].lArmPose[:,active_ts[0]] = lArmPose.copy()
        plan.params['baxter'].rArmPose[:,active_ts[0]] = rArmPose.copy()

        # self._solve_opt_prob(plan, priority=-2, active_ts=active_ts)
        # self._solve_opt_prob(plan, priority=0, active_ts=active_ts)
        # self._solve_opt_prob(plan, priority=1, active_ts=active_ts)
        # bg_ee_left = plan.params['bg_ee_left']
        # bg_ee_right = plan.params['bg_ee_right']
        # baxter_body = plan.params['baxter'].openrave_body

        # plan.params['table'].openrave_body.set_pose([10,10,10])
        # plan.params['basket'].openrave_body.set_pose([10,10,10])

        # if not len(baxter_body.get_ik_from_pose(bg_ee_left.value.flatten(), bg_ee_left.rotation.flatten(), "left_arm")) \
        #    or not len(baxter_body.get_ik_from_pose(bg_ee_right.value.flatten(), bg_ee_right.rotation.flatten(), "right_arm")):
        #    print bg_ee_left.value, bg_ee_right.value
        #    return []

        if len(plan.get_failed_preds(active_ts=(active_ts[0], active_ts[0]), tol=1e-3)):
            return []

        return np.r_[lArmPose, rArmPose, np.zeros((14,)), basket_x_pose, basket_y_pose, 0.81, basket_rot, np.zeros((4,))], target_lArmPose, target_rArmPose

    def _generate_random_basket_center_config(self, plan, active_ts=None):
        if not active_ts:
            active_ts = (0, plan.horizon-1)

        # TODO: Make these values general
        basket_x_pose = np.random.choice(range(-10,5))*.01+0.75
        basket_y_pose = np.random.choice(range(-20,20))*.01+0.02
        basket_z_pose = 0.81
        basket_rot = np.random.choice([5*np.pi/12, 11*np.pi/24, np.pi/2, 13*np.pi/24, 7*np.pi/12])

        belief_x = basket_x_pose + np.random.choice(range(-10, 10))*.02
        belief_y = basket_y_pose + np.random.choice(range(-10, 10))*.02
        belief_rot = basket_rot + np.random.choice([-np.pi/8, -np.pi/12, -np.pi/24, 0, np.pi/24, np.pi/12, np.pi/8])

        robot = plan.params['baxter']

        gripper_z = plan.params['basket'].pose[2, 0] + .125

        ee_left_x = belief_x + const.BASKET_OFFSET*np.cos(belief_rot)
        ee_left_y = belief_y + const.BASKET_OFFSET*np.sin(belief_rot)

        # Start with grippers placed as if basket at center of table
        # lArmPose = [[-0.6, -1.20315112, -0.116344, 1.3919732, 0.0424701, 1.384408, 0.06889261]]
        lArmPose = robot.openrave_body.get_ik_from_pose([ee_left_x, ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(lArmPose):
            lArmPose = lArmPose[0]
        else:
            return []

        ee_right_x = belief_x - const.BASKET_OFFSET*np.cos(belief_rot)
        ee_right_y = belief_y - const.BASKET_OFFSET*np.sin(belief_rot)


        # Start with grippers placed as if basket at center of table
        # rArmPose = [[0.6, -1.20315112, 0.116344, 1.3919732, -0.0424701, 1.384408, -0.06889261]]
        rArmPose = robot.openrave_body.get_ik_from_pose([ee_right_x, ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(rArmPose):
            rArmPose = rArmPose[0]
        else:
            return []

        target_ee_left_x = basket_x_pose + const.BASKET_OFFSET*np.cos(basket_rot)
        target_ee_left_y = basket_y_pose + const.BASKET_OFFSET*np.sin(basket_rot)

        target_lArmPose = robot.openrave_body.get_ik_from_pose([target_ee_left_x, target_ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(target_lArmPose):
            target_lArmPose = target_lArmPose[0]
        else:
            return []

        target_ee_right_x = basket_x_pose - const.BASKET_OFFSET*np.cos(basket_rot)
        target_ee_right_y = basket_y_pose - const.BASKET_OFFSET*np.sin(basket_rot)

        target_rArmPose = robot.openrave_body.get_ik_from_pose([target_ee_right_x, target_ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(target_rArmPose):
            target_rArmPose = target_rArmPose[0]
        else:
            return []

        plan.params['robot_end_pose'].lArmPose[:,0] = target_lArmPose
        plan.params['robot_end_pose'].rArmPose[:,0] = target_rArmPose
        plan.params['robot_end_pose'].lGripper[:,0] = 0.015
        plan.params['robot_end_pose'].rGripper[:,0] = 0.015
        plan.params['robot_end_pose'].value[:,0] = 0   
        plan.params['basket'].pose[:2, active_ts[0]] = [basket_x_pose, basket_y_pose]
        plan.params['basket'].rotation[0, active_ts[0]] = basket_rot
        plan.params['robot_init_pose'].lArmPose[:,0] = lArmPose
        plan.params['robot_init_pose'].rArmPose[:,0] = rArmPose
        plan.params['baxter'].lArmPose[:,active_ts[0]] = lArmPose
        plan.params['baxter'].rArmPose[:,active_ts[0]] = rArmPose

        if len(plan.get_failed_preds(active_ts=(active_ts[0], active_ts[0]), tol=1e-3)):
            return []

        return np.r_[lArmPose, rArmPose, np.zeros((14,)), basket_x_pose, basket_y_pose, basket_rot, np.zeros((3,))], target_lArmPose, target_rArmPose

    # def _get_reference_image(self, ee_pose, basket_pose):
    #     label = np.r_[basket_pose, ee_pose]
    #     closest_ind = 0
    #     closest_dist = np.sum((self.ref_labels[0] - label)**2)
    #     for i in range(1, len(self.ref_labels)):
    #         cur_label = self.ref_labels[i]
    #         if np.sum((cur_label - label)**2) < closest_dist:
    #             closest_ind = i
    #             closest_dist = np.sum((cur_label - label)**2)
    #     return self.ref_images[i]
