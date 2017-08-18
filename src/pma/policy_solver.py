import numpy as np
from gps.agent.agent_utils import generate_noise
from gps.agent.agent import Agent
from gps.algorithm.algorithm_utils import IterationData
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from  pma.robot_ll_solver import RobotLLSolver

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 64

class DummyAgent(object):
    def __init__(self, trajectory_length):
        self.T = trajectory_length
        self.dU = 7
        self.dX = 17
        self.dO = 0
        self.dM = 0
        self.dt = 0.05
        self.x_data_types = [JOINT_ANGLES, END_EFFECTOR_POINTS] #Using end effector points here in place of basket pose
        self.obs_data_types = []
        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 [list(range(7)), list(range(7, 14)), list(range(14, 17))])}

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
        super(BaxterPolicySolver, self).__init__(early_converge, transfer_norm)

    def train_action_policy(self, plan, n_samples=5, iterations=5, active_ts=None, num_conditions=1, callback=None, n_resamples=5, verbose=False):
        '''
        Use the PI2 trajectory optimizer from the GPS code base to generate
        policies for the plan, using the RobotLLSolver to create samples
        '''
        action = plan.actions.filter(lambda a: a.active_timesteps[0] >= active_ts[0])[0]
        action_type = action.name
        agent = DummyAgent(plan, active_ts[1] - active_ts[0] + 1)
        dummy_hyperparams = {
            'conditions': num_conditions,
            'x0': self._get_random_initial_states(plan, action_type, num_conditions, action.active_timesteps),
            'dU': agent.dU,
            'dX': agent.dX,
            'dO': agent.dO,
            'dM': agent.dM,
            'dQ': 17,
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

        if not active_ts:
                active_ts = action.active_timesteps

        local_policies = []

        # TODO: Move the block specific to centering the gripper to a separate function
        for i in range(num_conditions):
            self._reset_plan(plan, dummy_hyperparams['x0'][i])
            alg.cur[i].traj_distr = init_pd(dummy_hyperparams)
            samples = []
            sample_costs = np.zeros((n_samples, plan.horizon))
            for j in range(n_samples):
                self.solve(plan, callback, n_resamples, active_ts=active_ts, verbose, force_init=True)
                samples.append(self._traj_to_sample(plan, agent, active_ts))
                sample_costs[j] = self._get_traj_cost(plan, active_ts)
            alg.cur[i].sample_list = SampleList(samples)
            alg.cur[i].traj_distr = traj_opt.update(i, alg, costs=sample_costs)[0]
            local_policies.append(alg.cur[i].traj_distr)

        for _ in range(1, iterations):
            for i in range(num_conditions):
                self._reset_plan(plan, dummy_hyperparams['x0'][i])
                samples = []
                sample_costs = np.ndarray((n_samples, plan.horizon))
                for j in range(n_samples):
                    joint_values = self._sample_to_traj(self._sample_policy(local_policies[i], agent, plan, active_ts, dummy_hyperparams))
                    plan.params['baxter'].lArmPose[:,active_ts[0]:active_ts[1]+1] = joint_values[:7]
                    plan.params['baxter'].rArmPose[:,active_ts[0]:active_ts[1]+1] = joint_values[7:14]                
                    self.solve(plan, callback, n_resamples, active_ts=active_ts, verbose, force_init=False)
                    samples.append(self._traj_to_sample(plan, agent, active_ts))
                    sample_costs[j] = self._get_traj_cost(plan, active_ts)
                alg.cur[i].sample_list = SampleList(samples)
                alg.cur[i].traj_distr = traj_opt.update(i, alg, costs=sample_costs)[0]
                local_policies[i] = alg.cur[i].traj_distr
        return policy

    def _get_random_initial_states(self, plan, action_type, num_conditions, active_ts):
        # In this case moveto is for centering the gripper over the basket handle
        if action_type == 'moveto':
            x0 = []
            plan.target_arm_poses = []
            for i in range(num_conditions):
                x0_next, target_left, target_right = self._generate_random_basket_center_config(plan, active_ts)
                x0.append(x0_next)
                plan.target_arm_poses.append[(target_left, target_right)]
            print 'Generated random initial states for policy learning.\n'
            return x0
        elif action_type == 'basket_grasp':
            x0 = []
            for i in range(num_conditions):
                x0.append(self._generate_random_basket_grasp_config(plan, active_ts))
            print 'Generated random initial states for policy learning.\n'
            return x0
        else:
            raise NotImplementedError

    def _reset_plan(self, plan, initial_state):
        plan.params['robot_end_pose'].lArmPose = plan.target_arm_poses[i][0]
        plan.params['robot_end_pose'].rArmPose = plan.target_arm_poses[i][1]
        plan.params.basket.pose[:2, :] = initial_state[-3:-1]
        plan.params.basket.rotation[0, :] = initial_state[-1]
        plan.params['robot_init_pose'].lArmPose[:,0] = initial_state[:7]
        plan.params['robot_init_pose'].rArmPose[:,0] = initial_state[7:14]
        plan.params['baxter'].lArmPose[:,0] = initial_state[:7]
        plan.params['baxter'].rArmPose[:,0] = initial_state[7:14]

    def _traj_to_sample(self, plan, agent, active_ts):
        sample = Sample(agent)
        for t in range(active_ts[0], active_ts[1]+1):
            sample.set(JOINT_ANGLES, np.r_[plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
            if t < plan.horizon - 1:
                # sample.set(JOINT_VELOCITIES, (plan.params['baxter'].rArmPose[:,t+1]-plan.params['baxter'].rArmPose[:,t]) / plan.params['baxter'].time[0,t], t)
                sample.set(ACTION, np.r_[plan.params['baxter'].lArmPose[:,t+1]-plan.params['baxter'].lArmPose[:,t], plan.params['baxter'].rArmPose[:,t+1]-plan.params['baxter'].rArmPose[:,t]], t-active_ts[0])
            else:
                # sample.set(JOINT_VELOCITIES, np.zeros((7,)), t)
                sample.set(ACTION, np.zeros((7,)), t)
        return sample

    def _sample_to_traj(self, sample):
        traj = sample.get(JOINT_ANGLES)
        return traj.T

    def _get_traj_cost(self, plan, active_ts):
        '''
        Get a vector of the costs for each timestep in the plan
        '''
        act_num = 0
        cur_action = plan.actions[act_num] # filter(lambda a: a.active_timesteps[0] <= 0, plan.actions)[0]
        costs = np.ndarray((active_ts[1]-active_ts[0]+1))
        for ts in range(active_ts[0], active_ts[1]+1):
            timestep_cost = 0
            if ts >= cur_action.active_timesteps[1]:
                act_num += 1
                cur_action = act_num < len(plan.actions) and plan.actions[act_num] # filter(lambda a: a.active_timesteps[0] == ts, plan.actions)
                if not cur_action:
                    continue
            failed = cur_action.get_failed_preds((ts,ts))
            for p in plan.preds:
                if p['pred'] not in failed: continue
                negated = p['negated']
                pred = p['pred']
                param_vector = pred.get_param_vector(ts)
                timestep_cost += np.abs(pred.get_expr(negated=negated).expr.eval(param_vector))
            costs[ts] = timestep_cost
        return costs

    def _sample_policy(self, policy, agent, plan, active_ts, hyperparams):
        new_sample = Sample(agent)
        new_sample.set(JOINT_ANGLES, np.r_[plan.params['baxter'].lArmPose[:, active_ts[0]], plan.params['baxter'].rArmPose[:, active_ts[0]]], 0)
        # new_sample.set(JOINT_VELOCITIES, plan.params['baxter'].rArmPose[:, 1] - plan.params['baxter'].rArmPose[:, 0], 0)
        noise = generate_noise(agent.T, agent.dU, hyperparams)
        U = np.zeros([agent.T, agent.dU])
        for t in range(agent.T - 1):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            cur_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = cur_U
            new_sample.set(JOINT_ANGLES, X_t + cur_U, t+1)
            # new_sample.set(JOINT_VELOCITIES, cur_U / agent.dt, t+1)
        return new_sample

    def _generate_random_basket_grasp_config(self, plan, active_ts=None):
        if not active_ts:
            active_ts = (0, plan.horizon-1)

        basket_x_pose = np.random.choice(np.array(range(-20,20))*.01+0.75)
        basket_y_pose = np.random.choice(np.array(range(-30,30))*.01+0.02)
        basket_z_pose = 0.81
        basket_rot = np.random.choice([np.pi/4, 7*np.pi/24, np.pi/3, 3*np.pi/8, 5*np.pi/12, 11*np.pi/24, np.pi/2, 13*np.pi/24, 7*np.pi/12, 5*np.pi/8, 2*np.pi/3, 17*np.pi/24, 3*np.pi/4])

        robot = plan.params['baxter']

        gripper_z = plan.params['basket'].pose[2, 0] + .125

        ee_left_x = basket_x_pose + np.cos(basket_rot)
        ee_left_y = basket_y_pose + np.sin(basket_rot)

        lArmPose = robot.openrave_body.get_ik_from_pose([ee_left_x, ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(lArmPose):
            lArmPose = lArmPose[0]
        else:
            return []

        ee_right_x = basket_x_pose - np.cos(basket_rot)
        ee_right_y = basket_y_pose - np.sin(basket_rot)

        rArmPose = robot.openrave_body.get_ik_from_pose([ee_right_x, ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(rArmPose):
            rArmPose = rArmPose[0]
        else:
            return []

        return np.r_[lArmPose, rArmPose, basket_x_pose, basket_y_pose, basket_rot]

    def _generate_random_basket_center_config(self, plan, active_ts=None):
        if not active_ts:
            active_ts = (0, plan.horizon-1)

        basket_x_pose = np.random.choice(np.array(range(-20,20))*.01+0.75)
        basket_y_pose = np.random.choice(np.array(range(-30,30))*.01+0.02)
        basket_z_pose = 0.81
        basket_rot = np.random.choice([7*np.pi/24, np.pi/3, 3*np.pi/8, 5*np.pi/12, 11*np.pi/24, np.pi/2, 13*np.pi/24, 7*np.pi/12, 5*np.pi/8, 2*np.pi/3, 17*np.pi/24])

        belief_x = basket_x_pose + np.random(-10, 10)*.2
        belief_y = basket_y_pose + np.random(-10, 10)*.2
        belief_rot = basket_rot + np.random.choice([-np.pi/6, -np.pi/8, -np.pi/12, -np.pi/24, 0, np.pi/24, np.pi/12, np.pi/6])

        robot = plan.params['baxter']

        gripper_z = plan.params['basket'].pose[2, 0] + .125

        ee_left_x = belief_x + np.cos(belief_rot)
        ee_left_y = belief_y + np.sin(belief_rot)

        lArmPose = robot.openrave_body.get_ik_from_pose([ee_left_x, ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(lArmPose):
            lArmPose = lArmPose[0]
        else:
            return []

        ee_right_x = belief_x - np.cos(belief_rot)
        ee_right_y = belief_y - np.sin(belief_rot)

        rArmPose = robot.openrave_body.get_ik_from_pose([ee_right_x, ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(rArmPose):
            rArmPose = rArmPose[0]
        else:
            return []

        target_ee_left_x = basket_x_pose + np.cos(basket_rot)
        target_ee_left_y = basket_y_pose + np.sin(basket_rot)

        target_lArmPose = robot.openrave_body.get_ik_from_pose([ee_left_x, ee_left_y, gripper_z], [0, np.pi/2, 0], 'left_arm')

        if len(target_lArmPose):
            target_lArmPose = target_lArmPose[0]
        else:
            return []

        target_ee_right_x = basket_x_pose - np.cos(basket_rot)
        target_ee_right_y = basket_y_pose - np.sin(basket_rot)

        target_rArmPose = robot.openrave_body.get_ik_from_pose([ee_right_x, ee_right_y, gripper_z], [0, np.pi/2, 0], 'right_arm')

        if len(target_rArmPose):
            target_rArmPose = target_rArmPose[0]
        else:
            return []

        return np.r_[lArmPose, rArmPose, basket_x_pose, basket_y_pose, basket_rot], target_lArmPose, target_rArmPose


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
