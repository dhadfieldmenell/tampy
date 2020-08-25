import copy
import sys
import time
import traceback
from baxter_gym.envs import *
from baxter_gym.util_classes.mjc_xml_utils import *

from gps.agent.agent_utils import generate_noise
from gps.sample.sample_list import SampleList

from policy_hooks.sample import Sample
from policy_hooks.tamp_agent import TAMPAgent
from policy_hooks.utils.policy_solver_utils import *


class optimal_pol:
    def __init__(self, dU, action_inds, state_inds, opt_traj):
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj

    def act(self, X, O, t, noise):
        u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            u[self.action_inds[param, attr]] = self.opt_traj[t, self.action_inds[param, attr]].copy()
        return u


class BaxterMJCSortingAgent(TAMPAgent):
    def __init__(self, hyperparams):
        plans = hyperparams['plans']
        params = list(plans.values())[0].params
        items = []
        for p in list(params.values()):
            if p._type == 'Cloth':
                items.append(get_param_xml(p))
        self.im_h, self.im_w = hyperparams['image_height'], hyperparams['image_width']
        super(BaxterMJCSortingAgent, self).__init__(hyperparams)
        self.env = BaxterMJCEnv(items=items,
                                im_dims=(self.im_w, self.im_h),
                                obs_include=['end_effector', 'joints'],
                                mode='end_effector_pos',
                                view=False)

        self.n_items = self.prob.NUM_CLOTHS
        x0s = []
        for m in range(len(self.x0)):
            sample = Sample(self)
            mp_state = self.x0[m]
            self.fill_sample(m, sample, mp_state, 0, tuple(np.zeros(1+len(list(self.prim_dims.keys())), dtype='int32')))
            self.x0[m] = sample.get_X(t=0).copy()
            for param_name, attr in self.state_inds:
                if attr == 'pose':
                    self.env.set_item_pos(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False)

        for m in range(len(self.x0)):
            for param_name in list(self.plans.values())[0].params:
                if (param_name, 'pose') in self.state_inds:
                    pose = self.env.get_pos_from_label(param_name, mujoco_frame=False)
                    if pose is not None:                        self.x0[m][self._x_data_idx[STATE_ENUM]][self.state_inds[param_name, 'pose']] = pose
            self.x0[m].setflags(write=False)


    def sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True):
        start_time = time.time()
        task = tuple(task)
        plan = self.plans[task]
        self.reset_to_state(state.copy())

        x0 = state[self._x_data_idx[STATE_ENUM]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon
        sample = Sample(self)
        sample.init_t = 0

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        # self.env.sim_from_plan(plan, 0)

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
            noise[:, 3] *= 0
            noise[:, 7] *= 0
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            arm_joints = self.env.get_arm_joint_angles()
            grip_joints = self.env.get_gripper_joint_angles()
            self._clip_joint_angles(arm_joints[7:], grip_joints[1:], arm_joints[:7], grip_joints[:1], plan)

            X = np.zeros((plan.symbolic_bound))
            X[self.state_inds['baxter', 'rArmPose']] = arm_joints[:7]
            X[self.state_inds['baxter', 'rGripper']] = grip_joints[0]
            X[self.state_inds['baxter', 'lArmPose']] = arm_joints[7:]
            X[self.state_inds['baxter', 'lGripper']] = grip_joints[1]
            if ('baxter', 'ee_left_pos') in self.state_inds:
                X[self.state_inds['baxter', 'ee_left_pos']] = self.env.get_left_ee_pos(mujoco_frame=False)
            if ('baxter', 'ee_right_pos') in self.state_inds:
                X[self.state_inds['baxter', 'ee_right_pos']] = self.env.get_right_ee_pos(mujoco_frame=False)

            prim_val = self.get_prim_value(condition, state, task)
            for param_name in plan.params:
                if (param_name, 'pose') in self.state_inds and plan.params[param_name]._type == 'Cloth':
                    pose = self.env.get_pos_from_label(param_name, mujoco_frame=False)
                    if pose is not None:
                        if pose[2] < 0.625:
                            pose[2] = 0.625
                        X[self.state_inds[param_name, 'pose']] = pose

            sample.set(STATE_ENUM, X.copy(), t)
            if np.any(np.isnan(X)):
                raise Exception('Nans in state: {0}'.format(X))
            if OVERHEAD_IMAGE_ENUM in self._hyperparams['obs_include']:
                self.reset_to_state(sample.get_X(t=t))
                im = self.mjc_env.render(height=self.image_height, width=self.image_width)
                sample.set(IM_ENUM, im.flatten(), t)
            if LEFT_IMAGE_ENUM in self._hyperparams['obs_include']:
                self.reset_to_state(sample.get_X(t=t))
                im = self.mjc_env.render(height=self.image_height, width=self.image_width)
                sample.set(IM_ENUM, im.flatten(), t)
            if RIGHT_IMAGE_ENUM in self._hyperparams['obs_include']:
                self.reset_to_state(sample.get_X(t=t))
                im = self.mjc_env.render(height=self.image_height, width=self.image_width)
                sample.set(IM_ENUM, im.flatten(), t)

            sample.set(NOISE_ENUM, noise[t], t)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            self.fill_sample(condition, sample, X, t, task, fill_obs=True)
            sample.condition = condition
            state = sample.get_X(t=t)

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            if np.any(np.isnan(obs)):
                raise Exception('Nans in obs: {0}'.format(obs))

            U = policy.act(state, obs, t, noise[t])
            if np.any(np.isnan(U)):
                raise Exception('Nans in action: {0}'.format(U))

            if np.any(np.isnan(U)):
                raise Exception('Nans in action passed to env.')
                U[np.isnan(U)] = 0

            sample.set(ACTION_ENUM, U.copy(), t)
            if np.any(np.isnan(sample.get(ACTION_ENUM, t=t))):
                raise Exception('Nans in action from time {0}'.format(t))

            self.env.step(np.r_[U[self.action_inds['baxter', 'ee_right_pos']],
                                U[self.action_inds['baxter', 'rGripper']],
                                U[self.action_inds['baxter', 'ee_left_pos']],
                                U[self.action_inds['baxter', 'lGripper']]])

            self.traj_hist.append(U)
            while len(self.traj_hist) > self.hist_len:
                self.traj_hist.pop(0)

        X = np.zeros((plan.symbolic_bound))
        fill_vector(plan.params, plan.state_inds, X, plan.horizon-1)
        sample.end_state = X
        # print 'Sample time:', time.time() - start_time
        if np.any(np.isnan(sample.get_U(t=1))):
            raise Exception('Nans in final sample action.')
        return sample


    def _clip_joint_angles(self, lArmPose, lGripper, rArmPose, rGripper, plan):
        DOF_limits = plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.000001, DOF_limits[1][2:9]-0.000001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.000001, DOF_limits[1][10:17]-0.00001)
        left_joints = lArmPose
        left_grip = lGripper
        right_joints = rArmPose
        right_grip = rGripper

        if left_grip[0] < 0:
            left_grip[0] = 0.015
        elif left_grip[0] > 0.02:
            left_grip[0] = 0.02

        if right_grip[0] < 0:
            right_grip[0] = 0.015
        elif right_grip[0] > 0.02:
            right_grip[0] = 0.02

        for i in range(7):
            if left_joints[i] < left_DOF_limits[0][i]:
                left_joints[i] = left_DOF_limits[0][i]
            if left_joints[i] > left_DOF_limits[1][i]:
                left_joints[i] = left_DOF_limits[1][i]
            if right_joints[i] < right_DOF_limits[0][i]:
                right_joints[i] = right_DOF_limits[0][i]
            if right_joints[i] > right_DOF_limits[1][i]:
                right_joints[i] = right_DOF_limits[1][i]


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0):
        success = False
        x0 = state[self._x_data_idx[STATE_ENUM]]

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task]
        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for param in list(plan.params.values()):
            if param._type == 'Cloth':
                plan.params['{0}_init_target'.format(param.name)].value[:,0] = param.pose[:,0]

        prim_vals = self.get_prim_value(condition, state, task)
        prim_choices = self.prob.get_prim_choices()
        obj_name = prim_choices[OBJ_ENUM][task[1]]
        targ_name = prim_choices[TARG_ENUM][task[2]]

        plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
        plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
        plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
        plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
        # for t in range(len(traj_mean)):
        #     lArmPose = traj_mean[t, self.state_inds['baxter', 'lArmPose']]
        #     lGripper = traj_mean[t, self.state_inds['baxter', 'lGripper']]
        #     rArmPose = traj_mean[t, self.state_inds['baxter', 'rArmPose']]
        #     rGripper = traj_mean[t, self.state_inds['baxter', 'rGripper']]
        #     self._clip_joint_angles(lArmPose, lGripper, rArmPose, rGripper, plan)
        try:
            success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean, inf_f=inf_f)
        except Exception as e:
            print(e)
            traceback.print_exception(*sys.exc_info())
            success = False
            raise e

        if not success:
            for action in plan.actions:
                try:
                    print('Solve failed on action:')
                    print((action, plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)))
                    print(['{0}: {1}\n'.format(p.name, p.pose[:,0]) for p in list(plan.params.values()) if not p.is_symbol()])
                    print('\n')
                except:
                    pass
            print('\n\n')

        try:
            if not len(failed_preds):
                for action in plan.actions:
                    failed_preds += [(pred, obj_name, targ_name) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
        except Exception as e:
            traceback.print_exception(*sys.exc_info())

        if not success:
            sample = Sample(self)
            self.fill_sample(condition, sample, x0, 0, task, fill_obs=True)

            # vec = np.zeros(len(self.task_list))
            # vec[task[0]] = 1.
            # sample.set(TASK_ENUM, vec, 0)
            # for i in range(len(self.prim_dims.keys())):
            #     enum = self.prim_dims.keys()[i]
            #     vec = np.zeros((self.prim_dims[enum]))
            #     vec[task[i+1]] = 1.
            #     sample.set(enum, vec, 0)

            # set_params_attrs(plan.params, plan.state_inds, x0, 0)

            # sample.set(OBJ_POSE_ENUM, plan.params[obj_name].pose[:,0].copy(), 0)
            # sample.set(TARG_POSE_ENUM, plan.params[targ_name].value[:,0].copy(), 0)

            # sample.set(STATE_ENUM, x0.copy(), 0)
            # for data_type in self._x_data_idx:
            #     sample.set(data_type, state[self._x_data_idx[data_type]], 0)
            # sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), 0)
            # sample.condition = condition
            # sample.task = task
            return sample, failed_preds, success

        return self._sample_opt_traj(plan, state, task, condition)


    def _sample_opt_traj(self, plan, state, task, condition):
        '''
        Only call for successfully planned trajectories
        '''
        baxter = plan.params['baxter']
        baxter_body = baxter.openrave_body
        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                t2 = t+1 if t< plan.horizon-1 else t
                baxter_body.set_dof({'lArmPose': baxter.lArmPose[:,t],
                                     'lGripper': baxter.lGripper[:,t],
                                     'rArmPose': baxter.rArmPose[:,t],
                                     'rGripper': baxter.rGripper[:,t]})
                ee1 = baxter_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], t)
                baxter_body.set_dof({'lArmPose': baxter.lArmPose[:,t2],
                                     'lGripper': baxter.lGripper[:,t2],
                                     'rArmPose': baxter.rArmPose[:,t2],
                                     'rGripper': baxter.rGripper[:,t2]})
                ee2 = baxter_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], t2)
                U[plan.action_inds['baxter', 'ee_left_pos']] = ee2['left_gripper']['pos'] - ee1['left_gripper']['pos']
                U[plan.action_inds['baxter', 'ee_right_pos']] = ee2['right_gripper']['pos'] - ee1['right_gripper']['pos']
                U[plan.action_inds['baxter', 'lGripper']] = baxter.lGripper[:,t2]
                U[plan.action_inds['baxter', 'rGripper']] = baxter.rGripper[:,t2]
                return U

        state_traj = np.zeros((plan.horizon, self.symbolic_bound))
        for i in range(plan.horizon):
            fill_vector(plan.params, plan.state_inds, state_traj[i], i)
        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)
        self.optimal_samples[self.task_list[task[0]]].append(sample)
        while len(self.optimal_samples[self.task_list[0]]) > 25:
            del self.optimal_samples[self.task_list[0]][0]
        sample.set_ref_X(state_traj)
        sample.set_ref_U(sample.get(ACTION_ENUM))
        return sample, [], True


    def reset_to_sample(self, sample):
        self.env.reset()
        self.reset_to_state(sample.get_X(t=0))


    def reset(self, m):
        self.env.reset()
        self.reset_to_state(self.x0[m])


    def reset_to_state(self, x):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        lArmPose = mp_state[self.state_inds['baxter', 'lArmPose']]
        lGripper = mp_state[self.state_inds['baxter', 'lGripper']]
        rArmPose = mp_state[self.state_inds['baxter', 'rArmPose']]
        rGripper = mp_state[self.state_inds['baxter', 'rGripper']]
        self.env.physics.data.qpos[1:8] = rArmPose
        self.env.physics.data.qpos[8:10] = rGripper
        self.env.physics.data.qpos[10:17] = lArmPose
        self.env.physics.data.qpos[17:19] = lGripper
        self.env.physics.data.qvel[:] = 0
        self.env.physics.data.qacc[:] = 0
        for param_name, attr in self.state_inds:
            if attr == 'pose':
                # print param_name, mp_state[self.state_inds[param_name, 'pose']], self.env.get_item_pos(param_name, mujoco_frame=False)
                self.env.set_item_pos(param_name, mp_state[self.state_inds[param_name, 'pose']].copy(), mujoco_frame=False, forward=False)
                # print self.env.get_item_pos(param_name, mujoco_frame=False), '\n\n\n'
        self.env.physics.forward()


    def reset_to_mp_state(self, x):
        lArmPose = x[self.state_inds['baxter', 'lArmPose']]
        lGripper = x[self.state_inds['baxter', 'lGripper']]
        rArmPose = x[self.state_inds['baxter', 'rArmPose']]
        rGripper = x[self.state_inds['baxter', 'rGripper']]
        self.env.physics.data.qpos[1:8] = rArmPose
        self.env.physics.data.qpos[8:10] = rGripper
        self.env.physics.data.qpos[10:17] = lArmPose
        self.env.physics.data.qpos[17:19] = lGripper
        for param_name, attr in self.state_inds:
            if attr == 'pose':
                self.env.set_item_pos(param_name, x[self.state_inds[param_name, 'pose']], mujoco_frame=False)


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        self.reset_to_state(state)
        params = list(self.plans.values())[0].params
        return hl_plan_for_state(state, targets, plan_id, params, self.state_inds, failed_preds)


    def get_next_action(self):
        hl_plan = self.get_hl_plan(None, None, None, None)
        return hl_plan[0]


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False):
        plan = self.plans[task]
        sample.set(STATE_ENUM, mp_state.copy(), t)
        sample.condition = cond

        baxter = plan.params['baxter']
        lArmPose = mp_state[self.state_inds['baxter', 'lArmPose']]
        lGripper = mp_state[self.state_inds['baxter', 'lGripper']]
        rArmPose = mp_state[self.state_inds['baxter', 'rArmPose']]
        rGripper = mp_state[self.state_inds['baxter', 'rGripper']]
        self._clip_joint_angles(lArmPose, lGripper, rArmPose, rGripper, plan)
        baxter.openrave_body.set_dof({'lArmPose': lArmPose, 'lGripper': lGripper, 'rArmPose': rArmPose, 'rGripper': rGripper})
        right_ee = baxter.openrave_body.fwd_kinematics('right_gripper')
        left_ee = baxter.openrave_body.fwd_kinematics('left_gripper')

        # sample.set(RIGHT_EE_POS_ENUM, right_ee['pos'], t)
        # sample.set(RIGHT_EE_QUAT_ENUM, right_ee['quat'], t)
        # sample.set(LEFT_EE_POS_ENUM, left_ee['pos'], t)
        # sample.set(LEFT_EE_QUAT_ENUM, left_ee['quat'], t)
        U = np.zeros(self.dU)

        sample.task = task
        sample.task_name = self.task_list[task[0]]

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.set(TASK_ENUM, task_vec, t)

        prim_choices = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = list(self.prim_dims.keys())[i-1]
            vec = np.zeros((self.prim_dims[enum]))
            vec[task[i]] = 1.
            sample.set(enum, vec, t)

        # How state is represented:
        # - Obj is absolute pos
        # - Targ is displacement from absolute targ pos to obj pos
        # - EE pose is displacement from absolute ee pos to obj pos
        obj = prim_choices[OBJ_ENUM][task[1]]
        targ = prim_choices[TARG_ENUM][task[2]]
        param = plan.params[obj]
        obj_pose = mp_state[self.state_inds[obj, 'pose']]
        sample.set(OBJ_POSE_ENUM, obj_pose, t)
        param.pose[:, t] = obj_pose

        param = plan.params[targ]
        sample.set(TARG_POSE_ENUM, obj_pose - plan.params[targ].value[:,0], t)

        sample.set(RIGHT_EE_POS_ENUM, obj_pose - right_ee['pos'], t)
        sample.set(RIGHT_EE_QUAT_ENUM, right_ee['quat'], t)
        sample.set(RIGHT_GRIPPER_ENUM, mp_state[self.state_inds['baxter', 'rGripper']], t)
        sample.set(LEFT_EE_POS_ENUM, obj_pose - left_ee['pos'], t)
        sample.set(LEFT_EE_QUAT_ENUM, left_ee['quat'], t)
        sample.set(LEFT_GRIPPER_ENUM, mp_state[self.state_inds['baxter', 'lGripper']], t)

        if np.any(np.isnan(sample.get_X(t=t))):
            raise Exception('Nans in state: {0}'.format(sample.get_X(t=t)))

        # Assumes sample is filled in chronological order
        if t > 0:
            ee_right_1 = plan.params[obj].pose[:,t-1] - sample.get(RIGHT_EE_POS_ENUM, t=t-1)
            ee_left_1 = plan.params[obj].pose[:,t-1] - sample.get(LEFT_EE_POS_ENUM, t=t-1)
            U[self.action_inds['baxter', 'ee_right_pos']] = right_ee['pos'] - ee_right_1
            U[self.action_inds['baxter', 'ee_left_pos']] = left_ee['pos'] - ee_left_1
            U[self.action_inds['baxter', 'rGripper']] = mp_state[self.state_inds['baxter', 'rGripper']]
            U[self.action_inds['baxter', 'lGripper']] = mp_state[self.state_inds['baxter', 'lGripper']]
            sample.set(ACTION_ENUM, U, t-1)
            sample.set(ACTION_ENUM, U, t)

        if fill_obs:
            for param_name, attr in self.state_inds:
                param = plan.params[param_name]
                if param.is_symbol(): continue
                if attr == 'pose':
                    self.env.set_item_pos(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False)
                if attr == 'rotation':
                    self.env.set_item_rot(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False, use_euler=True)

            if OVERHEAD_IMAGE_ENUM in self._hyperparams['obs_include'] or OVERHEAD_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(OVERHEAD_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=0, view=False).flatten(), t)
            if LEFT_IMAGE_ENUM in self._hyperparams['obs_include'] or LEFT_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(LEFT_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=3, view=False).flatten(), t)
            if RIGHT_IMAGE_ENUM in self._hyperparams['obs_include'] or RIGHT_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(RIGHT_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=2, view=False).flatten(), t)

        return sample


    def get_prim_options(self, cond, state):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = self.prob.get_prim_choices()
        plan = list(self.plans.values())[0]
        for enum in self.prim_dims:
            if enum == TASK_ENUM: continue
            out[enum] = []
            for item in options[enum]:
                if item in plan.params:
                    param = plan.params[item]
                    if param.is_symbol():
                        out[enum].append(param.value[:,0].copy())
                    else:
                        out[enum].append(mp_state[self.state_inds[item, 'pose']].copy())
                    continue

                val = self.env.get_pos_from_label(item, mujoco_frame=False)
                if val is not None:
                    out[enum] = val
                out[enum].append(val)
            out[enum] = np.array(out[enum])
        return out


    def get_prim_value(self, cond, state, task):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = self.task_list[task[0]]
        plan = self.plans[task]
        options = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = list(self.prim_dims.keys())[i-1]
            item = options[enum][task[i]]
            if item in plan.params:
                param = plan.params[item]
                if param.is_symbol():
                    out[enum] = param.value[:,0]
                else:
                    out[enum] = mp_state[self.state_inds[item, 'pose']]
                continue

            val = self.env.get_pos_from_label(item, mujoco_frame=False)
            if val is not None:
                out[enum] = val

        return out


    def get_prim_index(self, enum, name):
        prim_options = self.prob.get_prim_choices()
        return prim_options[enum].index(name)


    def get_prim_indices(self, names):
        task = [self.task_list.index(names[0])]
        for i in range(1, len(names)):
            task.append(self.get_prim_index(list(self.prim_dims.keys())[i-1], names[i]))
        return tuple(task)


    def _failed_preds(self, Xs, task, condition, active_ts=None, debug=False):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
        plan = self.plans[task]
        tol = 1e-3

        if active_ts == None:
            active_ts = (1, plan.horizon-1)

        for t in range(active_ts[0], active_ts[1]+1):
            set_params_attrs(plan.params, plan.state_inds, Xs[t-active_ts[0]], t)

        robot = plan.params['baxter']
        robot_init_pose = plan.params['robot_init_pose']
        robot_init_pose.lArmPose[:,0] = robot.lArmPose[:,0]
        robot_init_pose.lGripper[:,0] = robot.lGripper[:,0]
        robot_init_pose.rArmPose[:,0] = robot.rArmPose[:,0]
        robot_init_pose.rGripper[:,0] = robot.rGripper[:,0]

        robot_end_pose = plan.params['robot_end_pose']
        robot_end_pose.lArmPose[:,0] = robot.lArmPose[:,-1]
        robot_end_pose.lGripper[:,0] = robot.lGripper[:,-1]
        robot_end_pose.rArmPose[:,0] = robot.rArmPose[:,-1]
        robot_end_pose.rGripper[:,0] = robot.rGripper[:,-1]

        for i in range(self.n_items):
            plan.params['cloth{0}_init_target'.format(i)].value[:,0] = plan.params['cloth{0}'.format(i)].pose[:,0]

        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        # if debug:
        #     print failed_preds
        return failed_preds


    def cost_f(self, Xs, task, condition, active_ts=None, debug=False):
        if active_ts == None:
            active_ts = (1, plan.horizon-1)
        failed_preds = self._failed_preds(Xs, task, condition, active_ts=active_ts, debug=debug)
        prim_choices = self.prob.get_prim_choices()

        cost = 0
        plan = self.plans[task]
        tol = 1e-3

        obj = prim_choices[OBJ_ENUM][task[1]]
        targ = prim_choices[TARG_ENUM][task[2]]
        obj_param = plan.params[obj]
        targ_param = plan.params[targ]

        if active_ts[0] == 0:
            if 'grasp' in plan.actions[1].name.lower():
                if 'right' in plan.actions[1].name.lower():
                    if obj_param.pose[1,0] > 0.1:
                        cost += 1e1
                else:
                    if obj_param.pose[1,0] < -0.1:
                        cost += 1e1
            if 'putdown' in plan.actions[1].name.lower():
                if 'right' in plan.actions[1].name.lower():
                    if targ_param.value[1,0] > 0.1:
                        cost += 1e1
                else:
                    if targ_param.value[1,0] < -0.1:
                        cost += 1e1

        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    # print viol, failed
                    if viol is not None:
                        cost += np.max(viol)
                except:
                    pass

        return cost


    def cost_info(self, Xs, task, cond, active_ts=(0,0)):
        failed_preds = self._failed_preds(Xs, task, cond, active_ts=active_ts)
        plan = self.plans[task]
        if active_ts[0] == -1:
            active_ts = (plan.horizon-1, plan.horizon-1)
        cost_info = [str(plan.actions)]
        tol=1e-3
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    if viol is not None:
                        cost_info.append((t, failed[1].get_type(), viol.flatten()))
                    else:
                        cost_info.append((t, failed[1].get_type(), "Violation returned null"))
                except Exception as e:
                    cost_info.append((t, failed[1].get_type(), 'Error on evaluating: {0}'.format(e)))

        return cost_info


    # def goal_f(self, condition, state):
    #     self.reset_to_state(state)
    #     mp_state = state[self._x_data_idx[STATE_ENUM]]
    #     diff = 0
    #     plan = self.plans.values()[0]
    #     l_targ = plan.params['left_target_1'].value
    #     r_targ = plan.params['right_target_1'].value

    #     for i in range(self.n_items):
    #         pos = mp_state[self.state_inds['cloth{0}'.format(i), 'pose']]
    #         if i > self.n_items / 2:
    #             dist = np.linalg.norm(pos - l_targ)
    #         else:
    #             dist = np.linalg.norm(pos - r_targ)

    #         if dist > 0.1:
    #             diff += dist

    #     return diff


    def goal_f(self, condition, state):
        self.reset_to_state(state)
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        diff = 0.
        plan = list(self.plans.values())[0]
        l_targ = plan.params['left_target_1'].value
        r_targ = plan.params['right_target_1'].value

        for i in range(self.n_items):
            pos = mp_state[self.state_inds['cloth{0}'.format(i), 'pose']]
            if i > self.n_items / 2:
                dist = np.linalg.norm(pos - l_targ)
            else:
                dist = np.linalg.norm(pos - r_targ)

            if dist > 0.1:
                diff += 1.

        return diff / self.n_items


    def perturb_solve(self, sample, perturb_var=0.01, inf_f=None):
        state = sample.get_X(t=0)
        condition = sample.condition
        task = sample.task
        out = self.solve_sample_opt_traj(state, task, condition, traj_mean=sample.get_X(), inf_f=inf_f, mp_var=perturb_var)
        return out


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean)

        exclude_targets = []
        plan = self.plans[task]
        act_traj = np.zeros((plan.horizon, self.dU))
        baxter = plan.params['baxter']
        baxter.lArmPose[:,0] = opt_traj[0, self.state_inds['baxter', 'lArmPose']]
        baxter.lGripper[:,0] = opt_traj[0, self.state_inds['baxter', 'lGripper']]
        baxter.rArmPose[:,0] = opt_traj[0, self.state_inds['baxter', 'rArmPose']]
        baxter.rGripper[:,0] = opt_traj[0, self.state_inds['baxter', 'rGripper']]
        cur_ee = baxter.openrave_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], 0)
        for t in range(plan.horizon-1):
            baxter.lArmPose[:,t] = opt_traj[t+1, self.state_inds['baxter', 'lArmPose']]
            baxter.lGripper[:,t] = opt_traj[t+1, self.state_inds['baxter', 'lGripper']]
            baxter.rArmPose[:,t] = opt_traj[t+1, self.state_inds['baxter', 'rArmPose']]
            baxter.rGripper[:,t] = opt_traj[t+1, self.state_inds['baxter', 'rGripper']]
            next_ee = baxter.openrave_body.param_fwd_kinematics(baxter, ['left_gripper', 'right_gripper'], t+1)
            act_traj[t, self.action_inds['baxter', 'ee_left_pos']] = next_ee['left_gripper']['pos'] - cur_ee['left_gripper']['pos']
            act_traj[t, self.action_inds['baxter', 'lGripper']] = opt_traj[t+1, self.state_inds['baxter', 'lGripper']]
            act_traj[t, self.action_inds['baxter', 'ee_right_pos']] = next_ee['right_gripper']['pos'] - cur_ee['right_gripper']['pos']
            act_traj[t, self.action_inds['baxter', 'rGripper']] = opt_traj[t+1, self.state_inds['baxter', 'rGripper']]
            cur_ee = next_ee
        act_traj[-1] = act_traj[-2]

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, act_traj), condition, state, task, noisy=False)
        self.optimal_samples[self.task_list[task[0]]].append(sample)
        while len(self.optimal_samples[self.task_list[task[0]]]) > 25:
            del self.optimal_samples[self.task_list[task[0]]][0]
        sample.set_ref_X(opt_traj)
        sample.set_ref_U(sample.get_U())
        return sample


    def animate_sample(self, sample):
        for t in range(sample.T):
            mp_state = sample.get(STATE_ENUM, t)
            for param_name, attr in self.state_inds:
                if param_name == 'baxter': continue
                if attr == 'pose':
                    self.mjc_env.set_item_pos(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False)
                elif attr == 'rotation':
                    self.mjc_env.set_item_rot(param_name, mp_state[self.state_inds[param_name, attr]], mujoco_frame=False, use_euler=True)
            self.mjc_env.set_arm_joint_angles(np.r_[mp_state[self.state_inds['baxter', 'rArmPose']], mp_state[self.state_inds['baxter', 'lArmPose']]])
            self.mjc_env.set_gripper_joint_angles(np.r_[mp_state[self.state_inds['baxter', 'rGripper']], mp_state[self.state_inds['baxter', 'lGripper']]])
            self.mjc_env.render(view=True)
            time.sleep(0.2)
