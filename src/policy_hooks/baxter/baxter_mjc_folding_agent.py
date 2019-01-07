from baxter_gym.envs import *

from policy_hooks.baxter.baxter_mjc_agent import BaxterMJCAgent
from policy_hooks.baxter.baxter_mjc_env import BaxterMJCEnv
from policy_hooks.tamp_agent import TAMPAgent
from policy_hooks.utils.mjc_xml_utils import *

class BaxterMJCFoldingAgent(TAMPAgent):
    def __init__(self, hyperparams):
        plans = hyperparams['plans']
        table = get_table()
        cloth_wid = hyperparams['cloth_width']
        cloth_len = hyperparams['cloth_length']
        cloth_spacing = hyperparams['cloth_spacing']
        cloth_radius = hyperparams['cloth_radius']
        cloth_info = {'width': cloth_wid, 'length': cloth_len, 'spacing': cloth_spacing, 'radius': cloth_radius}
        cloth = get_deformable_cloth(cloth_wid, cloth_len, cloth_spacing, cloth_radius, (0.5, -0.2, 0))
        im_h, im_w = hyperparams['image_height'], hyperparams['image_width']
        self.env = BaxterMJCEnv(cloth_info=cloth_info, im_dims=(im_h, im_w))
        super(TAMPAgent, self).__init__(hyperparams)


    def sample_task(self, policy, condition, x0, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True):
        task = tuple(task)
        plan = self.plans[task]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        self.env.sim_from_plan(plan, 0)

        base_t = 0
        self.T = plan.horizon
        sample = Sample(self)
        sample.init_t = 0

        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = 1e1 * generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            obs_include=['joints', 'end_effector', 'cloth_points']
            obs = self.env.get_obs(obs_include=obs_include)

            joints = self.env.get_obs_data(obs, 'joints')
            ee = self.env.get_obs_data(obs, 'end_effector')
            cloth_points = self.env.get_obs_data(obs, 'cloth_points')

            X = np.zeros((plan.symbolic_bound))
            X[self.state_inds['baxter', 'rArmPose']] = joints[:7]
            X[self.state_inds['baxter', 'rGripper']] = joints[7]
            X[self.state_inds['baxter', 'lArmPose']] = joints[9:16]
            X[self.state_inds['baxter', 'lGripper']] = joints[16]
            prim_val = self.get_prim_value(condition, X, task)
            X['right_corner', 'pose'] = prim_val[RIGHT_TARG_ENUM]
            X['left_corner', 'pose'] = prim_val[LEFT_TARG_ENUM]

            sample.set(STATE_ENUM, X.copy(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            sample.set(CLOTH_POINTS_ENUM, cloth_points.flatten(), t)
            sample.set(CLOTH_JOINTS_ENUM, self.env.get_cloth_joints.flatten(), t)
            self.fill_sample(condition, sample, X, t, task)
            sample.condition = condition

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U = policy.act(sample.get_X(t=t), obs, t, noise[t])
            if np.any(np.isnan(U)):
                U[np.isnan(U)] = 0
            sample.set(ACTION_ENUM, U.copy(), t)
            self.env.step(np.r_[U[self.action_inds['ee_right_pos']],
                                U[self.action_inds['right_gripper']],
                                U[self.action_inds['ee_left_pos']],
                                U[self.action_inds['left_gripper']]],
                          obs_include=obs_include)
            
            self.traj_hist.append(U)
            while len(self.traj_hist) > self.hist_len:
                self.traj_hist.pop(0)

            self.run_policy_step(U, X, self.plans[task[:2]], t)

        X = np.zeros((plan.symbolic_bound))
        fill_vector(plan.params, plan.state_inds, X, plan.horizon-1)
        sample.end_state = X
        return sample


    def _clip_joint_angles(self, u, plan):
        DOF_limits = plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.000001, DOF_limits[1][2:9]-0.000001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.000001, DOF_limits[1][10:17]-0.00001)
        left_joints = u[plan.action_inds['baxter', 'lArmPose']]
        left_grip = u[plan.action_inds['baxter', 'lGripper']]
        right_joints = u[plan.action_inds['baxter', 'rArmPose']]
        right_grip = u[plan.action_inds['baxter', 'rGripper']]

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

        u[plan.action_inds['baxter', 'lArmPose']] = left_joints
        u[plan.action_inds['baxter', 'lGripper']] = left_grip
        u[plan.action_inds['baxter', 'rArmPose']] = right_joints
        u[plan.action_inds['baxter', 'rGripper']] = right_grip


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def solve_sample_optimal_traj(self, state, task, condition, traj_mean=[]):
        success = False

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task] 
        set_params_attrs(plan.params, plan.state_inds, state, 0)

        prim_vals = self.get_prim_value(condition, state, task)            
        plan.params['left_corner'].pose[:,0] = prim_vals[utils.LEFT_TARG_ENUM]
        plan.params['left_corner_init_target'].value[:, 0] = plan.params['left_corner'].pose[:,0]
        plan.params['left_corner_end_target'].value[:, 0] = plan.params['left_corner'].pose[:,0]
        plan.params['right_corner'].pose[:,0] = prim_vals[utils.RIGHT_TARG_ENUM]
        plan.params['right_corner_init_target'].value[:, 0] = plan.params['right_corner'].pose[:,0]
        plan.params['right_corner_end_target'].value[:, 0] = plan.params['right_corner'].pose[:,0]

        plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
        plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
        plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
        plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
        try:
            success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean)
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            success = False

        if not success:
            for action in plan.actions:
                try:
                    print plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)
                except:
                    pass
            print '\n\n'

        try:
            if not len(failed_preds):
                for action in plan.actions:
                    failed_preds += [(pred, targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
        except:
            pass


        if not success:
            for i in range(len(self.prim_dims.keys())):
                enum = self.prim_dims.keys()[i]
                vec = np.zeros((self.prim_dims[enum]))
                vec[task[i]] = 1.
                sample.set(enum, vec, 0)
            
            set_params_attrs(plan.params, plan.state_inds, state, 0)
            
            sample.set(RIGHT_TARG_ENUM, plan.params['right_corner'].pose[:,0].copy(), 0)
            sample.set(LEFT_TARG_ENUM, plan.params['left_corner'].pose[:,0].copy(), 0)

            sample = Sample(self)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), 0)
            sample.condition = condition
            sample.task = task
            return sample, failed_preds, success

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    fill_vector(plan.params, plan.action_inds, U, t+1)
                else:
                    fill_vector(plan.params, plan.action_inds, U, t)
                return U

        sample = self.sample_task(optimal_pol(), condition, state, task, noisy=False)
        self.optimal_samples[task].append(sample)
        return sample, failed_preds, success


    def fill_sample(self, cond, sample, state, t, task):
        plan = self.plans[task]
        sample.set(STATE_ENUM, state.copy(), t)

        baxter = plan.params['baxter']
        lArmPose = state[self.state_inds['baxter', 'lArmPose']]
        lGripper = state[self.state_inds['baxter', 'lGripper']]
        rArmPose = state[self.state_inds['baxter', 'rArmPose']]
        rGripper = state[self.state_inds['baxter', 'rGripper']]
        baxter.openrave_body.set_dof({'lArmPose': lArmPose, 'lGripper': lGripper, 'rArmPose': rArmPose, 'rGripper': rGripper})
        right_ee = baxter.openrave_body.fwd_kinematics('right_gripper')
        left_ee = baxter.openrave_body.fwd_kinematics('left_gripper')

        sample.set(EE_RIGHT_POS_ENUM, right_ee['pos'], t)
        sample.set(EE_RIGHT_QUAT_ENUM, right_ee['quat'], t)
        sample.set(EE_LEFT_POS_ENUM, left_ee['pos'], t)
        sample.set(EE_LEFT_QUAT_ENUM, left_ee['quat'], t)

        sample.label = task

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.set(TASK_ENUM, task_vec, t)
        sample.task = self.task_list[task[0]]

        prim_options = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
            vec = np.zeros((self.prim_dims[enum]))
            vec[task[i]] = 1.
            sample.set(enum, vec, t)

        right_targ = prim_choices[RIGHT_TARG_ENUM][task[1]]
        left_targ = prim_choices[LEFT_TARG_ENUM][task[2]]
        if right_targ not in plan.params:
            sample.set(RIGHT_TARG_POSE_ENUM, state[self.state_inds['right_corner', 'pose']], t)
        else:
            sample.set(RIGHT_TARG_POSE_ENUM, plan.params[right_targ].value[:,0].copy(), t)

        if left_targ not in plan.params:
            sample.set(LEFT_TARG_POSE_ENUM, state[self.state_inds['left_corner', 'pose']], t)
        else:
            sample.set(LEFT_TARG_POSE_ENUM, plan.params[left_targ].value[:,0].copy(), t)



    def get_prim_options(self, cond, state):
        outs = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = self.prob.get_prim_options()
        plan = self.plans.values()[0]
        for enum in self.prim_out_data_types:
            if enum == TASK_ENUM: continue
            out[enum] = []
            for item in options[enum]:
                if item in plan.params:
                    param = plan.params[item]
                    if param.is_symbol():
                        out[enum].append(param.value[:,0].copy())
                    else:
                        out[enum].append(state[self.state_inds[item, 'pose']].copy())
                    continue

                val = self.env.get_pos_from_label(item)
                if val is not None:
                    out[enum] = val
                out[enum].append(val)
            out[enum] = np.array(out[enum])
        return outs


    def get_prim_value(self, cond, state, task):
        out = {}
        out[TASK_ENUM] = self.task_list[task[0]]
        plan = self.plans[task]
        options = self.prob.get_prim_options()
        for in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
            item = options[enum][task[i]]
            if item in plan.params:
                param = plan.params[item]
                if param.is_symbol():
                    out[enum] = param.value[:,0]
                else:
                    out[enum] = state[self.state_inds[item, 'pose']]
                continue

            val = self.env.get_pos_from_label(item)
            if val is not None:
                out[enum] = val

        return out


    def get_prim_index(self, enum, name):
        prim_options = se;f.prob.get_prim_options()
        return prim_options[enum].index(name)
