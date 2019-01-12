import time
from baxter_gym.envs import *

from gps.agent.agent_utils import generate_noise
from gps.sample.sample_list import SampleList

from policy_hooks.baxter.baxter_mjc_agent import BaxterMJCAgent
from policy_hooks.baxter.baxter_mjc_env import BaxterMJCEnv
from policy_hooks.sample import Sample
from policy_hooks.tamp_agent import TAMPAgent
from policy_hooks.utils.mjc_xml_utils import *
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
            u[self.action_inds[param, attr]] = self.opt_traj[t, self.state_inds[param, attr]]
        return u


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
        self.im_h, self.im_w = hyperparams['image_height'], hyperparams['image_width']
        super(BaxterMJCFoldingAgent, self).__init__(hyperparams)
        self.env = BaxterClothEnv(cloth_info=cloth_info, 
                                  im_dims=(self.im_w, self.im_h), 
                                  obs_include=['end_effector', 'cloth_joints', 'cloth_points', 'joints'],
                                  view=False)

        x0s = []
        for m in range(len(self.x0)):
            sample = Sample(self)
            self.fill_sample(m, sample, self.x0[m], 0, tuple(np.zeros(1+len(self.prim_dims.keys()), dtype='int32')))
            self.x0[m] = sample.get_X(t=0)

        if 'cloth_init_joints' not in hyperparams:
            self.cloth_init_joints = []
            for m in range(len(self.x0)):
                self.env.randomize_cloth()
                self.cloth_init_joints.append(self.env.get_cloth_joints())
                if CLOTH_JOINTS_ENUM in hyperparams['state_include']:
                    self.x0[m][self._x_data_idx[CLOTH_JOINTS_ENUM]] = self.env.get_cloth_joints()
                if CLOTH_POINTS_ENUM in hyperparams['state_include']:
                    self.x0[m][self._x_data_idx[CLOTH_POINTS_ENUM]] = self.env.get_cloth_points().flatten()
        else:
            self.cloth_init_joints = hyperparams['cloth_init_joints']
            for m in range(len(self.x0)):
                self.env.set_cloth_joints(self.cloth_init_joints[m])
                if CLOTH_JOINTS_ENUM in hyperparams['state_include']:
                    self.x0[m][self._x_data_idx[CLOTH_JOINTS_ENUM]] = self.env.get_cloth_joints()
                if CLOTH_POINTS_ENUM in hyperparams['state_include']:
                    self.x0[m][self._x_data_idx[CLOTH_POINTS_ENUM]] = self.env.get_cloth_points().flatten()


    def sample_task(self, policy, condition, state, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True):
        start_time = time.time()
        task = tuple(task)
        plan = self.plans[task]
        self.reset_to_state(state)

        x0 = state[self._x_data_idx[STATE_ENUM]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon
        sample = Sample(self)
        sample.init_t = 0

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        self.env.sim_from_plan(plan, 0)

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = 1e1 * generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            arm_joints = self.env.get_arm_joint_angles()
            grip_joints = self.env.get_gripper_joint_angles()

            X = np.zeros((plan.symbolic_bound))
            X[self.state_inds['baxter', 'rArmPose']] = arm_joints[:7]
            X[self.state_inds['baxter', 'rGripper']] = grip_joints[0]
            X[self.state_inds['baxter', 'lArmPose']] = arm_joints[7:]
            X[self.state_inds['baxter', 'lGripper']] = grip_joints[1]
            prim_val = self.get_prim_value(condition, state, task)
            X[self.state_inds['right_corner', 'pose']] = prim_val[RIGHT_TARG_ENUM]
            X[self.state_inds['left_corner', 'pose']] = prim_val[LEFT_TARG_ENUM]

            sample.set(STATE_ENUM, X.copy(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            self.fill_sample(condition, sample, X, t, task, fill_obs=True)
            sample.condition = condition
            state = sample.get_X(t=t)

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U = policy.act(state, obs, t, noise[t])
            if np.any(np.isnan(U)):
                U[np.isnan(U)] = 0
            sample.set(ACTION_ENUM, U.copy(), t)
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
        print 'Sample time:', time.time() - start_time
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


    def solve_sample_opt_traj(self, state, task, condition, traj_mean=[], inf_f=None, mp_var=0):
        success = False
        x0 = state[self._x_data_idx[STATE_ENUM]]

        failed_preds = []
        iteration = 0
        iteration += 1
        plan = self.plans[task] 
        set_params_attrs(plan.params, plan.state_inds, x0, 0)

        prim_vals = self.get_prim_value(condition, x0, task)            
        plan.params['left_corner'].pose[:,0] = prim_vals[utils.LEFT_TARG_ENUM]
        plan.params['left_corner'].pose[:2,0] += np.random.normal(0, mp_var, 2)
        plan.params['left_corner_init_target'].value[:, 0] = plan.params['left_corner'].pose[:,0]
        plan.params['left_corner_end_target'].value[:, 0] = plan.params['left_corner'].pose[:,0]
        plan.params['left_target_pose'].value[:, 0] = plan.params['left_corner'].pose[:,0]
        plan.params['right_corner'].pose[:,0] = prim_vals[utils.RIGHT_TARG_ENUM]
        plan.params['right_corner'].pose[:2,0] += np.random.normal(0, mp_var, 2)
        plan.params['right_corner_init_target'].value[:, 0] = plan.params['right_corner'].pose[:,0]
        plan.params['right_corner_end_target'].value[:, 0] = plan.params['right_corner'].pose[:,0]
        plan.params['right_target_pose'].value[:, 0] = plan.params['right_corner'].pose[:,0]

        plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
        plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
        plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
        plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
        try:
            success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean, inf_f=inf_f)
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
            
            set_params_attrs(plan.params, plan.state_inds, x0, 0)
            
            sample.set(RIGHT_TARG_POSE_ENUM, plan.params['right_corner'].pose[:,0].copy(), 0)
            sample.set(LEFT_TARG_POSE_ENUM, plan.params['left_corner'].pose[:,0].copy(), 0)

            sample = Sample(self)
            sample.set(STATE_ENUM, x0.copy(), 0)
            for data_type in self._x_data_idx:
                sample.set(data_type, state[self._x_data_idx[data_type]], 0)
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


    def reset_to_sample(self, sample):
        self.env.reset()
        self.env.set_cloth_joints(sample.get(CLOTH_JOINTS_ENUM, t=0))


    def reset(self, m):
        self.env.reset()
        self.env.set_cloth_joints(self.cloth_init_joints[m])


    def reset_to_state(self, x, cloth_joints=None):
        mp_state = x[self._x_data_idx[STATE_ENUM]]
        lArmPose = mp_state[self.state_inds['baxter', 'lArmPose']]
        lGripper = mp_state[self.state_inds['baxter', 'lGripper']]
        rArmPose = mp_state[self.state_inds['baxter', 'rArmPose']]
        rGripper = mp_state[self.state_inds['baxter', 'rGripper']]
        self.env.physics.data.qpos[1:8] = rArmPose
        self.env.physics.data.qpos[8:10] = rGripper
        self.env.physics.data.qpos[10:17] = lArmPose
        self.env.physics.data.qpos[17:19] = lGripper
        if cloth_joints is not None:
            self.env.set_cloth_joints(cloth_joints)
        elif CLOTH_JOINTS_ENUM in self._x_data_idx:
            self.env.set_cloth_joints(x[self._x_data_idx[CLOTH_JOINTS_ENUM]])


    def get_hl_plan(self, state, condition, failed_preds, plan_id=''):
        self.reset_to_state(state)
        cloth_state = self.env.check_cloth_state()
        hl_plan = []
        if ONE_FOLD in cloth_state: return hl_plan
        hl_plan.append([['putdown_corner_both_short', ['left_rest_pose', 'right_rest_pose']]])
        if LENGTH_GRASP in cloth_state: return hl_plan
        hl_plan.insert(0, [['grab_corner_both', ['top_left', 'bottom_left']]])
        if TWIST_FOLD in cloth_state: return hl_plan
        hl_plan.insert(0, [['putdown_corner_both_diagonal', ['left_rest_pose', 'right_rest_pose']]])
        if DIAGONAL_GRASP in cloth_state: return hl_plan
        hl_plan.insert(0, [['grab_corner_both', ['leftmost', 'rightmost']]])
        if LEFT_REACHABLE in cloth_state and RIGHT_REACHABLE in cloth_state: return hl_plan

        if IN_LEFT_GRIPPER in cloth_state:
            hl_plan.insert(0, [['putdown_corner_left', ['left_rest_pose', 'right_rest_pose']]])
            return hl_plan

        if IN_RIGHT_GRIPPER in cloth_state:
            hl_plan.insert(0, [['putdown_corner_right', ['left_rest_pose', 'right_rest_pose']]])
            return hl_plan

        if LEFT_REACHABLE in cloth_state:
            hl_plan.insert(0, [['putdown_corner_left', ['left_rest_pose', 'right_rest_pose']]])
            hl_plan.insert(0, [['grab_corner_left', ['rightmost', 'right_rest_pose']]])
            return hl_plan

        if RIGHT_REACHABLE in cloth_state:
            hl_plan.insert(0, [['putdown_corner_right', ['left_rest_pose', 'right_rest_pose']]])
            hl_plan.insert(0, [['grab_corner_right', ['left_rest_pose', 'leftmost']]])
            return hl_plan

        return hl_plan


    def get_next_action(self):
        hl_plan = self.get_hl_plan(None, None, None, None)
        return hl_plan[0]


    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False):
        plan = self.plans[task]
        sample.set(STATE_ENUM, mp_state.copy(), t)

        baxter = plan.params['baxter']
        lArmPose = mp_state[self.state_inds['baxter', 'lArmPose']]
        lGripper = mp_state[self.state_inds['baxter', 'lGripper']]
        rArmPose = mp_state[self.state_inds['baxter', 'rArmPose']]
        rGripper = mp_state[self.state_inds['baxter', 'rGripper']]
        baxter.openrave_body.set_dof({'lArmPose': lArmPose, 'lGripper': lGripper, 'rArmPose': rArmPose, 'rGripper': rGripper})
        right_ee = baxter.openrave_body.fwd_kinematics('right_gripper')
        left_ee = baxter.openrave_body.fwd_kinematics('left_gripper')

        sample.set(RIGHT_EE_POS_ENUM, right_ee['pos'], t)
        sample.set(RIGHT_EE_QUAT_ENUM, right_ee['quat'], t)
        sample.set(LEFT_EE_POS_ENUM, left_ee['pos'], t)
        sample.set(LEFT_EE_QUAT_ENUM, left_ee['quat'], t)
        U = np.zeros(self.dU)

        # Assumes sample is filled in chronological order
        if t > 0:
            ee_right_1 = sample.get(RIGHT_EE_POS_ENUM, t=t-1)
            ee_left_1 = sample.get(LEFT_EE_POS_ENUM, t=t-1)
            U[self.action_inds['baxter', 'ee_right_pos']] = right_ee['pos'] - ee_right_1
            U[self.action_inds['baxter', 'ee_left_pos']] = left_ee['pos'] - ee_left_1
            U[self.action_inds['baxter', 'rGripper']] = mp_state[self.state_inds['baxter', 'rGripper']]
            U[self.action_inds['baxter', 'lGripper']] = mp_state[self.state_inds['baxter', 'lGripper']]
            sample.set(ACTION_ENUM, U, t-1)
            sample.set(ACTION_ENUM, U, t)

        sample.task = task
        sample.task_name = self.task_list[task[0]]

        task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
        task_vec[task[0]] = 1.
        sample.set(TASK_ENUM, task_vec, t)
        sample.task = self.task_list[task[0]]

        prim_choices = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
            vec = np.zeros((self.prim_dims[enum]))
            vec[task[i]] = 1.
            sample.set(enum, vec, t)

        right_targ = prim_choices[RIGHT_TARG_ENUM][task[1]]
        left_targ = prim_choices[LEFT_TARG_ENUM][task[2]]
        if right_targ not in plan.params:
            sample.set(RIGHT_TARG_POSE_ENUM, mp_state[self.state_inds['right_corner', 'pose']], t)
        else:
            sample.set(RIGHT_TARG_POSE_ENUM, plan.params[right_targ].value[:,0].copy(), t)

        if left_targ not in plan.params:
            sample.set(LEFT_TARG_POSE_ENUM, mp_state[self.state_inds['left_corner', 'pose']], t)
        else:
            sample.set(LEFT_TARG_POSE_ENUM, plan.params[left_targ].value[:,0].copy(), t)

        sample.set(CLOTH_POINTS_ENUM, self.env.get_cloth_points().flatten(), t)
        sample.set(CLOTH_JOINTS_ENUM, self.env.get_cloth_joints().flatten(), t)
        if fill_obs:
            if OVERHEAD_IMAGE_ENUM in self._hyperparams['obs_include'] or OVERHEAD_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(OVERHEAD_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=0, view=False).flatten(), t)
            if LEFT_IMAGE_ENUM in self._hyperparams['obs_include'] or LEFT_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(LEFT_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=3, view=False).flatten(), t)
            if RIGHT_IMAGE_ENUM in self._hyperparams['obs_include'] or RIGHT_IMAGE_ENUM in self._hyperparams['prim_obs_include']:
                sample.set(RIGHT_IMAGE_ENUM, self.env.render(height=self.im_h, width=self.im_w, camera_id=2, view=False).flatten(), t)
            
        return sample


    def get_prim_options(self, cond, state):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        outs = {}
        out[TASK_ENUM] = copy.copy(self.task_list)
        options = self.prob.get_prim_choices()
        plan = self.plans.values()[0]
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

                val = self.env.get_pos_from_label(item)
                if val is not None:
                    out[enum] = val
                out[enum].append(val)
            out[enum] = np.array(out[enum])
        return outs


    def get_prim_value(self, cond, state, task):
        mp_state = state[self._x_data_idx[STATE_ENUM]]
        out = {}
        out[TASK_ENUM] = self.task_list[task[0]]
        plan = self.plans[task]
        options = self.prob.get_prim_choices()
        for i in range(1, len(task)):
            enum = self.prim_dims.keys()[i-1]
            item = options[enum][task[i]]
            if item in plan.params:
                param = plan.params[item]
                if param.is_symbol():
                    out[enum] = param.value[:,0]
                else:
                    out[enum] = mp_state[self.state_inds[item, 'pose']]
                continue

            val = self.env.get_pos_from_label(item)
            if val is not None:
                out[enum] = val

        return out


    def get_prim_index(self, enum, name):
        prim_options = se;f.prob.get_prim_options()
        return prim_options[enum].index(name)


    def cost_f(self, Xs, task, condition, active_ts=None, debug=False):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, Xs.shape[0])
        Xs = Xs[:, self._x_data_idx[STATE_ENUM]]
        plan = self.plans[task]
        tol = 1e-3

        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)

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
        plan.params['left_corner_init_target'].value[:,0] = plan.params['left_corner'].pose[:,0]
        plan.params['right_corner_init_target'].value[:,0] = plan.params['right_corner'].pose[:,0]

        failed_preds = plan.get_failed_preds(active_ts=active_ts, priority=3, tol=tol)
        if debug:
            print failed_preds

        cost = 0
        print plan.actions, failed_preds
        for failed in failed_preds:
            for t in range(active_ts[0], active_ts[1]+1):
                if t + failed[1].active_range[1] > active_ts[1]:
                    break

                try:
                    viol = failed[1].check_pred_violation(t, negated=failed[0], tol=tol)
                    if viol is not None:
                        cost += np.max(viol)
                except:
                    pass

        return cost


    def goal_f(self, condition, state):
        self.reset_to_state(state)
        state = self.env.check_cloth_state()
        if ONE_FOLD in state or TWO_FOLD in state: return 0
        if LENGTH_GRASP in state: return 1e1
        if TWIST_FOLD in state: return 2.5e1
        if DIAGONAL_GRASP in state: return 5e1
        if LEFT_REACHABLE in state and RIGHT_REACHABLE in state: return 7.5e1
        return 5e2

    def perturb_solve(self, sample, perturb_var=0.05, inf_f=None):
        state = sample.get(STATE_ENUM, t=0)
        condition = sample.get(condition)
        task = sample.task
        out = self.solve_sample_opt_traj(state, task, condition, traj_mean=sample.get_U(), inf_f=inf_f, mp_var=perturb_var)
        return out


    def sample_optimal_trajectory(self, state, task, condition, opt_traj=[], traj_mean=[]):
        if not len(opt_traj):
            return self.solve_sample_opt_traj(state, task, condition, traj_mean)

        exclude_targets = []
        opt_disp_traj = np.zeros_like(opt_traj)
        for t in range(0, len(opt_traj)-1):
            opt_disp_traj[t] = opt_traj[t+1] - opt_traj[t]

        sample = self.sample_task(optimal_pol(self.dU, self.action_inds, self.state_inds, opt_disp_traj), condition, state, task, noisy=False,)
        self.optimal_samples[task].append(sample)
        sample.set_ref_X(sample.get(STATE_ENUM))
        sample.set_ref_U(sample.get_U())
        return sample
