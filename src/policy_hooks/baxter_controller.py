import numpy as np
POS_VEL_MODE = 1
ACC_MODE = 2

class BaxterMujocoController(object):
    def __init__(self, model, pos_gains=1, vel_gains=1, acc_gains=1):
        '''
            This defines a P Controller for a Baxter simulated in Mujoco

            All gains are assumed to either have a dimension of 16 (7 joints per arm & 1 joint per gripper) to match with the real Baxter or to be scalars

            Some good values for pos/vel gains (empircally tested):
                pos_gains = 2.5e2
                vel_gains = 1e1
        '''
        self.pos_gains = pos_gains
        self.vel_gains = vel_gains
        self.acc_gains = acc_gains
        self.model = model
        self._prev_acc = np.zeros((14,))

    def _calculate_target_acceleration(self, plan, t, init_vel, real_ts_offset):
        if t == plan.horizon - 1:
            return np.zeros((14,))

        x_0 = np.zeros((14,))
        x_target = np.zeros((14,))
        a = np.zeros((14,))
        fill_vector(params, plan.action_inds, x_0, t)
        fill_vector(params, plan.actions_inds, x_target, t)
        real_t = plan.time[0, t] - real_ts_offset
        a = 2*(x_target - x_0 - init_vel * real_t) / (real_t**2)

        return action_inds

    def _pos_error(self, t_target):
        # Mujoco joint order is assumed to be: head, right_arm, right gripper fingers, left arm, left gripper fingers
        cur_pos = self.model.data.qpos.flatten()
        return np.r_[t_target[:8] - cur_pos[1:9], t_target[8:] - cur_pos[10:18]]

    def _vel_error(self, t_target):
        # Mujoco joint order is assumed to be: head, right_arm, right gripper fingers, left arm, left gripper fingers
        cur_vel = self.model.data.qvel.flatten()
        return np.r_[t_target[:8] - cur_vel[1:9], t_target[8:] - cur_vel[10:18]]

    def step_control_loop(self, plan, next_timestep, real_t, mode=POS_VEL_MODE):
        '''
            Returns a torque delta to follow a given trajectory

            PARAMETERS:
                plan: A TAMP plan containing a baxter object that stores trajectory information
                next_timestep: The target timestep within the trajectory; represented as an index into the trajectory
                real_t: The actual time into the current timestep the loop is (used ot calculate the true time remaining until next_timestep) 
                mode: How the loop decides the torque deltas
        '''
        t = next_timestep
        baxter = plan.params['baxter']
        time_to_go = plan.time[:, t] - real_t
        if mode == POS_VEL_MODE:
            pos_target = np.r_[baxter.rArmPose[:, t], baxter.rGripper[:, t], baxter.lArmPose[:, t], baxter.lGripper[:, t]]
            pos_error = self._pos_error(pos_target)
            vel_error = self._vel_error(pos_error / time_to_go)
            return self.pos_gains * (np.abs(pos_error) * pos_error) + self.vel_gains * vel_error

    def convert_torques_to_mujoco(self, torques):
        return np.r_[torques[:8], -torques[7], torques[8:], -torques[15]].reshape((18, 1))

    def get_model(self):
        return self.model