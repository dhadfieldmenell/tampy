import unittest, time, main, ipdb

import numpy as np

from mujoco_py import mjcore, mjviewer
from mujoco_py.mjlib import mjlib

from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from pma import hl_solver
from policy_hooks import baxter_controller, policy_solver_utils, tamp_agent


def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    return domain, problem, params

def traj_retiming(plan, velocity):
    baxter = plan.params['baxter']
    rave_body = baxter.openrave_body
    body = rave_body.env_body
    lmanip = body.GetManipulator("left_arm")
    rmanip = body.GetManipulator("right_arm")
    left_ee_pose = []
    right_ee_pose = []
    for t in range(plan.horizon):
        rave_body.set_dof({
            'lArmPose': baxter.lArmPose[:, t],
            'lGripper': baxter.lGripper[:, t],
            'rArmPose': baxter.rArmPose[:, t],
            'rGripper': baxter.rGripper[:, t]
        })
        rave_body.set_pose([0,0,baxter.pose[:, t]])

        left_ee_pose.append(lmanip.GetTransform()[:3, 3])
        right_ee_pose.append(rmanip.GetTransform()[:3, 3])
    time = np.zeros(plan.horizon)
    # import ipdb; ipdb.set_trace()
    for t in range(plan.horizon-1):
        left_dist = np.linalg.norm(left_ee_pose[t+1] - left_ee_pose[t])
        right_dist = np.linalg.norm(right_ee_pose[t+1] - right_ee_pose[t])
        time_spend = max(left_dist, right_dist)/velocity[t]
        time[t+1] = time_spend
    return time

class TestBaxterController(unittest.TestCase):
    def find_baxter_mujoco_pos_vel_controller(self):
        deserializer = PlanDeserializer()
        plan = deserializer.read_from_hdf5("vel_acc_test_plan.hdf5")
        plan.time = np.ones((1, plan.horizon))

        plans = [plan]
        for plan in plans:
            baxter = plan.params['baxter']
            cloth = plan.params['cloth_0']
            basket = plan.params['basket']
            table = plan.params['table']
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, x_params=[baxter, cloth, basket, table], \
                                                                                                                                                                                                  u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
            plan.T = plan.active_ts[1] - plan.active_ts[0] + 1
        dX, dU = plans[0].dX, plans[0].dU
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            policy_solver_utils.STATE_ENUM: dX,
            policy_solver_utils.ACTION_ENUM: dU
        }

        x0 = []
        for i in range(len(plans)):
            x0.append(np.zeros((dX,)))
            plan = plans[i]
            policy_solver_utils.fill_vector(policy_solver_utils.get_state_params(plan), plan.state_inds, x0[i], plan.active_ts[0])

        config = {
            'type': tamp_agent.LaundryWorldMujocoAgent,
            'x0': x0,
            'plans': plans,
            'T': T,
            'sensor_dims': sensor_dims,
            'state_include': [policy_solver_utils.STATE_ENUM],
            'obs_include': [],
            'conditions': len(plans),
            'dX': dX,
            'dU': dU,
            'solver': None
        }

        agent = tamp_agent.LaundryWorldMujocoAgent(config)
        model = agent.motor_model
        # pos_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01])
        # vel_gains = 5e-3
        pos_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01])
        vel_gains = 5e-3
        # controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)

        # viewer = mjviewer.MjViewer()
        # viewer.start()
        # viewer.set_model(model)

        x0 = agent.x0[0]
        active_ts, params = policy_solver_utils.get_plan_traj_info(plan)

        # viewer.cam.distance = 5
        # viewer.cam.azimuth = 220
        # viewer.cam.elevation = -20
        # viewer.loop_once()
        # ipdb.set_trace()
        # curr_pos_tracker = None

        best_avg_err = np.ones((16,))
        best_gains = np.zeros((32,))
        good_gains = []
        for pos_exp in range(-1,2):
            for vel_exp in range(-5,--4):
                for i in range(10):
                    pos_gains = np.ones((16)) * i * 10**pos_exp
                    vel_gains = np.zeros((16,)) * np.random.uniform(0, 10) * 10**vel_exp
                    controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)
                    avg_err = np.zeros((16,))
                    torques = np.zeros((16,))
                    for t in range(0, 30):
                        cur_t = 0
                        while cur_t < plan.time[:, t]:
                            torques += controller.step_control_loop(plan, t+1, cur_t)
                            model.data.ctrl = controller.convert_torques_to_mujoco(torques)
                            model.step()
                            cur_t += 0.0002
                        cur_pos_error  = controller._pos_error(np.r_[baxter.rArmPose[:, t+1], baxter.rGripper[:, t+1], baxter.lArmPose[:, t+1], baxter.lGripper[:, t+1]])
                        avg_err += cur_pos_error
                    avg_err /= 30
                    if np.mean(avg_err) < np.mean(best_avg_err):
                        best_avg_err = avg_err
                        print(best_avg_err)
                        best_gains = np.r_[pos_gains, vel_gains]

                    if np.all(avg_err) <= 1e-3:
                        good_gains.append(np.r_[pos_gains, vel_gains])

                    agent._set_simulator_state(x0, plan, active_ts[0])
                    model.data.qpos = agent._baxter_to_mujoco(plan, 0)

                for _ in range(10):
                    pos_gains = np.random.random((16,)) * 10**pos_exp
                    vel_gains = np.random.random((16,)) * 10**vel_exp * 0
                    controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)
                    avg_err = np.zeros((16,))
                    torques = np.zeros((16,))
                    for t in range(0, 30):
                        cur_t = 0
                        while cur_t < plan.time[:, t]:
                            torques += controller.step_control_loop(plan, t+1, cur_t)
                            model.data.ctrl = controller.convert_torques_to_mujoco(torques)
                            model.step()
                            cur_t += 0.0002
                        cur_pos_error  = controller._pos_error(np.r_[baxter.rArmPose[:, t+1], baxter.rGripper[:, t+1], baxter.lArmPose[:, t+1], baxter.lGripper[:, t+1]])
                        avg_err += cur_pos_error
                    avg_err /= 30
                    if np.mean(avg_err) < np.mean(best_avg_err):
                        best_avg_err = avg_err
                        best_gains = np.r_[pos_gains, vel_gains]

                    if np.all(avg_err) <= 1e-3:
                        good_gains.append(np.r_[pos_gains, vel_gains])

                    agent._set_simulator_state(x0, plan, active_ts[0])
                    model.data.qpos = agent._baxter_to_mujoco(plan, 0)
                print(best_avg_err)

        print(best_gains)
        print(good_gains)
        print(best_avg_err)
        np.save('best_gains_2', best_gains)
        np.save('good_gains_2', np.array(good_gains))

            # print curr_pos_error
            # if curr_pos_tracker is not None:
            #     print("Error trend")
            #     print curr_pos_error - curr_pos_tracker
            # curr_pos_tracker = curr_pos_error
            # viewer.cam.distance = 5
            # viewer.cam.azimuth = 220
            # viewer.cam.elevation = -20
            # viewer.loop_once()
            # ipdb.set_trace()
        # ipdb.set_trace()
        return True

    def evaluate_pos_vel_gains(self):
        deserializer = PlanDeserializer()
        plan = deserializer.read_from_hdf5("vel_acc_test_plan.hdf5")
        plan.time = np.ones((1, plan.horizon))

        plans = [plan]
        for plan in plans:
            baxter = plan.params['baxter']
            cloth = plan.params['cloth_0']
            basket = plan.params['basket']
            table = plan.params['table']
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, x_params=[baxter, cloth, basket, table], \
                                                                                                                                                                                                  u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
            plan.T = plan.active_ts[1] - plan.active_ts[0] + 1
        dX, dU = plans[0].dX, plans[0].dU
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            policy_solver_utils.STATE_ENUM: dX,
            policy_solver_utils.ACTION_ENUM: dU
        }

        x0 = []
        for i in range(len(plans)):
            x0.append(np.zeros((dX,)))
            plan = plans[i]
            policy_solver_utils.fill_vector(policy_solver_utils.get_state_params(plan), plan.state_inds, x0[i], plan.active_ts[0])

        config = {
            'type': tamp_agent.LaundryWorldMujocoAgent,
            'x0': x0,
            'plans': plans,
            'T': T,
            'sensor_dims': sensor_dims,
            'state_include': [policy_solver_utils.STATE_ENUM],
            'obs_include': [],
            'conditions': len(plans),
            'dX': dX,
            'dU': dU,
            'solver': None
        }

        agent = tamp_agent.LaundryWorldMujocoAgent(config)
        model = agent.motor_model
        pos_gains = 250 * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        vel_gains = 1e1
        controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)

        viewer = mjviewer.MjViewer()
        viewer.start()
        viewer.set_model(model)
        viewer.cam.distance = 5
        viewer.cam.azimuth = 220
        viewer.cam.elevation = -20
        viewer.loop_once()
        import ipdb; ipdb.set_trace()

        x0 = agent.x0[0]
        active_ts, params = policy_solver_utils.get_plan_traj_info(plan)

        controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)
        avg_err = np.zeros((16,))
        torques = np.ones((16,)) * 0.00001
        error_limits = np.array([.2, .75, .2, .075, .075, .5, .001, .001, .05, .5, .2, .075, .075, .5, .001, .001,])
        for t in range(0, 30):
            cur_t = 0
            cur_pos_error = np.ones((16,))
            i = 1.0;
            while np.any(cur_pos_error > error_limits) and i < 100:#cur_t < plan.time[:,t]:
                torques = controller.step_control_loop(plan, t+1, cur_t)
                model.data.ctrl = controller.convert_torques_to_mujoco(torques)
                model.step()
                cur_t += 0.002
                cur_pos_error  = controller._pos_error(np.r_[baxter.rArmPose[:, t+1], baxter.rGripper[:, t+1], baxter.lArmPose[:, t+1], baxter.lGripper[:, t+1]])
                i += 1.0
                print(cur_pos_error)
            avg_err += cur_pos_error
            viewer.loop_once()
            import ipdb; ipdb.set_trace()
        avg_err /= 30
        print(avg_err)


    def run_baxter_mujoco_pos_vel_controller(self):
        deserializer = PlanDeserializer()
        plan = deserializer.read_from_hdf5("vel_acc_test_plan.hdf5")
        plan.time = np.ones((1, plan.horizon))

        plans = [plan]
        for plan in plans:
            baxter = plan.params['baxter']
            cloth = plan.params['cloth_0']
            basket = plan.params['basket']
            table = plan.params['table']
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, x_params=[baxter, cloth, basket, table], \
                                                                                                                                                                                                  u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
            plan.T = plan.active_ts[1] - plan.active_ts[0] + 1
        dX, dU = plans[0].dX, plans[0].dU
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            policy_solver_utils.STATE_ENUM: dX,
            policy_solver_utils.ACTION_ENUM: dU
        }

        x0 = []
        for i in range(len(plans)):
            x0.append(np.zeros((dX,)))
            plan = plans[i]
            policy_solver_utils.fill_vector(policy_solver_utils.get_state_params(plan), plan.state_inds, x0[i], plan.active_ts[0])

        config = {
            'type': tamp_agent.LaundryWorldMujocoAgent,
            'x0': x0,
            'plans': plans,
            'T': T,
            'sensor_dims': sensor_dims,
            'state_include': [policy_solver_utils.STATE_ENUM],
            'obs_include': [],
            'conditions': len(plans),
            'dX': dX,
            'dU': dU,
            'solver': None
        }

        agent = tamp_agent.LaundryWorldMujocoAgent(config)
        model = agent.motor_model
        # pos_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01])
        # vel_gains = 5e-3
        pos_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 1e3, 0.5, 0.01])
        vel_gains = 5e-3
        # controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)

        # viewer = mjviewer.MjViewer()
        # viewer.start()
        # viewer.set_model(model)

        x0 = agent.x0[0]
        active_ts, params = policy_solver_utils.get_plan_traj_info(plan)

        # viewer.cam.distance = 5
        # viewer.cam.azimuth = 220
        # viewer.cam.elevation = -20
        # viewer.loop_once()
        # ipdb.set_trace()
        # curr_pos_tracker = None

        best_avg_err = np.ones((16,))
        best_gains = np.zeros((32,))
        good_gains = []
        for pos_exp in range(-4, 2):
            for vel_exp in range(-3, -2):
                for i in range(10):
                    pos_gains = np.ones((16)) * i * 10**pos_exp
                    vel_gains = np.ones((16,)) * np.random.uniform(0, 10) * 10**vel_exp
                    controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)
                    avg_err = np.zeros((16,))
                    for t in range(0, 30):
                        cur_t = 0
                        while cur_t < plan.time[:, t]:
                            torques = controller.step_control_loop(plan, t+1, cur_t)
                            model.data.ctrl = controller.convert_torques_to_mujoco(torques)
                            model.step()
                            cur_t += 0.002
                        cur_pos_error  = controller._pos_error(np.r_[baxter.rArmPose[:, t+1], baxter.rGripper[:, t+1], baxter.lArmPose[:, t+1], baxter.lGripper[:, t+1]])
                        avg_err += cur_pos_error
                    avg_err /= 30
                    if np.mean(avg_err) < np.mean(best_avg_err):
                        best_avg_err = avg_err
                        best_gains = np.r_[pos_gains, vel_gains]

                    if np.all(avg_err) <= 1e-3:
                        good_gains.append(np.r_[pos_gains, vel_gains])

                    agent._set_simulator_state(x0, plan, active_ts[0])
                    model.data.qpos = agent._baxter_to_mujoco(plan, 0)

                for _ in range(10):
                    pos_gains = np.random.random((16,)) * 10**pos_exp
                    vel_gains = np.random.random((16,)) * 10**vel_exp
                    controller = baxter_controller.BaxterMujocoController(model, pos_gains=pos_gains, vel_gains=vel_gains)
                    avg_err = np.zeros((16,))
                    for t in range(0, 30):
                        cur_t = 0
                        while cur_t < plan.time[:, t]:
                            torques = controller.step_control_loop(plan, t+1, cur_t)
                            model.data.ctrl = controller.convert_torques_to_mujoco(torques)
                            model.step()
                            cur_t += 0.002
                        cur_pos_error  = controller._pos_error(np.r_[baxter.rArmPose[:, t+1], baxter.rGripper[:, t+1], baxter.lArmPose[:, t+1], baxter.lGripper[:, t+1]])
                        avg_err += cur_pos_error
                    avg_err /= 30
                    if np.mean(avg_err) < np.mean(best_avg_err):
                        best_avg_err = avg_err
                        best_gains = np.r_[pos_gains, vel_gains]

                    if np.all(avg_err) <= 1e-3:
                        good_gains.append(np.r_[pos_gains, vel_gains])

                    agent._set_simulator_state(x0, plan, active_ts[0])
                    model.data.qpos = agent._baxter_to_mujoco(plan, 0)

        print(best_gains)
        print(good_gains)
        print(best_avg_err)
        np.save('best_gains', best_gains)
        np.save('good_gains', np.array(good_gains))

            # print curr_pos_error
            # if curr_pos_tracker is not None:
            #     print("Error trend")
            #     print curr_pos_error - curr_pos_tracker
            # curr_pos_tracker = curr_pos_error
            # viewer.cam.distance = 5
            # viewer.cam.azimuth = 220
            # viewer.cam.elevation = -20
            # viewer.loop_once()
            # ipdb.set_trace()
        # ipdb.set_trace()
        return True
