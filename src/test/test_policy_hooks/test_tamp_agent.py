import unittest, time, main, ipdb

import numpy as np

from mujoco_py import mjcore, mjviewer
from mujoco_py.mjlib import mjlib

from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from pma import hl_solver
from policy_hooks import policy_solver, tamp_agent, policy_hyperparams, policy_solver_utils


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

class TestTampAgent(unittest.TestCase):
    def test_laundry_agent_load(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print("loading laundry problem...")
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_policy.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '1: CLOTH_GRASP BAXTER CLOTH_1 CLOTH_TARGET_BEGIN_1 ROBOT_INIT_POSE CG_EE_1 ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        plan.params['baxter'].time = np.ones((1, plan.horizon))

        plans = [plan]
        for plan in plans:
            plan.dX, plan.state_inds, plan.dU, plan.action_inds = policy_solver_utils.get_plan_to_policy_mapping(plan, u_attrs=set(['lArmPose', 'lGripper', 'rArmPose', 'rGripper']))
            plan.active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
            plan.T = plan.active_ts[1] - plan.active_ts[0] + 1
        dX, dU = plans[0].dX, plans[0].dU
        active_ts = (plan.actions[0].active_timesteps[0], plan.actions[-1].active_timesteps[1])
        T = active_ts[1] - active_ts[0] + 1

        sensor_dims = {
            policy_solver_utils.STATE_ENUM: dX,
            policy_solver_utils.ACTION_ENUM: dU
        }

        x0 = np.zeros((len(plans), dX))
        for i in range(len(plans)):
            plan = plans[i]
            policy_solver_utils.fill_vector(plan.actions[0].params, plan.state_inds, x0[i], plan.active_ts[0])
        x0 = x0.tolist()

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
        viewer = mjviewer.MjViewer()
        viewer.start()
        viewer.set_model(model)
        viewer.cam.distance = 1
        viewer.cam.elevation = 0
        viewer.cam.lookat[0] += .35
        viewer.cam.lookat[0]

        viewer.loop_once()
        import ipdb; ipdb.set_trace()
        return True

    def test_mujoco_sim(self):
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
        viewer = mjviewer.MjViewer()
        viewer.start()
        viewer.set_model(model)

        U = agent._inverse_dynamics(plan)

        x0 = agent.x0[0]
        active_ts, params = policy_solver_utils.get_plan_traj_info(plan)
        agent._set_simulator_state(x0, plan, active_ts[0])
        model.data.qpos = agent._baxter_to_mujoco(plan, 0)

        viewer.cam.distance = 6
        viewer.elevation = 90
        viewer.loop_once()
        ipdb.set_trace()
        for t in range(active_ts[0], active_ts[1]+1):
            u = U[:, t]
            mj_u = np.zeros((18,1))
            mj_u[:8] = u[:8].reshape(-1, 1)
            mj_u[8] = -u[7]
            mj_u[9:17] = u[8:].reshape(-1, 1)
            mj_u[17] = -u[15]
            real_t = plan.time[:,t]
            agent.motor_model.data.ctrl = mj_u
            start_t = agent.motor_model.data.time
            cur_t = start_t
            while cur_t < start_t + real_t:
                agent.motor_model.step()
                cur_t += 0.01 # Make sure this value matches the time increment used in Mujoco
            viewer.cam.distance = 6
            viewer.elevation = 90
            viewer.loop_once()
            ipdb.set_trace()


        ipdb.set_trace()
        return True
