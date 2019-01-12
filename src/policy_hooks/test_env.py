import numpy as np
import time, main

from openravepy import quatFromAxisAngle, matrixFromPose, poseFromMatrix, \
axisAngleFromRotationMatrix, KinBody, GeometryType, RaveCreateRobot, \
RaveCreateKinBody, TriMesh, Environment, DOFAffine, IkParameterization, IkParameterizationType, \
IkFilterOptions, matrixFromAxisAngle, quatFromRotationMatrix

from pma import hl_solver, robot_ll_solver
from core.parsing import parse_domain_config, parse_problem_config
from baxter_gym.envs.baxter_cloth_env import BaxterClothEnv
from baxter_gym.envs.baxter_mjc_env import BaxterMJCEnv
from policy_hooks.utils.mjc_xml_utils import *
import policy_hooks.utils.transform_utils as trans_utils


def print_diff(target, qpos):
    print np.r_[target[:8] - qpos[1:9], target[8:] - qpos[10:18]]

def test_move():
    cloth = get_deformable_cloth(4, 3, (1., 0., 0.5))
    env = BaxterMJCEnv(items=[cloth], view=True)
    env.render(camera_id=0)

    act_one = np.zeros((16,))
    env.step(act_one)
    env.render()
    time.sleep(0.5)
    print_diff(act_one, env.physics.data.qpos)

    act_two = np.zeros((16,))
    act_two[0] = -0.75
    act_two[9] = 0.75
    env.step(act_two)
    env.render(camera_id=1)
    time.sleep(0.5)
    print_diff(act_two, env.physics.data.qpos)

    act_three = np.zeros((16,))
    act_three[0] = -1.5
    act_three[9] = 1.5
    env.step(act_three)
    env.render(camera_id=1)
    time.sleep(0.5)
    print_diff(act_three, env.physics.data.qpos)

    env.step(act_two)
    env.render(camera_id=1)
    time.sleep(0.5)
    print_diff(act_two, env.physics.data.qpos)

    env.step(act_one)
    env.render(camera_id=1)
    time.sleep(0.5)
    print_diff(act_one, env.physics.data.qpos[1:19])

    end_pose = np.array([0.7, -1.01026434, -0.078992, 0.90689717, 0.04219482, 1.67547625, -0.1828384, 0., -0.7, -1.06296608, 0.22662184, 1.00532877, -0.10973573, 1.63895279, 0.24181173, 0.])
    env.step(np.array(end_pose / 4.))
    env.render(camera_id=1)
    print_diff(end_pose / 4., env.physics.data.qpos)

    env.step(np.array(end_pose / 2.))
    env.render(camera_id=1)
    print_diff(end_pose / 2., env.physics.data.qpos)

    env.step(3*np.array(end_pose / 4.))
    env.render(camera_id=1)
    print_diff(3*end_pose / 4., env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.close()

def test_cloth_grasp():
    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/folding.prob')
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

    plan_str = [
    '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
    '1: CLOTH_GRASP BAXTER CLOTH0 CLOTH0_INIT_TARGET CLOTH_GRASP_BEGIN_0 CG_EE_LEFT_0 CLOTH_GRASP_END_0',
    ]

    plan = hls.get_plan(plan_str, domain, problem)
    c_wid = 7
    c_len = 4
    c_rad = 0.015
    c_spac = 0.1
    cloth = get_deformable_cloth(c_wid, c_len, c_spac, c_rad, (0.5, -0.4, 0.65+MUJOCO_MODEL_Z_OFFSET))
    table = get_param_xml(plan.params['table'])
    cloth_info={'width': c_wid, 'length': c_len, 'radius': c_rad, 'spacing': c_spac}
    env = BaxterMJCEnv(items=[cloth, table], view=True, cloth_info=cloth_info)
    env.render(camera_id=1)
    print dir(env.physics.model)
    print dir(env.physics.data)
    # for i in range(7):
    #     for j in range(4):
    #         ind = env.physics.model.name2id('B{0}_{1}'.format(j, i), 'body')
    #         print env.physics.data.xpos[ind]

    baxter, cloth = plan.params['baxter'], plan.params['cloth0']
    arm_jnts = env.get_arm_joint_angles()
    baxter.lArmPose[:,0] = arm_jnts[7:]
    baxter.rArmPose[:,0] = arm_jnts[:7]
    plan.params['robot_init_pose'].lArmPose[:,0] = arm_jnts[7:]
    plan.params['robot_init_pose'].rArmPose[:,0] = arm_jnts[:7]
    cloth.pose[:2,0] = (0.57, 0.2)
    plan.params['cloth0_init_target'].value[:2,0] = cloth.pose[:2,0]
    solver = robot_ll_solver.RobotLLSolver()
    result = solver.backtrack_solve(plan, callback = None, verbose=False)

    for t in range(plan.horizon):
        rGrip = 0 if baxter.rGripper[:, t] < 0.016 else 0.02
        lGrip = 0 if baxter.lGripper[:, t] < 0.016 else 0.02
        act = np.r_[baxter.rArmPose[:,t], rGrip, baxter.lArmPose[:,t], lGrip]
        env.step(act, debug=False)
        env.render(camera_id=1)

    for _ in range(5):
        env.step(act, debug=False)
        env.render(camera_id=1)
    time.sleep(0.5)

def test_ee_ctrl_cloth_grasp():
    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/folding.prob')
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

    plan_str = [
    '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
    '1: CLOTH_GRASP BAXTER CLOTH0 CLOTH0_INIT_TARGET CLOTH_GRASP_BEGIN_0 CG_EE_LEFT_0 CLOTH_GRASP_END_0',
    ]

    plan = hls.get_plan(plan_str, domain, problem)
    c_wid = 7
    c_len = 5
    c_rad = 0.02
    c_spac = 0.1
    cloth = get_deformable_cloth(c_wid, c_len, c_spac, c_rad, (0.5, -0.4, 0.65+MUJOCO_MODEL_Z_OFFSET))
    table = get_param_xml(plan.params['table'])
    cloth_info={'width': c_wid, 'length': c_len, 'radius': c_rad, 'spacing': c_spac}
    # env = BaxterMJCEnv(mode='end_effector', items=[cloth], view=True, cloth_info=cloth_info)
    env = BaxterClothEnv(view=True, cloth_info=cloth_info, cloth_pos=(0.5, -0.4, 0.65+MUJOCO_MODEL_Z_OFFSET))
    env.render(camera_id=1)
    env.render(camera_id=1)
    time.sleep(10)

    baxter, cloth = plan.params['baxter'], plan.params['cloth0']
    arm_jnts = env.get_arm_joint_angles()
    baxter.lArmPose[:,0] = arm_jnts[7:]
    baxter.rArmPose[:,0] = arm_jnts[:7]
    plan.params['robot_init_pose'].lArmPose[:,0] = arm_jnts[7:]
    plan.params['robot_init_pose'].rArmPose[:,0] = arm_jnts[:7]
    cloth.pose[:2,0] = (0.5, 0.2)
    plan.params['cloth0_init_target'].value[:2,0] = (0.5, 0.2)
    solver = robot_ll_solver.RobotLLSolver()
    result = solver.backtrack_solve(plan, callback = None, verbose=False)

    for t in range(plan.horizon):
        rGrip = 0 if baxter.rGripper[:, t] < 0.016 else 0.02
        lGrip = 0 if baxter.lGripper[:, t] < 0.016 else 0.02
        ee_cmd = baxter.openrave_body.param_fwd_kinematics(param=baxter, 
                                                           manip_names=['right_gripper', 'left_gripper'], 
                                                           t=t,
                                                           mat_result=False)

        cur_right_ee_pos = env.get_right_ee_pos(False)
        cur_right_ee_rot = env.get_right_ee_rot()
        cur_left_ee_pos = env.get_left_ee_pos(False)
        cur_left_ee_rot = env.get_left_ee_rot()

        act = np.r_[ee_cmd['right_gripper']['pos'] - cur_right_ee_pos,
                    # ee_cmd['right_gripper']['quat'],
                    rGrip,
                    ee_cmd['left_gripper']['pos'] - cur_left_ee_pos,
                    # ee_cmd['left_gripper']['quat'],
                    lGrip]

        old_cmd = ee_cmd

        env.step(act, debug=False)
        env.render(camera_id=1)
        print env.get_left_ee_pos()
        print env.get_cloth_points()

def test_reward():
    c_wid = 5 # 7
    c_len = 3 # 5
    c_rad = 0.02
    c_spac = 0.1
    cloth = get_deformable_cloth(c_wid, c_len, c_spac, c_rad, (0.5, -0.2, 0.65+MUJOCO_MODEL_Z_OFFSET))
    table = get_table()
    cloth_info={'width': c_wid, 'length': c_len, 'radius': c_rad, 'spacing': c_spac}
    env = BaxterMJCEnv(mode='end_effector', items=[cloth, table], view=True, cloth_info=cloth_info)
    env.list_joint_info()
    env.render(camera_id=0)
    for i in range(24+c_wid+1, len(env.physics.data.qpos), 2*c_wid):
        env.physics.data.qpos[i] = np.pi
    env.physics.forward()
    env.render(camera_id=0)
    print 'Cloth points:', env.get_cloth_points()
    print 'State 1:', env.check_cloth_state()
    env.get_reward()
    time.sleep(20)

# test_move()
# test_cloth_grasp()
test_ee_ctrl_cloth_grasp()
# test_reward()
