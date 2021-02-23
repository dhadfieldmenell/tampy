import numpy as np
import os
import pybullet as P
import sys
import time
import scipy as sp
from scipy.spatial.transform import Rotation

import robosuite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as robo_T
from sco.expr import *

import main
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from core.util_classes.viewer import PyBulletViewer
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from pma.robosuite_solver import RobotSolver
import core.util_classes.transform_utils as T

REF_QUAT = np.array([0, 0, -0.7071, -0.7071])
def theta_error(cur_quat, next_quat):
    sign1 = np.sign(cur_quat[np.argmax(np.abs(cur_quat))])
    sign2 = np.sign(next_quat[np.argmax(np.abs(next_quat))])
    next_quat = np.array(next_quat)
    cur_quat = np.array(cur_quat)
    angle = -(sign1 * sign2) * robo_T.get_orientation_error(sign1 * next_quat, sign2 * cur_quat)
    return angle

#controller_config = load_controller_config(default_controller="OSC_POSE")
#controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
#controller_config['control_delta'] = False
#controller_config['kp'] = 500
#controller_config['kp'] = [750, 750, 500, 5000, 5000, 5000]

ctrl_mode = "JOINT_POSITION"
true_mode = 'IK'
controller_config = load_controller_config(default_controller=ctrl_mode)
if ctrl_mode.find('JOINT') >= 0:
    controller_config['kp'] = [7500, 6500, 6500, 6500, 6500, 6500, 12000]

env = robosuite.make(
    "PickPlace",
    robots=["Sawyer"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=40,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    single_object_mode=2,
    object_type='cereal',
    ignore_done=True,
    initialization_noise={'magnitude': 0.1, 'type': 'gaussian'},
    camera_widths=128,
    camera_heights=128,
)
obs = env.reset()
env.sim.data.qvel[:] = 0
env.sim.data.qacc[:] = 0
env.sim.forward()

bt_ll.DEBUG = True
openrave_bodies = None
domain_fname = "../domains/robot_domain/right_robot.domain"
prob = "../domains/robot_domain/probs/pickplace_prob.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
visual = False
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params
#ll_plan_str = ["0: MOVE_TO_GRASP_LEFT BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_END_POSE"]
#plan = hls.get_plan(ll_plan_str, domain, problem)
#plan.d_c = d_c
#baxter = plan.params['baxter']
#print(plan.get_failed_preds((0,0)))
body_ind = env.mjpy_model.body_name2id('robot0_base')
params['sawyer'].pose[:,0] = env.sim.data.body_xpos[body_ind]

cereal_adr = env.mjpy_model.joint_name2id('cereal_joint0')
cereal_ind = env.mjpy_model.jnt_qposadr[cereal_adr]
pos = env.sim.data.qpos[cereal_ind:cereal_ind+3]
quat = env.sim.data.qpos[cereal_ind+3:cereal_ind+7]
quat = [quat[1], quat[2], quat[3], quat[0]]
euler = T.quaternion_to_euler(quat, 'xyzw')
params['cereal'].pose[:,0] = pos - np.array([0, 0, 0.035])
params['cereal'].rotation[:,0] = euler

params['bread'].pose[:,0] = [10,10,0]
params['milk'].pose[:,0] = [-10,10,0]
params['can'].pose[:,0] = [-10,10,0]

params['milk_init_target'].value[:,0] = params['milk'].pose[:, 0]
params['milk_init_target'].rotation[:,0] = params['milk'].rotation[:, 0]
params['cereal_init_target'].value[:,0] = params['cereal'].pose[:, 0]
params['cereal_init_target'].rotation[:,0] = params['cereal'].rotation[:, 0]
params['can_init_target'].value[:,0] = params['can'].pose[:, 0]
params['can_init_target'].rotation[:,0] = params['can'].rotation[:, 0]
params['bread_init_target'].value[:,0] = params['bread'].pose[:, 0]
params['bread_init_target'].rotation[:,0] = params['bread'].rotation[:, 0]

jnts = params['sawyer'].geom.jnt_names['right']
jnts = ['robot0_'+ jnt for jnt in jnts]
jnt_vals = []
sawyer_inds = []
for jnt in jnts:
    jnt_adr = env.mjpy_model.joint_name2id(jnt)
    jnt_ind = env.mjpy_model.jnt_qposadr[jnt_adr]
    sawyer_inds.append(jnt_ind)
    jnt_vals.append(env.sim.data.qpos[jnt_ind])
params['sawyer'].right[:,0] = jnt_vals
params['sawyer'].openrave_body.set_pose(params['sawyer'].pose[:,0])
params['sawyer'].openrave_body.set_dof({'right': params['sawyer'].right[:,0]})
info = params['sawyer'].openrave_body.fwd_kinematics('right')
params['sawyer'].right_ee_pos[:,0] = info['pos']
params['sawyer'].right_ee_pos[:,0] = T.quaternion_to_euler(info['quat'], 'xyzw')

goal = '(NearGripperRight sawyer cereal)' #'(At cereal cereal_end_target)'
#goal = '(At cereal cereal_end_target)'
solver = RobotSolver()
load_traj = False
replan = True
if load_traj:
    oldplan = np.load('MotionServer5.pkl', allow_pickle=True)
    if replan:
        for pname in oldplan.params:
            if pname.find('cereal') <0 and pname.find('sawyer') < 0: continue
            for attr in oldplan.params[pname]._free_attrs:
                if type(getattr(params[pname], attr)) is str: continue
                print('SETTING', pname, attr)
                getattr(params[pname], attr)[:,0] = getattr(oldplan.params[pname], attr)[:,0]

if not replan:
    plan = oldplan

if replan:
    plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)
if len(sys.argv) > 1 and sys.argv[1] == 'end':
    sys.exit(0)

#if load_traj:
#    inds, traj = np.load('MotionServer0_17.npy', allow_pickle=True)
#    import ipdb; ipdb.set_trace()
#    for anum, act in enumerate(plan.actions):
#        for pname, aname in inds:
#            for t in range(act.active_timesteps[0], act.active_timesteps[1]+1):
#                getattr(plan.params[pname], aname)[:,t] = traj[t-anum][inds[pname, aname]]

sawyer = plan.params['sawyer']
cmds = []
for t in range(plan.horizon):
    rgrip = sawyer.right_gripper[0,t]
    if true_mode.find('JOINT') >= 0:
        act = np.r_[sawyer.right[:,t], [-rgrip]]
    else:
        pos, euler = sawyer.right_ee_pos[:,t], sawyer.right_ee_rot[:,t]
        quat = np.array(T.euler_to_quaternion(euler, 'xyzw'))
        #angle = robosuite.utils.transform_utils.quat2axisangle(quat)

        rgrip = sawyer.right_gripper[0,t]
        act = np.r_[pos, quat, [-1e1*rgrip]]
        #act = np.r_[pos, angle, [-rgrip]]
        #act = np.r_[sawyer.right[:,t], [-rgrip]]
    cmds.append(act)

grip_ind = env.mjpy_model.site_name2id('gripper0_grip_site')
hand_ind = env.mjpy_model.body_name2id('robot0_right_hand')
env.reset()
env.sim.data.qpos[cereal_ind:cereal_ind+3] = plan.params['cereal'].pose[:,0]
env.sim.data.qpos[cereal_ind+3:cereal_ind+7] = T.euler_to_quaternion(plan.params['cereal'].rotation[:,0], 'wxyz')
env.sim.data.qpos[:7] = params['sawyer'].right[:,0]
env.sim.data.qacc[:] = 0
env.sim.data.qvel[:] = 0
env.sim.forward()
rot_ref = T.euler_to_quaternion(params['sawyer'].right_ee_rot[:,0], 'xyzw') 
for _ in range(40):
    if ctrl_mode.find('JOINT') >= 0:
        env.step(np.zeros(8))
    else:
        env.step(np.zeros(7))
    env.sim.data.qacc[:] = 0
    env.sim.data.qvel[:] = 0
    env.sim.data.qpos[:7] = params['sawyer'].right[:,0]
    env.sim.forward()
    env.render()
env.render()

nsteps = 50
cur_ind = 0
tol=1e-3
true_lb, true_ub = plan.params['sawyer'].geom.get_joint_limits('right')
factor = (np.array(true_ub) - np.array(true_lb)) / 5
ref_jnts = env.sim.data.qpos[:7]
for act in plan.actions:
    t = act.active_timesteps[0]
    plan.params['sawyer'].right[:,t] = env.sim.data.qpos[:7]
    plan.params['cereal'].pose[:,t] = env.sim.data.qpos[cereal_ind:cereal_ind+3]
    plan.params['cereal'].rotation[:,t] = T.quaternion_to_euler(env.sim.data.qpos[cereal_ind+3:cereal_ind+7], 'wxyz')
    failed_preds = plan.get_failed_preds(active_ts=(t,t), priority=3, tol=tol)
    #failed_preds = [p for p in failed_preds if (p[1]._rollout or not type(p[1].expr) is EqExpr)]
    print('FAILED:', t, failed_preds)
    import ipdb; ipdb.set_trace()

    sawyer = plan.params['sawyer']
    for t in range(act.active_timesteps[0], act.active_timesteps[1]):
        base_act = cmds[cur_ind]
        cur_ind += 1
        print('TIME:', t)
        init_jnts = env.sim.data.qpos[:7]
        if ctrl_mode.find('JOINT') >= 0 and true_mode.find('JOINT') <= 0:
            cur_jnts = env.sim.data.qpos[:7]
            if t < plan.horizon:
                targ_pos, targ_rot = sawyer.right_ee_pos[:,t+1], sawyer.right_ee_rot[:,t+1]
            else:
                targ_pos, targ_rot = sawyer.right_ee_pos[:,t], sawyer.right_ee_rot[:,t]
            lb = env.sim.data.qpos[:7] - factor
            ub = env.sim.data.qpos[:7] + factor
            sawyer.openrave_body.set_dof({'right': np.zeros(7)})

            targ_jnts = sawyer.openrave_body.get_ik_from_pose(targ_pos, targ_rot, 'right', bnds=(lb, ub))
            base_act = np.r_[targ_jnts, base_act[-1]]

        true_act = base_act.copy()
        if ctrl_mode.find('JOINT') >= 0:
            targ_jnts = base_act[:7] #+ env.sim.data.qpos[:7]
            for n in range(nsteps):
                act = base_act.copy()
                act[:7] = (targ_jnts - env.sim.data.qpos[:7])
                obs = env.step(act)
            print('END ERROR:', act[:7], true_act[:7]-env.sim.data.qpos[:7])
            end_jnts = env.sim.data.qpos[:7]
            print('JNT_DELTA:', true_act[:7] - init_jnts)
            print('PLAN VS SIM:', end_jnts, sawyer.right[:,t])
            print('EE PLAN VS SIM:', env.sim.data.site_xpos[grip_ind], sawyer.right_ee_pos[:,t], t)
            print('\n\n\n')

        else:
            targ = base_act[3:7]
            cur = env.sim.data.body_xquat[hand_ind]
            cur = np.array([cur[1], cur[2], cur[3], cur[0]])
            truerot = Rotation.from_quat(targ)
            currot = Rotation.from_quat(cur)
            base_angle = (truerot * currot.inv()).as_rotvec()
            #base_angle = robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
            rot = Rotation.from_rotvec(base_angle)
            targrot = (rot * currot).as_quat()
            #print('TARGETS:', targ, targrot)
            for n in range(nsteps):
                act = base_act.copy()
                act[:3] -= env.sim.data.site_xpos[grip_ind]
                #act[:3] *= 1e2
                cur = env.sim.data.body_xquat[hand_ind]
                cur = np.array([cur[1], cur[2], cur[3], cur[0]])
                #targ = act[3:7]
                sign = np.sign(targ[np.argmax(np.abs(targrot))])
                cur_sign = np.sign(targ[np.argmax(np.abs(cur))])
                targ = targrot
                #if sign != cur_sign:
                #    sign = -1.
                #else:
                #    sign = 1.
                rotmult = 1e0 # 1e1
                ##angle = 5e2*theta_error(cur, targ) #robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
                #angle = robosuite.utils.transform_utils.get_orientation_error(sign*targ, cur)
                #rot = Rotation.from_rotvec(angle)
                #currot = Rotation.from_quat(cur)
                angle = -rotmult*sign*cur_sign*robosuite.utils.transform_utils.get_orientation_error(sign*targrot, cur_sign*cur)
                #a = np.linalg.norm(angle)
                #if a > 2*np.pi:
                #    angle = (a - 2*np.pi)  * angle / a
                act = np.r_[act[:3], angle, act[-1:]]
                #act[3:6] -= robosuite.utils.transform_utils.quat2axisangle(cur)
                #act[:7] = (act[:7] - np.array([env.sim.data.qpos[ind] for ind in sawyer_inds]))
                obs = env.step(act)
        #print('ANGLE:', t, angle, targ, cur)
        #print(base_act[:3], env.sim.data.body_xpos[hand_ind], env.sim.data.site_xpos[grip_ind])
        #print('CEREAL:', t, plan.params['cereal'].pose[:,t], env.sim.data.qpos[cereal_ind:cereal_ind+3])
        env.render()
    import ipdb; ipdb.set_trace()
plan.params['sawyer'].right[:,t] = env.sim.data.qpos[:7]
plan.params['cereal'].pose[:,t] = env.sim.data.qpos[cereal_ind:cereal_ind+3]
plan.params['cereal'].rotation[:,t] = T.quaternion_to_euler(env.sim.data.qpos[cereal_ind+3:cereal_ind+7], 'wxyz')
print('CEREAL END:', plan.params['cereal'].pose[:,t])
failed_preds = plan.get_failed_preds(active_ts=(t,t), priority=3, tol=tol)
#failed_preds = [p for p in failed_preds if (p[1]._rollout or not type(p[1].expr) is EqExpr)]
print('FAILED:', t, failed_preds)
import ipdb; ipdb.set_trace()

