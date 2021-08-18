import numpy as np
import os
import pybullet as P
import sys
import time

from baxter_gym.envs import BaxterMJCEnv

import main
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from core.util_classes.viewer import PyBulletViewer
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver_gurobias bt_ll
from pma.robot_solver import RobotSolver


'''
server = P.GUI if len(os.environ.get('DISPLAY', '')) else P.DIRECT
P.connect(server)
robot = Baxter()
body = OpenRAVEBody(None, 'baxter', robot)
pos = [0.5, 0.5, 0.3]
quat = OpenRAVEBody.quat_from_v1_to_v2([0,0,1], [0,0,-1])
iks = body.get_ik_from_pose(pos, quat, 'left', multiple=True)
'''

bt_ll.DEBUG = True
env = None
openrave_bodies = None
domain_fname = "../domains/robot_domain/left_robot.domain"
prob = "../domains/robot_domain/probs/left_arm_prob1.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = False # len(os.environ.get('DISPLAY', '')) > 0
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, env, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params
xpos = np.random.uniform(0.4, 0.8)
ypos = np.random.uniform(0.4, 0.9)
params['cloth0'].pose[:,0] = [xpos, ypos, -0.04]

xpos = np.random.uniform(0.1, 0.5)
ypos = np.random.uniform(0.7, 0.8)
xpos = 0.75
ypos = 0.75
params['cloth0_end_target'].value[:,0] = [xpos, ypos, -0.04]
params['baxter'].left_gripper[:,0] = 0.0
for i in range(1):
    params['cloth{}_init_target'.format(i)].value[:,0] = params['cloth{}'.format(i)].pose[:,0]

#ll_plan_str = ["0: MOVE_TO_GRASP_LEFT BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_END_POSE"]
#plan = hls.get_plan(ll_plan_str, domain, problem)
#plan.d_c = d_c
#baxter = plan.params['baxter']
#print(plan.get_failed_preds((0,0)))

goal = '(At cloth0 cloth0_end_target)'
solver = RobotSolver()

plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)
if len(sys.argv) > 1 and sys.argv[1] == 'end':
    sys.exit(0)


baxter = plan.params['baxter']
cmds = []
for t in range(plan.horizon):
    #info = baxter.openrave_body.param_fwd_kinematics(baxter, ['left', 'right'], t)
    #left_pos, left_quat = np.array(info['left']['pos']), info['left']['quat']
    #right_pos, right_quat = np.array(info['right']['pos']), info['right']['quat']
    #left_pos, right_pos = baxter.left_ee_pos[:,t], baxter.right_ee_pos[:,t]
    left_pos, right_pos = baxter.left[:,t], baxter.right[:,t]
    lgrip, rgrip = baxter.left_gripper[:,t], baxter.right_gripper[:,t]
    act = np.r_[right_pos, rgrip, left_pos, lgrip]
    cmds.append(act)

im_dims = (64, 64)
view = True
obs_include = ['forward_image']
#items = [{'name': 'cloth0', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.02, 0.035), 'rgba': '1 1 1 1'}]
items = [{'name': 'cloth0', 'type': 'box', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.02, 0.02, 0.02), 'rgba': '1 1 1 1'}]
items.append({'name': 'table', 'type': 'box', 'is_fixed': True, 'pos': [1.23/2-0.1, 0, 0.97/2-0.375-0.665], 'dimensions': [1.23/2, 2.45/2, 0.97/2], 'rgba': (0, 0.5, 0, 1)})
config = {'include_items': items, 'view': view, 'timestep': 0.002, 'load_render': view, 'image dimensions': im_dims, 'obs_include': obs_include}
env = BaxterMJCEnv.load_config(config)
env.set_item_pos('cloth0', plan.params['cloth0'].pose[:,0])
env.set_arm_joint_angles(np.r_[baxter.right[:,0], baxter.left[:,0]])
start_t = time.time()
env.render(view=False, camera_id=1)
print('Time to render:', time.time() - start_t)
for i in range(10):
    env.render(view=False, camera_id=1)
    print('Time to render again:', time.time() - start_t, i)

env.render(view=True, camera_id=1)
import ipdb; ipdb.set_trace()
nsteps = 10
for act in plan.actions:
    for t in range(act.active_timesteps[0], act.active_timesteps[1]):
        base_act = cmds.pop(0)
        for n in range(nsteps):
            act = base_act.copy()
            act[:3] -= env.get_right_ee_pos()
            act[4:7] -= env.get_left_ee_pos()
            incl = ['joint_angle'] if n > 0 else ['forward_image']
            #env.step(act, mode='end_effector_pos', view=(n==0), obs_include=incl)
            env.step(act, mode='joint_angle', view=(n==0), obs_include=incl)
        print(env.get_left_ee_pos(), env.get_item_pos('cloth0'), base_act[4:7], base_act[7], plan.params['baxter'].left_ee_pos[:,t])
    import ipdb; ipdb.set_trace()
import ipdb; ipdb.set_trace()

