import baxter_gym
from baxter_gym.envs import MJCEnv

import policy_hooks.namo.grip_prob as prob
prob.NUM_OBJS = 2
prob.FIX_TARGETS = True
prob.NUM_TARGS = 8
prob.N_GRASPS = 4
prob.n_aux = 0
prob.END_TARGETS = prob.END_TARGETS[:8]
prob.domain_file = "../domains/namo_domain/namo_current_grip.domain"
from pma.namo_grip_solver import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from policy_hooks.utils.load_task_definitions import parse_state

plans = prob.get_plans()
plan = plans[0][(1,0,6,3)]
plan = plans[0][(0,0,7,2)]
# plan = plans[0][(1,0,0,2)]
domain = plan.domain
problem = plan.prob
state = problem.init_state
for p in plan.params.values():
    if p.openrave_body is not None:
        p.openrave_body.set_pose([20,20])

pr2_pose = [-1.3, -1.6]
can0_pose = [-3.7, -3.6]
can1_pose = [4.3, -4.]
pr2_pose = [-0.1, -3.2]
can0_pose = [-3.7, -3.2]
can1_pose = [3.5, -3.8]
plan.params['pr2'].pose[:,0] = pr2_pose
plan.params['pr2'].gripper[:,0] = -0.1
plan.params['robot_init_pose'].value[:,0] = pr2_pose
plan.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
plan.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
plan.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
plan.params['end_target_3'].value[:,0] = prob.END_TARGETS[3]
plan.params['end_target_4'].value[:,0] = prob.END_TARGETS[4]
plan.params['end_target_5'].value[:,0] = prob.END_TARGETS[5]
plan.params['end_target_6'].value[:,0] = prob.END_TARGETS[6]
plan.params['end_target_7'].value[:,0] = prob.END_TARGETS[7]
plan.params['can0'].pose[:,0] = can0_pose
plan.params['can0_init_target'].value[:,0] = can0_pose
plan.params['can1'].pose[:,0] = can1_pose
plan.params['can1_init_target'].value[:,0] = can1_pose

for param in plan.params.values():
    for attr in param._free_attrs:
        if np.any(np.isnan(getattr(param, attr)[:,0])):
            getattr(param, attr)[:,0] = 0

for param in state.params.values():
    for attr in param._free_attrs:
        if np.any(np.isnan(getattr(param, attr)[:,0])):
            getattr(param, attr)[:,0] = 0

solver = NAMOSolver()
bt_ll.DEBUG = True
hl_solver = FFSolver(plan.d_c)
# solver.backtrack_solve(plan)
abs_domain = hl_solver.abs_domain
import ipdb; ipdb.set_trace()

state.params['pr2'].pose[:,0] = pr2_pose
state.params['robot_init_pose'].value[:,0] = pr2_pose
state.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
state.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
state.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
state.params['end_target_3'].value[:,0] = prob.END_TARGETS[3]
state.params['end_target_4'].value[:,0] = prob.END_TARGETS[4]
state.params['end_target_5'].value[:,0] = prob.END_TARGETS[5]
state.params['end_target_6'].value[:,0] = prob.END_TARGETS[6]
state.params['end_target_7'].value[:,0] = prob.END_TARGETS[7]
state.params['can0'].pose[:,0] = can0_pose
state.params['can0_init_target'].value[:,0] = can0_pose

state.params['can1'].pose[:,0] = can1_pose
state.params['can1_init_target'].value[:,0] = can1_pose
goal = '(and (Near can0 end_target_3) (Near can1 end_target_5))'
goal = '(and (Near can0 end_target_3) (Near can1 end_target_4))'
initial = parse_state(plan, [], 0)
initial = list(set([p.get_rep() for p in initial]))
plan, descr = p_mod_abs(hl_solver, solver, domain, problem, goal=goal, initial=initial, debug=True, n_resamples=10)
import ipdb; ipdb.set_trace()

fpath = baxter_gym.__path__[0]
view = False
act_jnts = ['robot_x', 'robot_y', 'robot_theta', 'left_finger_joint', 'right_finger_joint']
items = []
fname = fpath+'/robot_info/lidar_namo.xml'
items.append({'name': 'can0', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.4), 'mass': 5.})
items.append({'name': 'can1', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.4), 'mass': 5.})
config = {'include_files': [fname], 'sim_freq': 100, 'include_items': items, 'act_jnts': act_jnts, 'step_mult': 1e1, 'view': view, 'timestep': 0.002}
env = MJCEnv.load_config(config)
xval, yval = pr2_pose
grip = -0.1
theta = 0.
env.set_joints({'robot_x': xval, 'robot_y': yval, 'left_finger_joint': grip, 'right_finger_joint': grip, 'robot_theta': theta}, forward=False)
env.set_item_pos('can0', np.r_[can0_pose, 0.5])
env.set_item_pos('can1', np.r_[can1_pose, 0.5])

pr2 = plan.params['pr2']
act = 0
for t in range(plan.horizon-1):
    cmdx, cmdy = pr2.pose[:,t+1] - pr2.pose[:,t]
    theta = -env.get_joints(['robot_theta'])['robot_theta'][0]
    cmdtheta = pr2.theta[0,t+1] - theta
    cmdx, cmdy = pr2.vel[0,t+1]*np.sin(theta), pr2.vel[0,t+1]*np.cos(theta)
    nsteps = int(max(abs(cmdx), abs(cmdy)) / 0.20) + 1
    grip = pr2.gripper[:,t] * 5
    x, y = pr2.pose[:,t]
    x, y, _ = env.get_item_pos('pr2')
    for n in range(nsteps+1):
        curx = x + float(n)/nsteps * cmdx
        cury = y + float(n)/nsteps * cmdy
        curtheta = theta + float(n)/nsteps * cmdtheta
        ctrl = [curx, cury, -curtheta, grip, grip]
        env.step(ctrl, mode='velocity')
    env.step(ctrl, mode='velocity')
    print(t, env.get_item_pos('can0'))
    print(t, env.get_item_pos('can1'))
    print(t, env.get_item_pos('pr2'))
    print(t, (env.get_item_pos('left_finger')+env.get_item_pos('right_finger'))/2.)
    print('\n\n')
    if t == plan.actions[act].active_timesteps[1]:
        act += 1
        import ipdb; ipdb.set_trace()
import ipdb; ipdb.set_trace()

