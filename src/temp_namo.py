import baxter_gym
from baxter_gym.envs import MJCEnv

import policy_hooks.namo.arm_prob as prob
prob.NUM_OBJS = 2
prob.FIX_TARGETS = True
prob.NUM_TARGS = 8
prob.N_GRASPS = 4
prob.n_aux = 0
prob.END_TARGETS = prob.END_TARGETS[:8]
prob.domain_file = "../domains/namo_domain/namo_current_arm.domain"
from pma.namo_arm_solver import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from policy_hooks.utils.load_task_definitions import parse_state
from core.util_classes.namo_grip_predicates import angle_diff

plans = prob.get_plans(use_tf=True)
plan = plans[0][(0,0,0,2)]
domain = plan.domain
problem = plan.prob
state = problem.init_state
for p in list(plan.params.values()):
    if p.openrave_body is not None and p.name != 'pr2':
        p.openrave_body.set_pose([20,20])

targ = 'end_target_4'
can0_pose = [1.5, 0.8]
can1_pose = [-5,5.]
plan.params['pr2'].gripper[:,0] = -0.3
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

for param in list(plan.params.values()):
    for attr in param._free_attrs:
        if np.any(np.isnan(getattr(param, attr)[:,0])):
            getattr(param, attr)[:,0] = 0

for param in list(state.params.values()):
    for attr in param._free_attrs:
        if np.any(np.isnan(getattr(param, attr)[:,0])):
            getattr(param, attr)[:,0] = 0

solver = NAMOSolver()
bt_ll.DEBUG = True
bt_ll.TRAJOPT_COEFF = 1e-1
hl_solver = FFSolver(plan.d_c)
# solver.backtrack_solve(plan)
abs_domain = hl_solver.abs_domain

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
goal = '(and (Near can0 end_target_3) (Near can1 end_target_4))'
goal = '(Near can0 end_target_4)'
goal = '(Near can0 end_target_0)'
initial = parse_state(plan, [], 0)
initial = list(set([p.get_rep() for p in initial]))
plans = []
for coeff in [0]: #[1e-1, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e-1, 5, 10]:
    bt_ll.COL_COEFF = coeff
    solver = NAMOSolver()
    solver.col_coeff = coeff
    plan, descr = p_mod_abs(hl_solver, solver, domain, problem, goal=goal, initial=initial, debug=True, n_resamples=10)
    plans.append(plan)
    min_dist = np.inf
    ts = -1
    dists = []
import ipdb; ipdb.set_trace()

fpath = baxter_gym.__path__[0]
view = False
act_jnts = ['joint1', 'joint2', 'wrist', 'left_finger_joint', 'right_finger_joint']
items = []
fname = fpath+'/robot_info/lidar_arm.xml'
items.append({'name': 'can0', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.25, 0.4), 'mass': 5.})
items.append({'name': 'can1', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.25, 0.4), 'mass': 5.})
config = {'include_files': [fname], 'sim_freq': 50, 'include_items': items, 'act_jnts': act_jnts, 'step_mult': 1e1, 'view': view, 'timestep': 0.002, 'load_render':False}
env = MJCEnv.load_config(config)
grip = -0.3
env.set_joints({'joint1': 0, 'joint2': 0, 'wrist': 0, 'left_finger_joint': -0.3, 'right_finger_joint': -0.3}, forward=False)
env.set_item_pos('can0', np.r_[can0_pose, 0.5])
env.set_item_pos('can1', np.r_[can1_pose, 0.5])

pr2 = plan.params['pr2']
act = 0
for t in range(plan.horizon-1):
    jnt1 = pr2.joint1[0,t]
    jnt2 = pr2.joint2[0,t]
    wrist = pr2.wrist[0,t]
    cmd1 = pr2.joint1[0,t+1] - pr2.joint1[0,t]
    cmd2 = pr2.joint2[0,t+1] - pr2.joint2[0,t]
    cmdwrist = pr2.wrist[0,t+1] -  pr2.wrist[0,t]
    nsteps = 10
    grip = pr2.gripper[0,t] * 5
    # x, y = pr2.pose[:,t]
    for n in range(nsteps+1):
        cur1 = jnt1 + float(n)/nsteps * cmd1
        cur2 = jnt2 + float(n)/nsteps * cmd2
        curwrist = wrist + float(n)/nsteps * cmdwrist
        ctrl = [cur1, cur2, curwrist, grip, grip]
        env.step(ctrl, mode='velocity')
    env.step(ctrl, mode='velocity')
    env.step(ctrl, mode='velocity')
    env.step(ctrl, mode='velocity')
    if t == plan.actions[act].active_timesteps[1]:
        act += 1
        import ipdb; ipdb.set_trace()
import ipdb; ipdb.set_trace()
