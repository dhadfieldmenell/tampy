import baxter_gym
from baxter_gym.envs import MJCEnv
import itertools
import policy_hooks.namo.door_prob as prob
prob.NUM_OBJS = 2
prob.FIX_TARGETS = True
prob.NUM_TARGS = 4
prob.N_GRASPS = 4
prob.n_aux = 0
prob.END_TARGETS = prob.END_TARGETS[:8]
prob.domain_file = "../domains/namo_domain/namo_current_door.domain"
from pma.namo_door_solver import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from policy_hooks.utils.load_task_definitions import parse_state
from policy_hooks.utils.policy_solver_utils import *
from core.util_classes.namo_grip_predicates import angle_diff
import pybullet as P
from core.util_classes.openrave_body import *
NAMO_XML = baxter_gym.__path__[0] + '/robot_info/lidar_namo.xml'

plans = prob.get_plans(use_tf=True)
plan = list(plans[0].values())[0]
domain = plan.domain
problem = plan.prob
state = problem.init_state
for p in list(plan.params.values()):
    if p.openrave_body is not None and p.name != 'pr2':
        p.openrave_body.set_pose([20,20])

can0_pose = [-2., -4.]
can1_pose = [3., -5.]
plan.params['pr2'].pose[:,0] = 0.
plan.params['pr2'].theta[:,0] = 0.
plan.params['pr2'].gripper[:,0] = -1.
plan.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
plan.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
plan.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
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
hl_solver = FFSolver(plan.d_c)
# solver.backtrack_solve(plan)
abs_domain = hl_solver.abs_domain

state.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
state.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
state.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
state.params['can0'].pose[:,0] = can0_pose
state.params['can0_init_target'].value[:,0] = can0_pose

state.params['can1'].pose[:,0] = can1_pose
state.params['can1_init_target'].value[:,0] = can1_pose
goal = '(not (DoorClosed door))'
goal = '(DoorClosed door)'
goal = '(and (InCloset can0) (InCloset can1) (DoorClosed door))'
goal = '(and (InCloset can1) (DoorClosed door))'
initial = parse_state(plan, [], 0)
initial = list(set([p.get_rep() for p in initial]))
plans = []
for coeff in [1e-2]: #[1e-1, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e-1, 5, 10]:
    bt_ll.COL_COEFF = coeff
    solver = NAMOSolver()
    solver.col_coeff = coeff
    plan, descr = p_mod_abs(hl_solver, solver, domain, problem, goal=goal, initial=initial, debug=True, n_resamples=10)
    plans.append(plan)
    min_dist = np.inf
    ts = -1
    dists = []
import ipdb; ipdb.set_trace()

view = True
fpath = baxter_gym.__path__[0]
im_dims = (256, 256)
wall_dims = OpenRAVEBody.get_wall_dims('closet')
config = {
    'obs_include': ['can{0}'.format(i) for i in range(2)],
    'include_files': [NAMO_XML],
    'include_items': [],
    'view': view,
    'sim_freq': 25,
    'timestep': 0.002,
    'image_dimensions': im_dims,
    'step_mult': 5e0,
    'act_jnts': ['robot_x', 'robot_y', 'robot_theta', 'right_finger_joint', 'left_finger_joint']
}

items = config['include_items']
prim_options = prob.get_prim_choices()
for name in prim_options[OBJ_ENUM]:
    if name =='pr2': continue
    cur_color = [0, 0, 0, 1] 
    items.append({'name': name, 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 0, 0.5), 'dimensions': (0.3, 0.2), 'rgba': tuple(cur_color), 'mass': 10.})

for i in range(len(wall_dims)):
    dim, next_trans = wall_dims[i]
    next_trans[0,3] -= 3.5
    next_dim = dim # [dim[1], dim[0], dim[2]]
    pos = next_trans[:3,3] # [next_trans[1,3], next_trans[0,3], next_trans[2,3]]
    items.append({'name': 'wall{0}'.format(i), 'type': 'box', 'is_fixed': True, 'pos': pos, 'dimensions': next_dim, 'rgba': (0.2, 0.2, 0.2, 1)})

items.append({'name': 'door', 'type': 'door_2d', 'handle_dims': (0.325, 0.4), 'door_dims': (0.75, 0.1, 0.4), 'hinge_pos': (-1., 3., 0.5), 'is_fixed': True})

config['load_render'] = True 
config['xmlid'] = '{0}'.format('temp_run')
env = MJCEnv.load_config(config)

env.set_item_pos('can0', np.r_[can0_pose, 0.5])
env.set_item_pos('can1', np.r_[can1_pose, 0.5])
env.set_joints({'door_hinge': plan.params['door'].theta[:,0]})

env.render(view=view)
pr2 = plan.params['pr2']
act = 0
for t in range(plan.horizon-1):
    cur_theta = env.get_joints(['robot_theta'], vec=True)[0] # x[self.state_inds['pr2', 'theta']][0]
    cur_x, cur_y, _ = env.get_item_pos('pr2') # x[self.state_inds['pr2', 'pose']]
    rel_x, rel_y = pr2.pose[0,t+1] - cur_x, pr2.pose[1,t+1] - cur_y 
    cmd_x, cmd_y = rel_x, rel_y
    cmd_theta = pr2.theta[0, t+1] - cur_theta
    nsteps = 20
    gripper = pr2.gripper[0,t]
    if gripper < 0:
        gripper = -0.1
    else:
        gripper = 0.1
    for n in range(nsteps+1):
        x = cur_x + float(n)/nsteps * cmd_x
        y = cur_y + float(n)/nsteps * cmd_y
        theta = cur_theta + float(n)/nsteps * cmd_theta
        ctrl_vec = np.array([x, y, theta, 5*gripper, 5*gripper])
        env.step(ctrl_vec, mode='velocity', gen_obs=(n==0), view=(n==0))
    env.step(ctrl_vec, mode='velocity')
    env.step(ctrl_vec, mode='velocity')
    env.step(ctrl_vec, mode='velocity')
    print(t, env.get_joints(['door_hinge']), env.get_item_pos('pr2'), env.get_item_pos('door'))
    if t == plan.actions[act].active_timesteps[1]:
        act += 1
        import ipdb; ipdb.set_trace()
import ipdb; ipdb.set_trace()
