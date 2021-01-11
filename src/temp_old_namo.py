import baxter_gym
from baxter_gym.envs import MJCEnv

import policy_hooks.namo.sort_prob as prob
prob.NUM_OBJS = 2
prob.FIX_TARGETS = True
prob.NUM_TARGS = 8
prob.N_GRASPS = 4
prob.n_aux = 0
prob.END_TARGETS = prob.END_TARGETS[:8]
prob.domain_file = "../domains/namo_domain/namo_current.domain"
from pma.namo_grip_solver import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from policy_hooks.utils.load_task_definitions import parse_state
from core.util_classes.namo_grip_predicates import angle_diff

plans = prob.get_plans(use_tf=True)[0]
plan = list(plans.values())[0]
solver = NAMOSolver()
hl_solver = FFSolver(plan.d_c)
import ipdb; ipdb.set_trace()

fpath = baxter_gym.__path__[0]
view = False
act_jnts = ['robot_x', 'robot_y', 'robot_theta', 'left_finger_joint', 'right_finger_joint']
items = []
fname = fpath+'/robot_info/lidar_namo.xml'
items.append({'name': 'can0', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.4), 'mass': 5., 'rgba': '1 1 0 1'})
items.append({'name': 'can1', 'type': 'cylinder', 'is_fixed': False, 'pos': (0, 1.5, 0.5), 'dimensions': (0.3, 0.4), 'mass': 5., 'rgba': '1 0 1 1'})
config = {'include_files': [fname], 'sim_freq': 50, 'include_items': items, 'act_jnts': act_jnts, 'step_mult': 1e1, 'view': view, 'timestep': 0.002, 'load_render': True}
env = MJCEnv.load_config(config)
xval, yval = [0,0]
grip = -0.1
theta = 0.
env.set_joints({'robot_x': xval, 'robot_y': yval, 'left_finger_joint': grip, 'right_finger_joint': grip, 'robot_theta': theta}, forward=False)
env.set_item_pos('can0', np.r_[[-2,-2], 0.5])
env.set_item_pos('can1', np.r_[[-2, 2], 0.5])
import ipdb; ipdb.set_trace()

