import numpy as np
import pybullet as p

import robodesk

import main
from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.common_constants as const
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from pma.robot_solver import RobotSolver
import core.util_classes.transform_utils as T

from policy_hooks.multiprocess_main import load_config, setup_dirs, DIR_KEY
from policy_hooks.run_training import argsparser
from policy_hooks.utils.load_agent import *
import policy_hooks.robodesk.hyp as hyp
import policy_hooks.robodesk.desk_prob as prob

args = argsparser()
args.config = 'policy_hooks.robodesk.hyp'
args.render = True

base_config = hyp.refresh_config()
base_config['id'] = 0
base_config.update(vars(args))
base_config['args'] = args
config, config_module = load_config(args, base_config)
config.update(base_config)
agent_config = load_agent(config)
agent = build_agent(agent_config)
env = agent.base_env

import ipdb; ipdb.set_trace()

try:
    p.disconnect()
except Exception as e:
    print(e)

const.NEAR_GRIP_COEFF = 1e-1
const.GRASP_DIST = 0.15
const.APPROACH_DIST = 0.015
const.EEREACHABLE_ROT_COEFF = 8e-3
bt_ll.DEBUG = True
openrave_bodies = None
domain_fname = "../domains/robot_domain/right_desk.domain"
prob = "../domains/robot_domain/probs/robodesk_prob.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
#visual = False
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params

for param in ['ball', 'upright_block', 'flat_block']:
    params[param].pose[:,0] = env.physics.named.data.qpos[param][:3]
    quat = env.physics.named.data.qpos[param][3:7]
    quat = [quat[1], quat[2], quat[3], quat[0]]
    euler = T.quaternion_to_euler(quat)
    params[param].rotation[:,0] = euler

params['upright_block'].rotation[:,0] = [1.57, 1.57, 0.]
params['ball'].rotation[:,0] = [0., 0., 0.]

for param in params:
    targ = '{}_init_target'.format(param)
    if targ in params:
        params[targ].value[:,0] = params[param].pose[:,0]
        params[targ].rotation[:,0] = params[param].rotation[:,0]

#goal = '(NearGripperRight panda ball)'
#goal = '(Lifted flat_block panda)'
#goal = '(Lifted upright_block panda)'
#goal = '(SlideDoorOpen shelf_handle shelf)'
goal = '(SlideDoorOpen drawer_handle drawer)'
solver = RobotSolver()
plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)


import ipdb; ipdb.set_trace()

if visual:
    agent.add_viewer()

import ipdb; ipdb.set_trace()

