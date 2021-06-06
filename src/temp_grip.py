import numpy as np
import pybullet as p

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
from policy_hooks.name.grip_agent import *

args = argsparser()
args.config = 'policy_hooks.namo.hyperparams_v98'
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
agent.mjc_env.reset()

try:
    p.disconnect()
except Exception as e:
    print(e)

bt_ll.DEBUG = True
openrave_bodies = None
domain_fname = "../domains/namo_domain/namo_current_holgrip.domain"
prob = "../domains/namo_domain/probs/grip_prob.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
#visual = False
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params
for param in params:
    targ = '{}_init_target'.format(param)
    if targ in params:
        params[targ].value[:,0] = params[param].pose[:,0]
print('CONSISTENT?', problem.init_state.is_consistent())
import ipdb; ipdb.set_trace()
solver = RobotSolver()

plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=3)

import ipdb; ipdb.set_trace()

