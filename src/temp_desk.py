import numpy as np

import robodesk

import main
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from pma.robot_solver import RobotSolver
import core.util_classes.transform_utils as T

from policy_hooks.utils.load_agent import *
import policy_hooks.robodesk.hyp as hyp
import policy_hooks.robodesk.desk_prob as prob

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

#goal = '(Lifted ball panda)'
goal = '(NearApproachRight panda ball)'
solver = RobotSolver()
plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=5)


import ipdb; ipdb.set_trace()

config = hyp.refresh_config()
agent_config = load_agent(config)
agent = build_agent(agent_config)

