import numpy as np
import os
import pybullet as P

import main
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from core.util_classes.viewer import PyBulletViewer
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll


'''
server = P.GUI if len(os.environ.get('DISPLAY', '')) else P.DIRECT
P.connect(server)
robot = Baxter()
body = OpenRAVEBody(None, 'baxter', robot)
pos = [0.5, 0.5, 0.3]
quat = OpenRAVEBody.quat_from_v1_to_v2([0,0,1], [0,0,-1])
iks = body.get_ik_from_pose(pos, quat, 'left', multiple=True)
'''

env = None
openrave_bodies = None
domain_fname = "../domains/robot_domain/robot.domain"
prob = "../domains/robot_domain/probs/base_prob.prob"
ll_plan_str = ["0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE"]
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, env, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params
params['cloth0'].pose[:,0] = [0.6, 0.5, -0.029]
params['cloth1'].pose[:,0] = [0.3, 0.8, -0.029]
params['cloth2'].pose[:,0] = [0.3, -0.8, -0.029]
params['cloth3'].pose[:,0] = [0.6, -0.5, -0.029]
plan = hls.get_plan(ll_plan_str, domain, problem)
plan.d_c = d_c
print(plan.get_failed_preds((0,0)))

import ipdb; ipdb.set_trace()

