import os
import sys

import main
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.viewer import PyBulletViewer
from pma.hl_solver import *
from pma.pr_graph import *
import argparse

domain_file = "../domains/namo_domain/namo_current.domain"
prob_file = "../domains/namo_domain/namo_probs/verify_2_object.prob"

parser = argparse.ArgumentParser(description='Args for the verify NAMO example script')
parser.add_argument('--solver', default='osqp', choices=['gurobi', 'osqp'], help='the backend solver to use. Currently, either osqp or gurobi (default: osqp)') 
args = parser.parse_args()

if args.solver == 'osqp':
    from pma.namo_solver_OSQP import NAMOSolver_OSQP as NAMOSolver
    from pma import backtrack_ll_solver_OSQP as bt_ll
elif args.solver == 'gurobi':
    from pma.namo_solver_gurobi import NAMOSolverGurobi as NAMOSolver
    from pma import backtrack_ll_solver_gurobi as bt_ll

# Create a PyBulletViewer for viz purposes
pbv = PyBulletViewer()
pbv = pbv.create_viewer()

d_c = main.parse_file_to_dict(domain_file)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
p_c = main.parse_file_to_dict(prob_file)
problem = parse_problem_config.ParseProblemConfig.parse(
    p_c, domain, env=pbv.env, openrave_bodies={}, use_tf=False, sess=None
)
state = problem.init_state
solver = NAMOSolver()
bt_ll.DEBUG = True
hl_solver = FFSolver(d_c)
abs_domain = hl_solver.abs_domain

# state represents the initial configuration of the world
# Normally these value are just set in the problem file itself
# However sometimes it's useful to adjust them at run-time; e.g.
# if you wish to programmatically generate sample problems
pr2_pose = [0, -5.0]
can0_pose = [-3.4, -3.8]
can1_pose = [2.4, -1.2]
END_TARGETS = [
    [4.5, 2.0],
    [3.5, 2.0],
    [2.5, 2.0],
    [1.5, 2],
    [-1.5, 2.0],
    [2.5, 2.0],
    [3.5, 2.0],
    [4.5, 2.0],
]
state.params["pr2"].pose[:, 0] = pr2_pose
state.params["robot_init_pose"].value[:, 0] = pr2_pose
state.params["end_target_0"].value[:, 0] = END_TARGETS[0]
state.params["end_target_1"].value[:, 0] = END_TARGETS[1]
state.params["end_target_2"].value[:, 0] = END_TARGETS[2]
state.params["end_target_3"].value[:, 0] = END_TARGETS[3]
state.params["end_target_4"].value[:, 0] = END_TARGETS[4]
state.params["end_target_5"].value[:, 0] = END_TARGETS[5]
state.params["end_target_6"].value[:, 0] = END_TARGETS[6]
state.params["end_target_7"].value[:, 0] = END_TARGETS[7]
state.params["can0"].pose[:, 0] = can0_pose
state.params["can0_init_target"].value[:, 0] = can0_pose
state.params["can1"].pose[:, 0] = can1_pose
state.params["can1_init_target"].value[:, 0] = can1_pose
goal = "(and (Near can0 end_target_3) (Near can1 end_target_4))"

# Call p_mod_abs to find a plan given the above described problem.
plan, descr = p_mod_abs(
    hl_solver, solver, domain, problem, goal=goal, debug=True, n_resamples=10
)

# Visualize the resulting plan as an animation:
# pbv.draw_plan(plan) # This led to extremely strange and unintelligeble output
pbv.animate_plan(plan)

print("\n\n\n\n")

if plan is None or type(plan) is str:
    print("PLANNING FAILED")
elif len(plan.get_failed_preds()):
    print(f"PLAN FINISHED WITH FAILED PREDICATES: {plan.get_failed_preds()}")
else:
    print('PLAN FINISHED SUCCESSFULLY')

import ipdb; ipdb.set_trace()

