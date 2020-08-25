import policy_hooks.namo.sorting_prob_10 as prob
prob.NUM_OBJS = 4
prob.NUM_TARGS = 4
prob.N_GRASPS = 4
prob.n_aux = 0
prob.END_TARGETS = prob.END_TARGETS[:4]
from pma.namo_solver import *
from pma.hl_solver import *
from pma.pr_graph import *

plans = prob.get_plans()
plan = plans[0][(0,0,0,2)]
domain = plan.domain
problem = plan.prob
state = problem.init_state
for p in list(plan.params.values()):
    if p.openrave_body is not None:
        p.openrave_body.set_pose([20,20])
pr2_pose = [2.6, -2.8]
can0_pose = [-5, -3.4]
can1_pose = [2, 0]
can2_pose = [4, -1]
can3_pose = [-4, -1]

plan.params['pr2'].pose[:,0] = pr2_pose
plan.params['robot_init_pose'].value[:,0] = pr2_pose
plan.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
plan.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
plan.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
plan.params['end_target_3'].value[:,0] = prob.END_TARGETS[3]
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
hl_solver = FFSolver(plan.d_c)
solver.backtrack_solve(plan)
abs_domain = hl_solver.abs_domain
import ipdb; ipdb.set_trace()

state.params['pr2'].pose[:,0] = pr2_pose
state.params['robot_init_pose'].value[:,0] = pr2_pose
state.params['end_target_0'].value[:,0] = prob.END_TARGETS[0]
state.params['end_target_1'].value[:,0] = prob.END_TARGETS[1]
state.params['end_target_2'].value[:,0] = prob.END_TARGETS[2]
state.params['end_target_3'].value[:,0] = prob.END_TARGETS[3]
state.params['can0'].pose[:,0] = can0_pose
state.params['can0_init_target'].value[:,0] = can0_pose

state.params['can1'].pose[:,0] = can1_pose
state.params['can1_init_target'].value[:,0] = can1_pose

state.params['can2'].pose[:,0] = can2_pose
state.params['can2_init_target'].value[:,0] = can2_pose

state.params['can3'].pose[:,0] = can3_pose
state.params['can3_init_target'].value[:,0] = can3_pose

goal = '(and (Near can0 end_target_1) (Near can1 end_target_0) (Near can2 end_target_3) (Near can3 end_target_2))'
plan, descr = p_mod_abs(hl_solver, solver, domain, problem, goal=goal, debug=True)
import ipdb; ipdb.set_trace()
