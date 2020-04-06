import policy_hooks.namo.sorting_prob_7 as prob
from pma.namo_solver import *

plans = prob.get_plans()
plan = plans[0][(0,1,0,0)]
for p in plan.params.values():
    if p.openrave_body is not None:
        p.openrave_body.set_pose([20,20])

solver = NAMOSolver()
for c in range(1,4):
    plan.params['can{0}'.format(c)].pose[:,0] = [10+c, 10+c]
    plan.params['can{0}_init_target'.format(c)].value[:,0] = [10+c, 10+c]
plan.params['can0'].pose[:,0] = [-2.6, -3.6]
plan.params['can0_init_target'].value[:,0] = plan.params['can0'].pose[:,0]
plan.params['can0_end_target'].value[:,0] = plan.params['can0'].pose[:,0]

plan.params['can1'].pose[:,0] = [-0.2, -2.3]
plan.params['can1_init_target'].value[:,0] = plan.params['can1'].pose[:,0]


plan.params['can2'].pose[:,0] = [-20, 1.3]
plan.params['can2_init_target'].value[:,0] = plan.params['can2'].pose[:,0]


plan.params['can3'].pose[:,0] = [22, -1.1]
plan.params['can3_init_target'].value[:,0] = plan.params['can3'].pose[:,0]

plan.params['pr2'].pose[:,0] = [3.4, -3.5]
plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
plan.params['robot_end_pose'].value[:,0] = [-0.2, -2.901]
plan.params['obs0'].pose[:] = [[-3.5], [0]]
success = solver.backtrack_solve(plan, n_resamples=10)

import ipdb; ipdb.set_trace()

