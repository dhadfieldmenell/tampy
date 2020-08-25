from pma import hl_solver, ll_solver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.util_classes.viewer import OpenRAVEViewer

from main import parse_file_to_dict

import time



domain_fname = '../domains/namo_domain/namo.domain'
problem_fname = '../domains/namo_domain/namo_probs/swap_1234_0.prob'

def pick_place_str(start_idx, r_start, r_end, c_i, c_start_i, c_end_i):
    return ['{}: GRASP PR2 CAN{} TARGET{} {} PDP_TARGET{} GRASP{}'.format(start_idx, c_i, c_start_i, r_start, c_start_i, c_start_i),
            '{}: MOVETOHOLDING PR2 PDP_TARGET{} PDP_TARGET{} CAN{} GRASP{}'.format(start_idx+1, c_start_i, c_end_i, c_i, c_start_i),
            '{}: PUTDOWN PR2 CAN{} TARGET{} PDP_TARGET{} {} GRASP{}'.format(start_idx +2, c_i, c_end_i, c_end_i, r_end, c_start_i)]

PLAN_SWAP_STR = pick_place_str(0, 'ROBOT_INIT_POSE', 'PDP_TARGET7', 0, 0, 2)
PLAN_SWAP_STR.extend(pick_place_str(3, 'PDP_TARGET7', 'PDP_TARGET6', 1, 1, 3))
PLAN_SWAP_STR.extend(pick_place_str(6, 'PDP_TARGET6', 'PDP_TARGET5', 0, 2, 1))
PLAN_SWAP_STR.extend(pick_place_str(6, 'PDP_TARGET5', 'PDP_TARGET4', 1, 3, 0))


def _test_plan(plan, method='SQP', plot=False, animate=True, verbose=False,
               early_converge=False):
    print("testing plan: {}".format(plan.actions))
    if not plot:
        callback = None
        viewer = None
    else:
        viewer = OpenRAVEViewer.create_viewer()
        if method=='SQP':
            def callback():
                namo_solver._update_ll_params()
                # viewer.draw_plan_range(plan, range(57, 77)) # displays putdown action
                # viewer.draw_plan_range(plan, range(38, 77)) # displays moveholding and putdown action
                viewer.draw_plan(plan)
                # viewer.draw_cols(plan)
                time.sleep(0.3)
        elif method == 'Backtrack':
            def callback(a):
                namo_solver._update_ll_params()
                viewer.clear()
                viewer.draw_plan_range(plan, a.active_timesteps)
                time.sleep(0.3)
    namo_solver = ll_solver.NAMOSolver(early_converge=early_converge)
    start = time.time()
    if method == 'SQP':
        success = namo_solver.solve(plan, callback=callback, verbose=verbose)
    elif method == 'Backtrack':
        success = namo_solver.backtrack_solve(plan, callback=callback, verbose=verbose)
    print("Solve Took: {}\tSolution Found: {}".format(time.time() - start, success))

    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    if animate:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

def main():
    d_c = parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)

    p_c = parse_file_to_dict(problem_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

    plan = hls.get_plan(PLAN_SWAP_STR, domain, problem)
    _test_plan(plan, 'SQP', plot=False, animate=False, verbose=True, early_converge=True)
    # _test_plan(plan, 'Backtrack', plot=False, animate=True, verbose=False)

if __name__ == '__main__':
    main()
