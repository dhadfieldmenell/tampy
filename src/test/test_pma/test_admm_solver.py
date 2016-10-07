import unittest
import time, main
from pma import hl_solver
from pma.admm_solver import ADMMHelper, NAMOADMMSolver
from core.parsing import parse_domain_config
from core.parsing import parse_problem_config
from core.util_classes.viewer import OpenRAVEViewer

class TestADMMSolver(unittest.TestCase):
    def setUp(self):

        domain_fname = '../domains/namo_domain/namo.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        def get_plan(p_fname, plan_str=None):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            if plan_str is not None:
                return hls.get_plan(plan_str, domain, problem)
            return hls.solve(abs_problem, domain, problem)

        self.move_no_obs = get_plan('../domains/namo_domain/namo_probs/move_no_obs.prob')
        self.move_w_obs = get_plan('../domains/namo_domain/namo_probs/move_w_obs.prob')
        self.move_grasp = get_plan('../domains/namo_domain/namo_probs/move_grasp.prob')
        self.move_grasp_moveholding = get_plan('../domains/namo_domain/namo_probs/moveholding.prob')
        self.place = get_plan('../domains/namo_domain/namo_probs/place.prob')
        self.putaway = get_plan('../domains/namo_domain/namo_probs/putaway.prob')
        self.putaway3 = get_plan('../domains/namo_domain/namo_probs/putaway3.prob')

        self.putaway2 = get_plan('../domains/namo_domain/namo_probs/putaway2.prob',
                ['0: MOVETO PR2 ROBOT_INIT_POSE PDP_TARGET2',
                '1: GRASP PR2 CAN0 TARGET0 PDP_TARGET2 PDP_TARGET0 GRASP0',
                '2: MOVETOHOLDING PR2 PDP_TARGET0 PDP_TARGET2 CAN0 GRASP0',
                '3: PUTDOWN PR2 CAN0 TARGET2 PDP_TARGET2 ROBOT_END_POSE GRASP0'])

    def test_compute_shared_timesteps(self):
        nas = NAMOADMMSolver()
        plan = self.move_no_obs
        shared_timesteps, unshared_ranges = nas._compute_shared_timesteps(plan)
        self.assertTrue(len(shared_timesteps) == 0)
        self.assertTrue(len(unshared_ranges) == 1)
        action = plan.actions[0]
        self.assertTrue((0, plan.horizon-1) in unshared_ranges)

        plan = self.move_grasp_moveholding
        shared_timesteps, unshared_ranges = nas._compute_shared_timesteps(plan)
        self.assertTrue(len(plan.actions) == 3)
        self.assertTrue(len(shared_timesteps) == 2)
        self.assertTrue(len(unshared_ranges) == 3)
        self.assertTrue(plan.actions[0].active_timesteps[1] in shared_timesteps)
        self.assertTrue(plan.actions[1].active_timesteps[1] in shared_timesteps)

        for i, action in enumerate(plan.actions):
            start, end = action.active_timesteps
            if i == 0:
                self.assertTrue((start, end-1) in unshared_ranges)
            elif i == len(plan.actions) - 1:
                self.assertTrue((start+1, end) in unshared_ranges)
            else:
                self.assertTrue((start+1, end-1) in unshared_ranges)

    def test_classify_variables(self):
        nas = NAMOADMMSolver()

        # test that all variables are nonconsensus for self.move_no_obs
        plan = self.move_no_obs
        consensus_dict, nonconsensus_dict = nas._classify_variables(plan)
        param = plan.params['robot_init_pose']
        self.assertTrue(param in consensus_dict)
        self.assertTrue(param not in nonconsensus_dict)
        self.assertTrue(0 in consensus_dict[param])
        param = plan.params['robot_end_pose']
        self.assertTrue(param in consensus_dict)
        self.assertTrue(param not in nonconsensus_dict)
        self.assertTrue(0 in consensus_dict[param])
        param = plan.params['pr2']
        self.assertTrue(param not in consensus_dict)
        self.assertTrue(param in nonconsensus_dict)
        self.assertTrue((0,19) in nonconsensus_dict[param])

        # test that some variables are consensus for self.move_grasp_moveholding
        plan = self.move_grasp_moveholding
        consensus_dict, nonconsensus_dict = nas._classify_variables(plan)
        param = plan.params['pr2']
        self.assertTrue(param in consensus_dict)
        self.assertTrue(param in nonconsensus_dict)
        self.assertTrue(19 in consensus_dict[param])
        self.assertTrue(38 in consensus_dict[param])
        self.assertTrue((0,18) in nonconsensus_dict[param])
        self.assertTrue((20,37) in nonconsensus_dict[param])
        self.assertTrue((39,57) in nonconsensus_dict[param])
        param = plan.params['target1']
        self.assertTrue(param in consensus_dict)
        self.assertTrue(param not in nonconsensus_dict)
        self.assertTrue(0 in consensus_dict[param])
        param = plan.params['robot_end_pose']
        self.assertTrue(param in consensus_dict)
        self.assertTrue(param not in nonconsensus_dict)
        self.assertTrue(0 in consensus_dict[param])

    def test_move_no_obs(self):
        _test_plan(self, self.move_no_obs, method='ADMM', plot=True,
                   animate=True, verbose=True)

    def test_move_grasp(self):
        _test_plan(self, self.move_grasp, method='ADMM', plot=True,
                   animate=True, verbose=False)

def _test_plan(test_obj, plan, method='ADMM', plot=False, animate=False, verbose=False,
               early_converge=False):
    print "testing plan: {}".format(plan.actions)
    if not plot:
        callback = None
        viewer = None
    else:
        viewer = OpenRAVEViewer.create_viewer()
        if method=='ADMM':
            def callback_solv_and_plan(solver, plan):
                solver._update_ll_params()
                viewer.draw_plan(plan)
    admm_solver = NAMOADMMSolver()
    start = time.time()
    if method == 'ADMM':
        admm_solver.solve(plan, n_resamples=0, callback=callback_solv_and_plan,
                          verbose=verbose)
    print "Solve Took: {}".format(time.time() - start)
    fp = plan.get_failed_preds()
    _, _, t = plan.get_failed_pred()
    if animate:
        viewer = OpenRAVEViewer.create_viewer()
        viewer.animate_plan(plan)
        if t < plan.horizon:
            viewer.draw_plan_ts(plan, t)

    test_obj.assertTrue(plan.satisfied())
