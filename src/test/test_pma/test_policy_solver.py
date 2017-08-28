import ipdb;
import numpy as np
import unittest, time, main
from pma import hl_solver, robot_ll_solver, policy_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes import baxter_sampling
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

class TestPolicySolver(unittest.TestCase):
    def test_pi2_learning(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/center_grasp.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
            '0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE',
        ]


        plan = hls.get_plan(plan_str, domain, problem)
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        # def callback(): return viewer
        # viewer.draw_plan_ts(plan, 0)
        solver = policy_solver.BaxterPolicySolver()
        policy = solver.train_pi2_policy(plan, n_samples=12, iterations=100, num_conditions=1)
        sample = solver._sample_policy(policy, solver.agent, plan, (0,29), solver.dummy_hyperparams, use_noise=False)
        traj = solver._sample_to_traj(sample)
        plan.params['baxter'].lArmPose[:,1:] = traj[:7]
        plan.params['baxter'].rArmPose[:,1:] = traj[7:]
        print 'Target Left:', plan.params['robot_end_pose'].lArmPose[:,-1]
        print 'Actual Left:', plan.params['baxter'].lArmPose[:,-1]
        print 'Target Right:', plan.params['robot_end_pose'].rArmPose[:,-1]
        print 'Actual Right:', plan.params['baxter'].rArmPose[:,-1]
        self.assertTrue(not len(plan.get_failed_preds(tol=1e-3)))
        ipdb.set_trace()

    def test_pi2_grasp_learning(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/basket_grasp_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
            '0: BASKET_GRASP BAXTER BASKET INIT_TARGET ROBOT_INIT_POSE BG_EE_LEFT BG_EE_RIGHT ROBOT_END_POSE',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback(a): return None # viewer
        # viewer.draw_plan_ts(plan, 0)
        solver = policy_solver.BaxterPolicySolver()
        policy = solver.train_pi2_grasp_policy(plan, n_samples=12, iterations=500, num_conditions=10, callback=callback)
        solver._reset_plan(plan, action='basket_grasp')
        sample = solver._sample_policy(policy, solver.agent, plan, (0,20), solver.dummy_hyperparams, use_noise=False, action='basket_grasp')
        traj = solver._sample_to_traj(sample, 'basket_grasp')
        plan.params['baxter'].lArmPose[:,1:] = traj[:7]
        plan.params['baxter'].rArmPose[:,1:] = traj[7:14]
        print 'Target Left:', plan.params['robot_end_pose'].lArmPose[:,-1]
        print 'Actual Left:', plan.params['baxter'].lArmPose[:,-1]
        print 'Target Right:', plan.params['robot_end_pose'].rArmPose[:,-1]
        print 'Actual Right:', plan.params['baxter'].rArmPose[:,-1]
        self.assertTrue(not len(plan.get_failed_preds(tol=1e-3)))
        ipdb.set_trace()
