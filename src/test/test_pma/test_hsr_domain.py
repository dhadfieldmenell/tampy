import numpy as np
import unittest, time, main
from pma import hl_solver, hsr_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.hsr_prob_gen import save_prob
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
# from ros_interface import action_execution
import core.util_classes.hsr_constants as const
from openravepy import matrixFromAxisAngle
import itertools
from collections import OrderedDict

class TestHSRDomain(unittest.TestCase):
    def test_move_obj(self):
        domain_fname = '../domains/hsr_domain/hsr.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)

        save_prob('../domains/hsr_domain/hsr_probs/hsr_prob.prob')
        p_c = main.parse_file_to_dict('../domains/hsr_domain/hsr_probs/hsr_prob.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO HSR ROBOT_INIT_POSE CAN_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CAN_GRASP HSR CAN0 CAN0_INIT_TARGET CAN_GRASP_BEGIN_0 CG_EE_0 CAN_GRASP_END_0\n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CAN HSR CAN_GRASP_END_0 CAN_PUTDOWN_BEGIN_0 CAN0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CAN_PUTDOWN HSR CAN0 CAN0_END_TARGET CAN_PUTDOWN_BEGIN_0 CP_EE_0 CAN_PUTDOWN_END_0 \n'.format(act_num))
        act_num += 1
        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['can0'].pose[:,0] = [1.55, 0, 0.44]
        # plan.params['can0'].pose[:,0] = [-0.5, 0.5, 0.]
        plan.params['can0_init_target'].value[:,0] = plan.params['can0'].pose[:,0]
        plan.params['can0_end_target'].value[:,0] = [1.55, 0.3, 0.44]
        # plan.params['cg_ee_0'].value[:,0] = plan.params['can0_init_target'].value[:,0]
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        def callback(a):
            return viewer
        solver = hsr_solver.HSRSolver()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False, n_resamples=2)
        import ipdb; ipdb.set_trace()
