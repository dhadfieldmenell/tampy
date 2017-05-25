import numpy as np
import unittest, time, main
from pma import hl_solver, robot_ll_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
from ros_interface import action_execution

class TestBasketDomain(unittest.TestCase):

    def test_basket_domain(self):
        domain_fname = '../domains/baxter_domain/baxter_basket_grasp.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading basket problem..."
        p_c = main.parse_file_to_dict('../domains/baxter_domain/baxter_probs/basket_env.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        plan_str = ['0: BASKET_GRASP BAXTER BASKET TARGET ROBOT_INIT_POSE EE_LEFT EE_RIGHT ROBOT_END_POSE',
                    '1: BASKET_PUTDOWN BAXTER BASKET TARGET ROBOT_INIT_POSE EE_LEFT EE_RIGHT ROBOT_END_POSE']
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def animate(delay = 0.5):
            viewer.animate_plan(plan, delay)
        def draw_ts(ts):
            viewer.draw_plan_ts(plan, ts)
        def draw_cols_ts(ts):
            viewer.draw_cols_ts(plan, ts)
        def callback():
            return viewer
        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, n_resamples=5)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        print plan.get_failed_preds()

        print "Saving current plan to file basket_plan.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_plan.hdf5", plan)
        """
            Uncomment to execution plan in baxter
        """
        # print "executing plan in Baxter..."
        # for act in plan.actions:
        #     action_execution.execute_action(act)


if __name__ == "__main__":
    unittest.main()
