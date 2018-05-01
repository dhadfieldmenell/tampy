import numpy as np
import unittest, time, main
from pma import hl_solver, robot_ll_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes import baxter_sampling
from core.util_classes.baxter_predicates import BaxterCollides
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
# from ros_interface import action_execution
import core.util_classes.baxter_constants as const
from openravepy import matrixFromAxisAngle
import itertools
from collections import OrderedDict

def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    return domain, problem, params

def traj_retiming(plan, velocity):
    baxter = plan.params['baxter']
    rave_body = baxter.openrave_body
    body = rave_body.env_body
    lmanip = body.GetManipulator("left_arm")
    rmanip = body.GetManipulator("right_arm")
    left_ee_pose = []
    right_ee_pose = []
    for t in range(plan.horizon):
        rave_body.set_dof({
            'lArmPose': baxter.lArmPose[:, t],
            'lGripper': baxter.lGripper[:, t],
            'rArmPose': baxter.rArmPose[:, t],
            'rGripper': baxter.rGripper[:, t]
        })
        rave_body.set_pose([0,0,baxter.pose[:, t]])

        left_ee_pose.append(lmanip.GetTransform()[:3, 3])
        right_ee_pose.append(rmanip.GetTransform()[:3, 3])
    time = np.zeros(plan.horizon)
    # import ipdb; ipdb.set_trace()
    for t in range(plan.horizon-1):
        left_dist = np.linalg.norm(left_ee_pose[t+1] - left_ee_pose[t])
        right_dist = np.linalg.norm(right_ee_pose[t+1] - right_ee_pose[t])
        time_spend = max(left_dist, right_dist)/velocity[t]
        time[t+1] = time[t] + time_spend
    return time

class TestBasketDomain(unittest.TestCase):

    """
    CLOTH_GRASP action Isolation
    """
    def cloth_grasp_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_1 ROBOT_INIT_POSE CG_EE_1 ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, cloth = plan.params['baxter'], plan.params['cloth']
        print "solving cloth grasp isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:5] = 0.3
        velocites[5:16] = 0.1
        velocites[16:21] = 0.3

        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file cloth_grasp_isolated_plan.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("cloth_grasp_isolated_plan.hdf5", plan)
        self.assertTrue(result)

    """
    CLOTH_PUTDOWN action Isolation
    """
    def cloth_putdown_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_putdown_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 ROBOT_INIT_POSE CP_EE_1 ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, cloth = plan.params['baxter'], plan.params['cloth']
        print "solving cloth putdown isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:5] = 0.3
        velocites[5:16] = 0.1
        velocites[16:21] = 0.3
        baxter = plan.params['baxter']
        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file cloth_putdown_isolation_plan.hdf5..."

        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("cloth_putdown_isolation_plan.hdf5", plan)
        self.assertTrue(result)

    """
    MOVETO action Isolation
    """
    def move_to_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/move_to_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving move to isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=20)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        # baxter.time = traj_retiming(plan).reshape((1, plan.horizon))
        velocites = np.zeros((plan.horizon,))
        velocites[0:19] = 1
        baxter = plan.params['baxter']
        ee_times = traj_retiming(plan, velocites)
        baxter.time = ee_times.reshape((1, ee_times.shape[0]))

        print "Saving current plan to file move_to_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("move_to_isolation.hdf5", plan)
        # import ipdb; ipdb.set_trace()
        self.assertTrue(result)

    """
    MOVETO action Isolation
    """
    def move_into_washer_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/move_into_washer_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        plan.params['washer'].openrave_body.set_dof({'door': -np.pi/2})

        print "solving move to isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=20)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        # baxter.time = traj_retiming(plan).reshape((1, plan.horizon))
        velocites = np.zeros((plan.horizon,))
        velocites[0:19] = 1
        baxter = plan.params['baxter']
        ee_times = traj_retiming(plan, velocites)
        baxter.time = ee_times.reshape((1, ee_times.shape[0]))

        print "Saving current plan to file move_into_washer_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("move_into_washer_isolation.hdf5", plan)
        import ipdb; ipdb.set_trace()
        self.assertTrue(result)

    """
    BASKET_GRASP action Isolation
    """
    def basket_grasp_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/basket_grasp_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN',
         '1: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, basket = plan.params['baxter'], plan.params['basket']
        print "solving basket grasp isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback(a):
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.backtrack_solve(plan, callback = callback)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:5] = 0.3
        velocites[5:16] = 0.1
        velocites[16:21] = 0.3

        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file basket_grasp_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_grasp_isolation.hdf5", plan)
        import ipdb; ipdb.set_trace()
        self.assertTrue(result)

    """
    BASKET_PUTDOWN action Isolation
    """
    def basket_putdown_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/basket_putdown_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: BASKET_PUTDOWN BAXTER BASKET END_TARGET ROBOT_INIT_POSE BP_EE_LEFT BP_EE_RIGHT ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, basket = plan.params['baxter'], plan.params['basket']
        print "solving basket putdown isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:5] = 0.3
        velocites[5:16] = 0.1
        velocites[16:21] = 0.3

        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))
        # import ipdb; ipdb.set_trace()

        print "Saving current plan to file basket_putdown_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_putdown_isolation_plan.hdf5", plan)
        self.assertTrue(result)

    """
    MOVEHOLDING_CLOTH action Isolation
    """
    #TODO having trouble to solve
    def moveholding_cloth_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/moveholding_cloth_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE ROBOT_END_POSE CLOTH',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, cloth = plan.params['baxter'], plan.params['cloth']
        print "solving moveholding cloth isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        # import ipdb; ipdb.set_trace()
        # basket = plan.params['basket']
        # basket.openrave_body.set_pose(basket.pose[:, 0], basket.rotation[:, 0])
        # lArmPose = baxter.openrave_body.get_ik_from_pose(cloth.pose[:, 0],[0, np.pi/2, 0],"left_arm")[0]
        # baxter.openrave_body.set_dof({"lArmPose": lArmPose})
        # cloth.openrave_body.set_pose(cloth.pose[:, 0], cloth.rotation[:, 0])

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:20] = 0.3

        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file moveholding_cloth_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("moveholding_cloth_isolation.hdf5", plan)
        self.assertTrue(result)

    """
    MOVEHOLDING_BASKET action Isolation
    """
    def moveholding_basket_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/moveholding_basket_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVEHOLDING_BASKET BAXTER ROBOT_INIT_POSE ROBOT_END_POSE BASKET',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        baxter, basket = plan.params['baxter'], plan.params['basket']
        print "solving basket putdown isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        print "Saving current plan to file basket_putdown_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_putdown_isolation_plan.hdf5", plan)
        self.assertTrue(result)
        # import ipdb; ipdb.set_trace()

    """
    OPEN_DOOR action Isolation
    """
    def open_door_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/open_door_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: OPEN_DOOR BAXTER WASHER ROBOT_INIT_POSE OPEN_DOOR_EE CLOSE_DOOR_EE ROBOT_END_POSE WASHER_INIT_POSE WASHER_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving open door isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        # viewer.draw_plan_ts(plan, 0)
        # robot, washer = plan.params["baxter"], plan.params["washer"]
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # tool_link = washer_body.env_body.GetLink("washer_handle")
        # washer_body.set_pose([1, 1.1, 0.85], [np.pi/2+np.pi/6,0,0])

        # def set_washer_pose(door):
        #     washer_body.set_dof({'door': door})
        #     handle_pos = tool_link.GetTransform()[:3,3]
        #     arm_pose = robot_body.get_ik_from_pose(handle_pos, [np.pi/4, 0, 0], 'left_arm')
        #     robot_body.set_dof({'lArmPose': arm_pose[0]})
        #
        # set_washer_pose(-np.pi/18*0)
        # import ipdb; ipdb.set_trace()

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        print "Saving current plan to file open_door_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("open_door_isolation_plan.hdf5", plan)
        self.assertTrue(result)

        import ipdb; ipdb.set_trace()

    """
    PUSH_DOOR action Isolation
    """
    def push_door_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/push_door_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: PUSH_DOOR BAXTER WASHER ROBOT_INIT_POSE ROBOT_END_POSE WASHER_INIT_POSE WASHER_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving push door isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        # viewer.draw_plan_ts(plan, 0)
        # offset = np.array([-0.035,0.055,-0.1])
        # robot, washer = plan.params["baxter"], plan.params["washer"]
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # tool_link = washer_body.env_body.GetLink("washer_handle")
        # handle_pos = np.dot(tool_link.GetTransform(), np.r_[offset, 1])[:3]
        #
        # import ipdb; ipdb.set_trace()

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=20)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        # print "Saving current plan to file push_door_isolation.hdf5..."
        # serializer = PlanSerializer()
        # serializer.write_plan_to_hdf5("push_door_isolation.hdf5", plan)
        self.assertTrue(result)

        import ipdb; ipdb.set_trace()


    """
    CLOSE_DOOR action Isolation
    """
    def close_door_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/close_door_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
         '0: CLOSE_DOOR BAXTER WASHER ROBOT_INIT_POSE CLOSE_DOOR_EE OPEN_DOOR_EE ROBOT_END_POSE WASHER_INIT_POSE WASHER_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving open door isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        # viewer.draw_plan_ts(plan, 0)
        # robot, washer = plan.params["baxter"], plan.params["washer"]
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # tool_link = washer_body.env_body.GetLink("washer_handle")
        # handle_pos = tool_link.GetTransform()[:3,3]
        # def set_washer_pose(door):
        #     washer_body.set_dof({'door': door})
        #     handle_pos = tool_link.GetTransform()[:3,3]
        #     arm_pose = robot_body.get_ik_from_pose(handle_pos, [np.pi/4, 0, 0], 'left_arm')
        #     robot_body.set_dof({'lArmPose': arm_pose[0]})

        # set_washer_pose(-np.pi/18*9)
        # import ipdb; ipdb.set_trace()

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        print "Saving current plan to file open_door_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("open_door_isolation_plan.hdf5", plan)
        self.assertTrue(result)

        import ipdb; ipdb.set_trace()
    """
    Test Post Suggester in backtrack solve
    """
    def test_pose_suggester(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/laundry.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        offset = [0,0,const.EEREACHABLE_STEPS* const.APPROACH_DIST]
        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_1 CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 ROBOT_END_POSE',
        '5: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '6: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '7: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '8: MOVETO BAXTER BASKET_PUTDOWN_END ROBOT_END_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        robot, ee_pose = plan.params['baxter'], plan.params["cg_ee_1"]
        rave_body = robot.openrave_body
        l_manip = rave_body.env_body.GetManipulator("left_arm")
        r_manip = rave_body.env_body.GetManipulator("right_arm")

        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = lambda: None, n_resamples=0)

        robot_poses = solver.pose_suggester(plan, 0)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(l_manip.GetTransform()[:3,3], ee_pose.value[:, 0] + offset))

        ee_pose = plan.params["cp_ee_1"]
        robot_poses = solver.pose_suggester(plan, 2)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(l_manip.GetTransform()[:3,3], ee_pose.value[:, 0] + offset))

        ee_left, ee_right = plan.params["bg_ee_left"], plan.params["bg_ee_right"]
        robot_poses = solver.pose_suggester(plan, 4)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(l_manip.GetTransform()[:3,3], ee_left.value[:, 0] + offset))
        self.assertTrue(np.allclose(r_manip.GetTransform()[:3,3], ee_right.value[:, 0] + offset))

        ee_left, ee_right = plan.params["bp_ee_left"], plan.params["bp_ee_right"]
        robot_poses = solver.pose_suggester(plan, 6)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(l_manip.GetTransform()[:3,3], ee_left.value[:, 0] + offset))
        self.assertTrue(np.allclose(r_manip.GetTransform()[:3,3], ee_right.value[:, 0] + offset))

    def test_basket_position(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter_basket_grasp.domain',
                       '../domains/baxter_domain/baxter_probs/basket_move.prob')
        env = problem.env

        viewer = OpenRAVEViewer.create_viewer(env)
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        viewer.draw(objLst, 0, 0.7)

        robot = params['baxter']
        basket = params['basket']
        table = params['table']
        end_targ = params['end_target']
        baxter_body = OpenRAVEBody(env, 'baxter', robot.geom)
        basket_body = OpenRAVEBody(env, 'basket', basket.geom)
        offset = [0,const.BASKET_OFFSET,0]
        basket_pos = basket.pose.flatten()

        col_pred = BaxterCollides("collision_checker", [basket, table], ["Basket", "Obstacle"], env)

        max_offset = const.EEREACHABLE_STEPS*const.APPROACH_DIST
        ver_off = [0, 0,max_offset]
        #Grasping Pose
        left_arm_pose = baxter_body.get_ik_from_pose(basket_pos + offset, [0,np.pi/2,0], "left_arm")[0]
        right_arm_pose = baxter_body.get_ik_from_pose(basket_pos - offset, [0,np.pi/2,0], "right_arm")[0]
        baxter_body.set_dof({'lArmPose': left_arm_pose, "rArmPose": right_arm_pose})

        left_arm_pose = baxter_body.get_ik_from_pose(basket_pos + offset + ver_off, [0,np.pi/2,0], "left_arm")[0]
        right_arm_pose = baxter_body.get_ik_from_pose(basket_pos - offset + ver_off, [0,np.pi/2,0], "right_arm")[0]
        baxter_body.set_dof({'lArmPose': left_arm_pose, "rArmPose": right_arm_pose})

        self.assertFalse(col_pred.test(0))
        # Holding Pose
        left_arm_pose = baxter_body.get_ik_from_pose(np.array([0.75, 0.02, 1.005 + max_offset]) + offset, [0,np.pi/2,0], "left_arm")[0]
        right_arm_pose = baxter_body.get_ik_from_pose(np.array([0.75, 0.02, 1.005 + max_offset]) - offset, [0,np.pi/2,0], "right_arm")[0]
        baxter_body.set_dof({'lArmPose': left_arm_pose, "rArmPose": right_arm_pose})
        basket_body.set_pose([0.75, 0.02, 1.01 + 0.15], end_targ.rotation.flatten())

        #Putdown Pose
        basket_body.set_pose(end_targ.value.flatten(), end_targ.rotation.flatten())
        left_arm_pose = baxter_body.get_ik_from_pose(end_targ.value.flatten() + offset, [0,np.pi/2,0], "left_arm")[0]
        right_arm_pose = baxter_body.get_ik_from_pose(end_targ.value.flatten() - offset, [0,np.pi/2,0], "right_arm")[0]
        baxter_body.set_dof({'lArmPose': left_arm_pose, "rArmPose": right_arm_pose})

        left_arm_pose = baxter_body.get_ik_from_pose(end_targ.value.flatten() + offset + ver_off, [0,np.pi/2,0], "left_arm")[0]
        right_arm_pose = baxter_body.get_ik_from_pose(end_targ.value.flatten() - offset + ver_off, [0,np.pi/2,0], "right_arm")[0]
        baxter_body.set_dof({'lArmPose': left_arm_pose, "rArmPose": right_arm_pose})
        basket.pose = end_targ.value
        self.assertFalse(col_pred.test(0))

    def search_washer_position(self):


        robot, washer = ParamSetup.setup_baxter(), ParamSetup.setup_washer()
        env = ParamSetup.setup_env()
        viewer = OpenRAVEViewer.create_viewer(env)
        objLst = [robot, washer]
        viewer.draw(objLst, 0, 0.5)


        rave_body = OpenRAVEBody(env, robot.name, robot.geom)
        washer_body = OpenRAVEBody(env, washer.name, washer.geom)
        tool_link = washer_body.env_body.GetLink("washer_handle")
        offset = [-0.04, 0.07, -0.115]

        def varify_feasibility(robot_base_pose, pos, rot, time_steps = 20, arm='right', side = [0, np.pi/2, 0]):
            rave_body.set_pose([0,0,robot_base_pose])
            rave_body.set_dof({'lArmPose': [0,0,0,0,0,0,0], 'rArmPose': [0,0,0,0,0,0,0]})
            washer_body.set_pose(pos, rot)
            washer_body.set_dof({'door': 0})
            washer_trans, last_arm_pose = tool_link.GetTransform(), [0,0,0,0,0,0,0]
            targ_pos, targ_rot = washer_trans.dot(np.r_[offset, 1])[:3],  side
            reaching_rot = targ_rot
            ik_arm_poses = rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(arm))
            last_arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, last_arm_pose)
            if last_arm_pose is None:
                return False

            for door in np.linspace(0, -np.pi/2, time_steps):
                washer_body.set_dof({'door': door})
                washer_trans = tool_link.GetTransform()
                targ_pos, targ_rot = washer_trans.dot(np.r_[offset, 1])[:3],  side
                ik_arm_poses = rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(arm))

                arm_pose = baxter_sampling.get_is_mp_arm_pose(rave_body, ik_arm_poses, last_arm_pose, arm)
                if arm_pose is None:
                    return False
                rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})
                last_arm_pose = arm_pose

            # rot_mat = matrixFromAxisAngle([np.pi/2, 0, 0])
            # trans = washer_body.env_body.GetTransform().dot(rot_mat)
            # ik_arm_poses_left = rave_body.get_ik_solutions("left_arm", trans)
            # ik_arm_poses_right = rave_body.get_ik_solutions("right_arm", trans)
            # import ipdb; ipdb.set_trace()
            # if not len(ik_arm_poses_left) and not len(ik_arm_poses_right):
            #     return False
            # elif not len(ik_arm_poses_left):
                # rave_body.set_dof({'rArmPose': ik_arm_poses_right[0]})
            # elif not len(ik_arm_poses_right):
                # rave_body.set_dof({'lArmPose': ik_arm_poses_left[0]})

            return True

        print "calculating effective rotation"
        washer_pose = [0, 1.2, 0.7]
        effective_rot = []
        cashe_list = set()

        rot_angles = np.linspace(-np.pi, np.pi, 5)[:-1]
        rot_list = itertools.product(rot_angles,rot_angles,rot_angles)
        for rot in rot_list:
            washer_trans = OpenRAVEBody.transform_from_obj_pose(washer_pose, rot)
            local_dir = np.array([0,0,1])
            washer_dir = tuple(washer_trans[:3,:3].dot(local_dir))
            if washer_dir[2] < 0:
                continue

            if washer_dir in cashe_list:
                continue
            else:
                cashe_list.add(washer_dir)
                effective_rot.append(rot)
        print "{} effective rotation".format(len(effective_rot))

        print "calculating effective position"
        effective_pos = []
        for radius in np.linspace(0.7, 1.2, 6):
            for hight in np.linspace(0.1, 1.0, 10):
                for angle in np.linspace(-np.pi/4, np.pi/4, 5):
                    effective_pos.append([radius*np.cos(angle), radius*np.sin(angle), hight])

        print "{} effective poses".format(len(effective_pos))

        print "finding_grasping_position"
        grasp_poses = []
        cashe_list = set()
        midpoint = [0.87, 0, 0.8]
        washer_body.set_pose([2,2,2])
        for grasp_dir in itertools.product(rot_angles,rot_angles,rot_angles):
            arm_poses = rave_body.get_ik_from_pose(midpoint, grasp_dir, "left_arm")
            if not len(arm_poses):
                continue
            rave_body.set_dof({'lArmPose': arm_poses[0]})
            manip = rave_body.env_body.GetManipulator("left_arm")
            trans = manip.GetTransform()
            gripper_dir = tuple(np.dot(trans, np.array([1, 0, 0, 1]))[:3])
            if gripper_dir in cashe_list:
                continue
            else:
                cashe_list.add(gripper_dir)
                grasp_poses.append(grasp_dir)

        print "{} grasp_poses".format(len(grasp_poses))

        print "search space: {}".format(len(effective_pos) * len(effective_rot)* len(grasp_poses))
        print "finding good poses"
        good_washer_poses = []
        for pos in effective_pos:
            for rot in effective_rot:
                for grasp_dir in grasp_poses:
                    if varify_feasibility(0, pos, rot, time_steps=20, arm='right', side=grasp_dir):
                        good_washer_poses.append((pos, rot, 'right', grasp_dir))
                        print '{} good poses so far'.format(len(good_washer_poses))
                    if varify_feasibility(0, pos, rot, time_steps=20, arm='left', side=grasp_dir):
                        good_washer_poses.append((pos, rot, 'left', grasp_dir))
                        print '{} good poses so far'.format(len(good_washer_poses))
            print "saving good poses..."
            np.save("good_poses_fix_grasp_dir.npy", np.array(good_washer_poses))

    def test_washer_position(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/laundry.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        plan_str = ['0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE']

        plan = hls.get_plan(plan_str, domain, problem)
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        robot, washer = plan.params['baxter'], plan.params['washer']
        basket, cloth = plan.params['basket'], plan.params['cloth']
        table = plan.params['table']
        robot_body, washer_body = robot.openrave_body, washer.openrave_body
        washer_body.set_pose([0.85, 0.90, 0.85], [np.pi/2,0,0])
        washer_body.set_dof({'door': -np.pi/2})

        rot_mat = matrixFromAxisAngle([np.pi/2, 0, 0])
        trans = washer_body.env_body.GetTransform().dot(rot_mat)
        offset = np.eye(4)
        offset[:3,3] = [0,0,0]
        final_trans = np.dot(trans, offset)
        ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", final_trans)
        obj_pos = OpenRAVEBody.obj_pose_from_transform(trans)
        robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})

        import ipdb; ipdb.set_trace()

    def show_good_washer_poses(self):
        good_poses = np.load("final_poses2.npy")
        env = ParamSetup.setup_env()
        robot, washer = ParamSetup.setup_baxter(), ParamSetup.setup_washer()
        rave_body, washer_body = OpenRAVEBody(env, robot.name, robot.geom), OpenRAVEBody(env, washer.name, washer.geom)
        tool_link = washer_body.env_body.GetLink("washer_handle")

        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw([robot, washer], 0, 0.5)

        def get_feasible_arm_poses(robot_base_pose, pos, rot, time_steps = 20, arm='right', side = [0, np.pi/2, 0]):
            offset = [-0.04, 0.07, -0.115]
            rave_body.set_pose([0,0,robot_base_pose])
            rave_body.set_dof({'lArmPose': [0,0,0,0,0,0,0], 'rArmPose': [0,0,0,0,0,0,0]})
            washer_body.set_pose(pos, rot)
            washer_body.set_dof({'door': 0})
            washer_trans, last_arm_pose = tool_link.GetTransform(), [0,0,0,0,0,0,0]
            targ_pos, targ_rot = washer_trans.dot(np.r_[offset, 1])[:3],  side
            reaching_rot = targ_rot
            ik_arm_poses = rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(arm))
            last_arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, last_arm_pose)
            if last_arm_pose is None:
                return None
            feasible_arm_poses = []
            for door in np.linspace(0, -np.pi/2, time_steps):
                washer_body.set_dof({'door': door})
                washer_trans = tool_link.GetTransform()
                targ_pos, targ_rot = washer_trans.dot(np.r_[offset, 1])[:3],  side
                ik_arm_poses = rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(arm))

                arm_pose = baxter_sampling.get_is_mp_arm_pose(rave_body, ik_arm_poses, last_arm_pose, arm)
                if arm_pose is None:
                    return None
                feasible_arm_poses.append(arm_pose)
                rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})
                last_arm_pose = arm_pose

            rot_mat = matrixFromAxisAngle([np.pi/2, 0, 0])
            trans = washer_body.env_body.GetTransform().dot(rot_mat)
            offset = np.eye(4)
            offset[:3,3] = [0,0,0]
            ik_trans = np.dot(trans, offset)
            ik_arm_poses_left = rave_body.get_ik_solutions("left_arm", ik_trans)
            ik_arm_poses_right = rave_body.get_ik_solutions("right_arm", ik_trans)

            if not len(ik_arm_poses_left) and not len(ik_arm_poses_right):
                return None
            elif not len(ik_arm_poses_left):
                rave_body.set_dof({'rArmPose': ik_arm_poses_right[0]})
                feasible_arm_poses.append(ik_arm_poses_right)
            elif not len(ik_arm_poses_right):
                rave_body.set_dof({'lArmPose': ik_arm_poses_left[0]})
                feasible_arm_poses.append(ik_arm_poses_left)

            return feasible_arm_poses


        i = 0
        for pos, rot, arm, direction in good_poses:
            print i
            arm_poses = get_feasible_arm_poses(0, pos, rot, time_steps = 20, arm=arm, side=direction)
            import ipdb; ipdb.set_trace()
            i +=1

        import ipdb; ipdb.set_trace()

    def filter_good_washer_poses2(self):

        good_poses = np.load("final_poses2.npy")
        env = ParamSetup.setup_env()
        robot, washer = ParamSetup.setup_baxter(), ParamSetup.setup_washer()
        rave_body, washer_body = OpenRAVEBody(env, robot.name, robot.geom), OpenRAVEBody(env, washer.name, washer.geom)
        tool_link = washer_body.env_body.GetLink("washer_handle")
        offset = [-0.04, 0.07, -0.115]
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw([robot, washer], 0, 0.5)

        washer_body.set_dof({"door": -np.pi/2})
        def check_hand_clearance(local_vec, pos, rot, arm, direction):
            rave_body.set_dof({'lArmPose': [0,0,0,0,0,0,0], 'rArmPose': [0,0,0,0,0,0,0]})
            washer_body.set_pose(pos, rot)
            rot_mat = matrixFromAxisAngle([np.pi/2, 0, 0])
            trans = washer_body.env_body.GetTransform().dot(rot_mat)
            offset = np.eye(4)
            offset[:3,3] = local_vec
            ik_trans = np.dot(trans, offset)
            ik_arm_poses_left = rave_body.get_ik_solutions("left_arm", ik_trans)
            ik_arm_poses_right = rave_body.get_ik_solutions("right_arm", ik_trans)
            feasible_arm_poses = []
            if not len(ik_arm_poses_left) and not len(ik_arm_poses_right):
                return None
            if not len(ik_arm_poses_left):
                rave_body.set_dof({'rArmPose': ik_arm_poses_right[0]})
                feasible_arm_poses.append(ik_arm_poses_right)
            if not len(ik_arm_poses_right):
                rave_body.set_dof({'lArmPose': ik_arm_poses_left[0]})
                feasible_arm_poses.append(ik_arm_poses_left)

            return feasible_arm_poses
        final_good_poses = []

        for pos, rot, arm, direction in good_poses:
            good = True
            local_vec = [0, 0, -0.2]
            if not check_hand_clearance(local_vec, pos, rot, arm, direction):
                good = False
            if good:
                final_good_poses.append((pos, rot, arm, direction))
                print "found one"

        print len(final_good_poses)
        import ipdb; ipdb.set_trace()

        np.save("final_poses3.npy", np.array(final_good_poses))

    def test_laundry_position(self):
        domain, problem, params = load_environment('../domains/laundry_domain/laundry.domain',
                       '../domains/laundry_domain/laundry_probs/laundry.prob')
        env = problem.env

        viewer = OpenRAVEViewer.create_viewer(env)
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        viewer.draw(objLst, 0, 0.7)

        robot = params['baxter']
        basket = params['basket']
        table = params['table']
        washer = params['washer']
        end_targ = params['end_target']
        offset = [0,const.BASKET_OFFSET,0]
        basket_pos = basket.pose.flatten()
        left_targ = basket_pos + offset + [0,0,5*const.APPROACH_DIST]
        right_targ = basket_pos - offset + [0,0,5*const.APPROACH_DIST]
        grasp_rot = np.array([0,np.pi/2,0])

        robot_body = robot.openrave_body
        baskey_body = basket.openrave_body
        washer_body = washer.openrave_body
        #Grasp Begin Pose
        robot_body.set_pose([0,0, -np.pi/8])
        l_arm_pose = robot_body.get_ik_from_pose(left_targ, grasp_rot, "left_arm")[0]
        r_arm_pose = robot_body.get_ik_from_pose(right_targ, grasp_rot, "right_arm")[0]
        robot_body.set_dof({'lArmPose':l_arm_pose, 'rArmPose': r_arm_pose})

        # Putdown Pose
        baskey_body.set_pose([0.65, 0.323, 0.81], [np.pi/2, 0, np.pi/2])
        basket_pos = np.array([0.65, 0.323, 0.81])
        left_targ = basket_pos + offset + [0,0,5*const.APPROACH_DIST]
        right_targ = basket_pos - offset + [0,0,5*const.APPROACH_DIST]
        grasp_rot = np.array([0,np.pi/2,0])

        robot_body.set_pose([0,0, np.pi/8])
        l_arm_pose = robot_body.get_ik_from_pose(left_targ, grasp_rot, "left_arm")[0]
        r_arm_pose = robot_body.get_ik_from_pose(right_targ, grasp_rot, "right_arm")[0]
        robot_body.set_dof({'lArmPose':l_arm_pose, 'rArmPose': r_arm_pose})

        # import ipdb; ipdb.set_trace()

    def collision_debug_env(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("debug1.hdf5")
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        robot = plan.params['baxter']
        basket = plan.params['basket']
        viewer.draw_plan_ts(plan, 23)
        viewer.draw_cols_ts(plan, 23)

        pred = plan.find_pred("BaxterObstructs")[0]
        print (pred.get_type()+"_"+str(23), np.max(pred.get_expr(negated=True).expr.eval(pred.get_param_vector(23))))

        basket_pos = plan.params['basket'].pose[:, 23]
        offset = [0,0.312,0]
        shift = [0.1,0,0]
        robot_body = robot.openrave_body
        l_arm_pose = robot_body.get_ik_from_pose(basket_pos+shift+offset, [0,np.pi/2,0], "left_arm")[0]
        r_arm_pose = robot_body.get_ik_from_pose(basket_pos+[0,0,-0.2], [0,np.pi/2,0], "right_arm")[0]

        robot.lArmPose[:, 23] = l_arm_pose
        robot.rArmPose[:, 23] = r_arm_pose
        print (pred.get_type()+"_"+str(23), np.max(pred.get_expr(negated=True).expr.eval(pred.get_param_vector(23))))
        viewer.draw_plan_ts(plan, 23)
        viewer.draw_cols_ts(plan, 23)
        # import ipdb; ipdb.set_trace()

    def laundry_basket_mesh(self):
        from openravepy import KinBody, RaveCreateKinBody
        env = ParamSetup.setup_env()
        env.SetViewer('qtcoin')
        basket = ParamSetup.setup_basket()
        basket_mesh = env.ReadKinBodyXMLFile(basket.geom.shape)
        basket_mesh.SetName("basket")
        env.Add(basket_mesh)
        basket_body = OpenRAVEBody.create_basket_col(env)
        basket_body.SetName("basket_col")
        env.Add(basket_body)

        trans = OpenRAVEBody.transform_from_obj_pose([.2, .2, .2],[np.pi/4, np.pi/4, np.pi/4])

        basket_mesh.SetTransform(trans)
        basket_body.SetTransform(trans)

        # import ipdb; ipdb.set_trace()

    def find_cloth_position(self):
        domain, problem, params = load_environment('../domains/laundry_domain/laundry.domain',
                       '../domains/laundry_domain/laundry_probs/laundry.prob')
        env = problem.env

        viewer = OpenRAVEViewer.create_viewer(env)
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        viewer.draw(objLst, 0, 0.7)

        robot = params['baxter']
        rave_body = robot.openrave_body
        cloth = params['cloth']
        cloth_target = params['cloth_target_end_1']

        cloth.pose[:, 0] = cloth_target.value[:, 0]
        cloth.rotation[:, 0] = cloth_target.rotation[:, 0]
        cloth.openrave_body.set_pose(cloth.pose[:, 0], cloth.rotation[:, 0])

        ee_pos, ee_rot = cloth_target.value[:, 0] + np.array([0,0,const.APPROACH_DIST*const.EEREACHABLE_STEPS]), np.array([0, np.pi/2, 0])
        facing_pose = ee_pos[:2].dot([0,1])/np.linalg.norm(ee_pos[:2])

        arm_pose = rave_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")[0]
        rave_body.set_dof({'lArmPose': arm_pose})
        print arm_pose, facing_pose
        # import ipdb; ipdb.set_trace()

    def _test_backtrack_solve_action_isolation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/laundry.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_1 CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 BASKET_GRASP_BEGIN',
        '5: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '6: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '7: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '8: MOVETO BAXTER BASKET_PUTDOWN_END ROBOT_END_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback():
            return viewer

        solver = robot_ll_solver.RobotLLSolver()

        prev_action_values = {}
        for param in plan.params.values():
            prev_action_values[param] = {}
            for attr in param._free_attrs.keys():
                prev_action_values[param][attr] = getattr(param, attr).copy()

        for action_n in range(len(plan.actions)):
            print "optimizing action {}: {}".format(action_n, plan.actions[action_n])
            ts = plan.actions[action_n].active_timesteps
            solver._backtrack_solve(plan, anum=action_n, amax=action_n)
            for param in plan.params.values():
                if not param.is_symbol():
                    for attr in param._free_attrs.keys():
                        prev_action_values[param][attr][:,ts[0]+1:ts[1]+1] = getattr(param, attr)[:,ts[0]+1:ts[1]+1]
                else:
                    for attr in param._free_attrs.keys():
                        prev_action_values[param][attr][:,0] = getattr(param, attr)[:,0]

            serializer.write_plan_to_hdf5("test_backtrack_solve_{}.hdf5".format(action_n), plan)

            print "finished optimizing action {}".format(plan.actions[action_n])
            if plan.params['cloth'].pose.shape[1] < 176:
                import ipdb; ipdb.set_trace()

            for param in plan.params.values():
                if not param.is_symbol():
                    for attr in param._free_attrs.keys():
                        self.assertTrue(np.all(prev_action_values[param][attr][:,0:ts[0]+1] == getattr(param, attr)[:,0:ts[0]+1]))
                elif not np.all(np.isnan(prev_action_values[param]['value'][:,0])):
                    for attr in param._free_attrs.keys():
                        self.assertTrue(np.all(prev_action_values[param][attr][:,0] == getattr(param, attr)[:,0]))

    def test_full_plan(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/laundry.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_1 CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 BASKET_GRASP_BEGIN',
        '5: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '6: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '7: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '8: MOVETO BAXTER BASKET_PUTDOWN_END CLOTH_GRASP_BEGIN_2',
        '9: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_2 CLOTH_GRASP_BEGIN_2 CG_EE_2 CLOTH_GRASP_END_2',
        '10: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_2 CLOTH_PUTDOWN_BEGIN_2 CLOTH',
        '11: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_2 CLOTH_PUTDOWN_BEGIN_2 CP_EE_2 CLOTH_PUTDOWN_END_2',
        '12: MOVETO BAXTER CLOTH_PUTDOWN_END_2 ROBOT_END_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer

        velocites = np.ones((plan.horizon, ))*1
        slow_inds = np.array([range(19,39), range(58,78), range(97,117), range(136,156), range(175,195), range(214,234)]).flatten()
        velocites[slow_inds] = 0.6

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file cloth_manipulation_plan.hdf5..."

        serializer.write_plan_to_hdf5("cloth_manipulation_plan.hdf5", plan)
        self.assertTrue(result)


    def test_traj_smoother(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("washer_manipulation_plan.hdf5")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        # solver = robot_ll_solver.RobotLLSolver()
        # print "Test Trajectory Smoother"
        # result = solver.traj_smoother(plan, callback=None, n_resamples=10, active_ts=None, verbose=False)
        # self.assertTrue(result)
        import ipdb; ipdb.set_trace()


    def test_washer_manipulation(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/washer_manipulator_plan.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE OPEN_DOOR_BEGIN',
        '1: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN OPEN_DOOR_EE_1 OPEN_DOOR_EE_2 OPEN_DOOR_END WASHER_CLOSE_POSE WASHER_OPEN_POSE',
        '2: MOVETO BAXTER OPEN_DOOR_END  CLOSE_DOOR_BEGIN',
        '4: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN CLOSE_DOOR_EE_1 CLOSE_DOOR_EE_2 CLOSE_DOOR_END WASHER_OPEN_POSE WASHER_CLOSE_POSE',
        '5: MOVETO BAXTER CLOSE_DOOR_END ROBOT_END_POSE'
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        # robot, washer = plan.params['baxter'], plan.params['washer']
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # washer_body.set_dof({'door':-np.pi/2})
        # rot_mat = matrixFromAxisAngle([0, 0, 0])
        # trans = washer_body.env_body.GetTransform().dot(rot_mat)
        # ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", trans)
        # import ipdb; ipdb.set_trace()
        # robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})


        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        result = solver.traj_smoother(plan, callback = None)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file washer_manipulation_plan.hdf5..."
        serializer.write_plan_to_hdf5("washer_manipulation_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    def test_washer_manipulation_2(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE OPEN_DOOR_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 OPEN_DOOR_EE_APPROACH_0 OPEN_DOOR_EE_RETREAT_0 OPEN_DOOR_END_0 WASHER_CLOSE_POSE_0 WASHER_OPEN_POSE_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER OPEN_DOOR_END_0 ARM_BACK_1 \n'.format(act_num))
        act_num += 1
        # ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER OPEN_DOOR_END_0 ARM_BACK_2 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARM_BACK_2 ROBOT_INIT_POSE \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE CLOSE_DOOR_BEGIN_0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 CLOSE_DOOR_EE_APPROACH_0 CLOSE_DOOR_EE_RETREAT_0 CLOSE_DOOR_END_0 WASHER_OPEN_POSE_0 WASHER_CLOSE_POSE_0 \n'.format(act_num))
        # act_num += 1
        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['washer'].door[0, 0] = 0 # -np.pi/2
        plan.params['baxter'].pose[0,0] = 8./9. * np.pi
        plan.params['basket'].pose[:,:] = [[1], [1], [1]]
        plan.params['robot_init_pose'].value[0,0] = plan.params['baxter'].pose[0,0]
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1
        import ipdb; ipdb.set_trace()

        # robot, washer = plan.params['baxter'], plan.params['washer']
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # washer_body.set_dof({'door':-np.pi/2})
        # rot_mat = matrixFromAxisAngle([0, 0, 0])
        # trans = washer_body.env_body.GetTransform().dot(rot_mat)
        # ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", trans)
        # import ipdb; ipdb.set_trace()
        # robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})


        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        # result = solver.traj_smoother(plan, callback = None)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file washer_manipulation_plan.hdf5..."
        # serializer.write_plan_to_hdf5("washer_manipulation_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()


    def test_washer_manipulation_3(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE OPEN_DOOR_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 OPEN_DOOR_EE_APPROACH_0 OPEN_DOOR_EE_RETREAT_0 OPEN_DOOR_END_0 WASHER_CLOSE_POSE_0 WASHER_PUSH_POSE_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER OPEN_DOOR_END_0 OPEN_DOOR_BEGIN_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: PUSH_DOOR_OPEN BAXTER WASHER OPEN_DOOR_BEGIN_1  OPEN_DOOR_END_1 WASHER_PUSH_POSE_0 WASHER_OPEN_POSE_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER OPEN_DOOR_END_1 CLOSE_DOOR_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: PUSH_DOOR_CLOSE BAXTER WASHER CLOSE_DOOR_BEGIN_0 CLOSE_DOOR_END_0 WASHER_OPEN_POSE_0 WASHER_CLOSE_POSE_0 \n'.format(act_num))
        act_num += 1
        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['washer'].door[0, 0] = 0
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        # robot, washer = plan.params['baxter'], plan.params['washer']
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # washer_body.set_dof({'door':-np.pi/2})
        # rot_mat = matrixFromAxisAngle([0, 0, 0])
        # trans = washer_body.env_body.GetTransform().dot(rot_mat)
        # ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", trans)
        # import ipdb; ipdb.set_trace()
        # robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})


        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        result = solver.traj_smoother(plan, callback = None)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file washer_manipulation_plan.hdf5..."
        # serializer.write_plan_to_hdf5("washer_manipulation_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()


    def put_into_washer(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/put_into_washer.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE OPEN_DOOR_BEGIN',
        '1: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN OPEN_DOOR_EE_1 OPEN_DOOR_EE_2 OPEN_DOOR_END WASHER_INIT_POSE WASHER_END_POSE',
        '2: MOVETO BAXTER OPEN_DOOR_END CLOTH_GRASP_BEGIN_2',
        '3: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_2 CLOTH_GRASP_BEGIN_2 CG_EE_2 CLOTH_GRASP_END_2',
        '4: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_2 CLOTH_PUTDOWN_BEGIN_2 CLOTH',
        '5: PUT_INTO_WASHER BAXTER WASHER WASHER_END_POSE CLOTH CLOTH_TARGET_END_2 CLOTH_PUTDOWN_BEGIN_2 CP_EE_2 CLOTH_PUTDOWN_END_2',
        '6: MOVETO BAXTER CLOTH_PUTDOWN_END_2 CLOSE_DOOR_BEGIN',
        '7: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN CLOSE_DOOR_EE_1 CLOSE_DOOR_EE_2 CLOSE_DOOR_END WASHER_END_POSE WASHER_INIT_POSE',
        '8: MOVETO BAXTER CLOSE_DOOR_END ROBOT_END_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        if result:
            result = solver.traj_smoother(plan, callback = None)
        else: 
            import ipdb; ipdb.set_trace()
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file put_into_washer.hdf5..."
        serializer.write_plan_to_hdf5("put_into_washer.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    def put_into_washer_2(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        '1: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0',
        '3: MOVEHOLDING_CLOTH BAXTER LOAD_WASHER_INTERMEDIATE_POSE CLOTH_PUTDOWN_BEGIN_0 CLOTH0',
        '4: PUT_INTO_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        ]

        act_num = 5
        plan_str.append('{0}: GRAB_CORNER_LEFT BAXTER PUT_INTO_WASHER_EE_1 PUT_INTO_WASHER_EE_2 CLOTH_PUTDOWN_END_0 GRAB_EE_1 CLOTH_PUTDOWN_BEGIN_1 \n'.format(act_num))
        act_num += 1
        plan_str.append('{0}: MOVETO_EE_POS_LEFT BAXTER PUT_INTO_WASHER_EE_3 CLOTH_PUTDOWN_BEGIN_1 CLOTH_PUTDOWN_END_1 \n'.format(act_num))
        act_num += 1
        plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_1 PUT_INTO_WASHER_BEGIN \n'.format(act_num))
        act_num += 1
        plan_str.append('{0}: MOVETO BAXTER PUT_INTO_WASHER_BEGIN LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        act_num += 1

        plan = hls.get_plan(plan_str, domain, problem)
        plan.params['cloth0'].pose[:,:30] = [[-0.65], [0.15], [0.68]]
        plan.params['cloth_target_begin_0'].value[:,0] = plan.params['cloth0'].pose[:,0]
        plan.params['basket'].pose[:,:] = plan.params['basket_near_target'].value[:,:]
        plan.params['basket'].rotation[:,:] = plan.params['basket_near_target'].rotation[:,:]

        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        import ipdb; ipdb.set_trace()

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        import ipdb; ipdb.set_trace()
        if result:
            result = solver.traj_smoother(plan, callback = None)
        else: 
            import ipdb; ipdb.set_trace()
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file put_into_washer.hdf5..."
        serializer.write_plan_to_hdf5("put_into_washer.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    def take_out_of_washer(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE PUT_INTO_WASHER_BEGIN \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER PUT_INTO_WASHER_BEGIN INTERMEDIATE_UNLOAD \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD UNLOAD_WASHER_3 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER UNLOAD_WASHER_3 INTERMEDIATE_UNLOAD CLOTH0'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER INTERMEDIATE_UNLOAD PUT_INTO_WASHER_BEGIN CLOTH0'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER PUT_INTO_WASHER_BEGIN LOAD_WASHER_INTERMEDIATE_POSE CLOTH0'.format(act_num))

        # ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE PUT_INTO_WASHER_BEGIN \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_BEGIN_0 PUT_INTO_WASHER_BEGIN CP_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_WASHER_INTERMEDIATE_POSE CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_1 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE WASHER_SCAN_POSE \n'.format(act_num))
        # act_num += 1

        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['washer'].door[0, 0] = -np.pi/2
        plan.params['baxter'].pose[:,:] = 8*np.pi/9
        plan.params['robot_init_pose'].value[:,:] = 8*np.pi/9
        plan.params['basket'].pose[:,0] = plan.params['basket_near_target'].value[:,0]
        plan.params['basket'].rotation[:,0] = plan.params['basket_near_target'].rotation[:,0]
        plan.params['cloth0'].pose[:, 0] = plan.params['washer'].pose[:,0] + np.array([0.1, 0, -.14]) # self.state.washer_cloth_poses[0]
        plan.params['cloth_target_begin_0'].value[:,0] = plan.params['cloth0'].pose[:,0]
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        import ipdb; ipdb.set_trace()

        # robot, washer = plan.params['baxter'], plan.params['washer']
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # washer_body.set_dof({'door':-np.pi/2})
        # rot_mat = matrixFromAxisAngle([0, 0, 0])
        # trans = washer_body.env_body.GetTransform().dot(rot_mat)
        # ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", trans)
        # import ipdb; ipdb.set_trace()
        # robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})


        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        result = solver.traj_smoother(plan, callback = None)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file washer_manipulation_plan.hdf5..."
        # serializer.write_plan_to_hdf5("washer_manipulation_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()
    
    def grab_cloth_from_handle_region_1(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CLOTH_GRASP_FROM_HANDLE BAXTER CLOTH0 BASKET BASKET_NEAR_TARGET CLOTH_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_INIT_POSE \n'.format(act_num))

        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['washer'].door[0, 0] = -np.pi/2
        plan.params['cloth0'].pose[:, 0] = 0
        plan.params['baxter'].pose[:,:] = (2*np.pi/9)
        plan.params['robot_init_pose'].value[:,:] = (2*np.pi/9)
        plan.params['basket'].pose[:,0] = plan.params['basket_near_target'].value[:,0]
        plan.params['basket'].rotation[:,0] = plan.params['basket_near_target'].rotation[:,0]
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        # robot, washer = plan.params['baxter'], plan.params['washer']
        # robot_body, washer_body = robot.openrave_body, washer.openrave_body
        # washer_body.set_dof({'door':-np.pi/2})
        # rot_mat = matrixFromAxisAngle([0, 0, 0])
        # trans = washer_body.env_body.GetTransform().dot(rot_mat)
        # ik_arm_poses_left = robot_body.get_ik_solutions("left_arm", trans)
        # import ipdb; ipdb.set_trace()
        # robot_body.set_dof({'lArmPose': ik_arm_poses_left[0]})


        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        # result = solver.traj_smoother(plan, callback = None)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    
    def test_both_end_cloth_grasp(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_END_CLOTH_GRASP BAXTER CLOTH_LONG_EDGE CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CG_EE_1 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_LONG_EDGE CLOTH_FOLD_AIR_TARGET_1 CLOTH_GRASP_END_0 CLOTH_GRASP_END_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_LONG_EDGE CLOTH_FOLD_TABLE_TARGET_1 CLOTH_GRASP_END_1 CLOTH_GRASP_END_2 \n'.format(act_num))
        act_num += 1

        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['cloth_long_edge'].pose[:,0] = [0.7, 0, 0.655]
        plan.params['cloth_long_edge'].rotation[:,0] = [np.pi/8, 0, -np.pi/2]
        plan.params['cloth_target_begin_0'].value[:,0] = [0.7, 0, 0.655]
        plan.params['cloth_target_begin_0'].rotation[:,0] = [np.pi/8, 0, -np.pi/2]
        plan.params['cloth_target_begin_0']._free_attrs['value'][:] = 0
        plan.params['cloth_target_begin_0']._free_attrs['rotation'][:] = 0
        plan.params['baxter'].pose[:,:] = (2*np.pi/180)
        plan.params['basket'].pose[:,:] = 2
        plan.params['robot_init_pose'].value[:,:] = (2*np.pi/180)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()


    def test_both_end_cloth_grasp_2(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_END_CLOTH_GRASP BAXTER CLOTH_SHORT_EDGE CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CG_EE_1 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_AIR_TARGET_2 CLOTH_GRASP_END_0 CLOTH_GRASP_END_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_TABLE_TARGET_2 CLOTH_GRASP_END_1 CLOTH_GRASP_END_2 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_TABLE_TARGET_3 CLOTH_GRASP_END_2 CLOTH_GRASP_END_3 \n'.format(act_num))
        act_num += 1

        plan = hls.get_plan(ll_plan_str, domain, problem)
        plan.params['cloth_short_edge'].pose[:,0] = [0.65, 0.25, 0.63]
        plan.params['cloth_short_edge'].rotation[:,0] = [np.pi/4, 0, -np.pi/2]
        plan.params['cloth_target_begin_0'].value[:,0] = [0.65, 0.25, 0.63]
        plan.params['cloth_target_begin_0'].rotation = np.array([[np.pi/4], [0], [-np.pi/2]])
        plan.params['cloth_target_begin_0']._free_attrs['value'][:] = 0
        plan.params['cloth_target_begin_0']._free_attrs['rotation'][:] = 0
        plan.params['baxter'].pose[:,:] = (2*np.pi/180)
        plan.params['basket'].pose[:,:] = 2
        plan.params['robot_init_pose'].value[:,:] = (2*np.pi/180)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()

        print "Planning finished within {}s.".format(end - start)
        print "Planning succeeded: {}".format(result)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()


    def move_basket(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN_0',
        '1: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0',
        '2: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_1_POSE_0 REGION1',
        '3: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_1_POSE_0 BASKET_PUTDOWN_BEGIN_0 BASKET',
        '3: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        plan.params['basket'].pose[:,:35] = [[0.55], [0.], [0.875]]
        plan.params['basket_far_target'].value[:,:] = [[0.55], [0.], [0.875]]
        plan.params['basket_near_target'].value[:,:] = [[-0.6], [0.15], [0.875]]
        plan.params['basket'].rotation[:,:35] = [[np.pi/2], [0], [np.pi/2]]
        # plan.params['basket_near_target'].rotation[:,:] = [[-np.pi/2], [0], [np.pi/2]]
        plan.params['robot_init_pose'].lArmPose[:,0] = [0, -0.75, 0, 0, 0, 0, 0]
        plan.params['robot_init_pose'].rArmPose[:,0] = [0, -0.75, 0, 0, 0, 0, 0]
        plan.params['robot_init_pose'].value[0,0] = 0
        plan.params['baxter'].lArmPose[:,0] = [0, -0.75, 0, 0, 0, 0, 0]
        plan.params['baxter'].rArmPose[:,0] = [0, -0.75, 0, 0, 0, 0, 0]
        plan.params['baxter'].pose[0,0] = 0
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        import ipdb; ipdb.set_trace()

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        # import ipdb; ipdb.set_trace()
        # if result:
        #     result = solver.traj_smoother(plan, callback = None)
        # else: 
        #     import ipdb; ipdb.set_trace()
        end = time.time()
        import ipdb; ipdb.set_trace()

    def move_around_washer_cube(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/move_around_washer.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_INIT_TARGET CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVETO BAXTER CLOTH_GRASP_END_1 ROBOT_END_POSE',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()

        plan.params['robot_init_pose'].lArmPose[:,0] = [-0.2, -1.48486412, -0.57556553, 1.49087887, 0.04673458, 1.57856552, 0.01188452]
        plan.params['baxter'].lArmPose[:,0] = [-0.2, -1.48486412, -0.57556553, 1.49087887, 0.04673458, 1.57856552, 0.01188452]
        plan.params['cloth_grasp_begin_1'].lArmPose[:,0] = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
        plan.params['cloth_grasp_begin_1']._free_attrs['lArmPose'][:,:] = 0

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
        # viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        # import ipdb; ipdb.set_trace()

        plan.params['robot_init_pose'].lArmPose[:,0] = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
        plan.params['cloth_grasp_begin_1'].lArmPose[:,0] = [-0.4, -1.17058344, -0.60703633,  1.29084892,  0.22444338, 1.51769343, -0.19577044]
        plan.params['cloth_grasp_begin_1']._free_attrs['lArmPose'][:,:] = 0

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    def test_prototype2(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/prototype2.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_INIT_TARGET CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: PUT_INTO_BASKET BAXTER CLOTH BASKET CLOTH_TARGET_END_1 BASKET_INIT_TARGET CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 MONITOR_POSE',
        '5: MOVETO BAXTER MONITOR_POSE BASKET_GRASP_BEGIN',
        '6: BASKET_GRASP BAXTER BASKET BASKET_INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        '7: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        '8: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        '9: MOVETO BAXTER BASKET_PUTDOWN_END MONITOR_POSE',
        '10: MOVETO BAXTER MONITOR_POSE OPEN_DOOR_BEGIN',
        '11: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN OPEN_DOOR_EE_1 OPEN_DOOR_EE_2 OPEN_DOOR_END WASHER_INIT_POSE WASHER_END_POSE',
        '12: MOVETO BAXTER OPEN_DOOR_END  CLOTH_GRASP_BEGIN_2',
        '13: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_2 CLOTH_GRASP_BEGIN_2 CG_EE_2 CLOTH_GRASP_END_2',
        '14: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_2 CLOTH_PUTDOWN_BEGIN_2 CLOTH',
        '15: PUT_INTO_WASHER BAXTER WASHER WASHER_END_POSE CLOTH CLOTH_TARGET_END_2 CLOTH_PUTDOWN_BEGIN_2 CP_EE_2 CLOTH_PUTDOWN_END_2',
        '16: MOVETO BAXTER CLOTH_PUTDOWN_END_2 CLOSE_DOOR_BEGIN',
        '17: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN CLOSE_DOOR_EE_1 CLOSE_DOOR_EE_2 CLOSE_DOOR_END WASHER_END_POSE WASHER_INIT_POSE',
        '18: MOVETO BAXTER CLOSE_DOOR_END ROBOT_END_POSE'
        ]

        # plan_str = [
        # '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        # '1: CLOTH_GRASP BAXTER CLOTH CLOTH_INIT_TARGET CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        # '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        # '3: PUT_INTO_BASKET BAXTER CLOTH BASKET CLOTH_TARGET_END_1 BASKET_INIT_TARGET CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        # '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 MONITOR_POSE',
        # '5: MOVETO BAXTER MONITOR_POSE BASKET_GRASP_BEGIN',
        # '6: BASKET_GRASP BAXTER BASKET BASKET_INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        # '7: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        # '8: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        # '9: MOVETO BAXTER BASKET_PUTDOWN_END MONITOR_POSE',
        # '10: MOVETO BAXTER MONITOR_POSE OPEN_DOOR_BEGIN',
        # '11: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN OPEN_DOOR_EE_1 OPEN_DOOR_EE_2 OPEN_DOOR_END WASHER_INIT_POSE WASHER_PUSH_POSE',
        # '12: MOVETO BAXTER OPEN_DOOR_END PUSH_DOOR_BEGIN',
        # '13: PUSH_DOOR BAXTER WASHER PUSH_DOOR_BEGIN PUSH_DOOR_END WASHER_PUSH_POSE WASHER_END_POSE',
        # '14: MOVETO BAXTER PUSH_DOOR_END CLOTH_GRASP_BEGIN_2',
        # '15: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_2 CLOTH_GRASP_BEGIN_2 CG_EE_2 CLOTH_GRASP_END_2',
        # '16: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_2 CLOTH_PUTDOWN_BEGIN_2 CLOTH',
        # '17: PUT_INTO_WASHER BAXTER WASHER WASHER_END_POSE CLOTH CLOTH_TARGET_END_2 CLOTH_PUTDOWN_BEGIN_2 CP_EE_2 CLOTH_PUTDOWN_END_2',
        # '18: MOVETO BAXTER CLOTH_PUTDOWN_END_2 CLOSE_DOOR_BEGIN',
        # '19: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN CLOSE_DOOR_EE_1 CLOSE_DOOR_EE_2 CLOSE_DOOR_END WASHER_END_POSE WASHER_INIT_POSE',
        # '20: MOVETO BAXTER CLOSE_DOOR_END ROBOT_END_POSE'
        # ]

        # plan = hls.get_plan(plan_str, domain, problem)
        # print "solving basket domain problem..."
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('vel_acc_test_plan.hdf5')
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        serializer = PlanSerializer()
        def callback(a): return viewer
        velocites = np.ones((plan.horizon, ))*1

        viewer.draw_plan_ts(plan, 0)
        import ipdb; ipdb.set_trace()

        solver = robot_ll_solver.RobotLLSolver()
        start = time.time()
        result = solver.backtrack_solve(plan, callback = callback, verbose=False)
        if result:
            print "Saving current non-smooth plan to file prototype_plan.hdf5..."
            serializer.write_plan_to_hdf5("prototype_plan.hdf5", plan)
            result = solver.traj_smoother(plan, callback = None)
        else:
            import ipdb; ipdb.set_trace()
        end = time.time()
        # pd = PlanDeserializer()
        # plan = pd.read_from_hdf5('prototype_plan.hdf5')
        # import ipdb; ipdb.set_trace()
        print "Planning finished within {}s.".format(end - start)
        ee_time = traj_retiming(plan, velocites)
        plan.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file prototype_plan.hdf5..."
        serializer.write_plan_to_hdf5("prototype_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()

    def generate_one_cloth_prob(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_policy_1.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        '1: CLOTH_GRASP BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
        '3: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 INIT_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_INIT_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        ps = PlanSerializer()
        ps.write_plan_to_hdf5('one_cloth_grasp', plan)


    def test_monitor_update(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/move_to_isolation.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)

        print "solving move to isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=20)
        end = time.time()

        cloth = plan.params['cloth']
        basket = plan.params['basket']
        update_values = []

        attr_inds, res = OrderedDict(), OrderedDict()
        baxter_sampling.add_to_attr_inds_and_res(5, attr_inds, res, cloth, [('pose', np.array([0,1,0])), ('rotation', np.array([0,0,0]))])
        update_values.append((res, attr_inds))

        attr_inds, res = OrderedDict(), OrderedDict()
        baxter_sampling.add_to_attr_inds_and_res(5, attr_inds, res, basket, [('pose', np.array([1,1,1])), ('rotation', np.array([0,np.pi/2,0]))])
        update_values.append((res, attr_inds))
        result = solver.monitor_update(plan, update_values)

        self.assertTrue(result)


    def monitor_update_real_context(self):
        from ros_interface.environment_monitor import EnvironmentMonitor
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("env_mon.hdf5")
        solver = robot_ll_solver.RobotLLSolver()
        update_values = []
        cloth = plan.params['cloth']
        basket = plan.params['basket']
        cloth_pos = cloth.pose[:, 5] + [0,0,0.01]
        basket_pos = basket.pose[:, 5] + [-0.01,0,0]
        np.save('basket_pose.npy', basket_pos)
        np.save('cloth_pose.npy', cloth_pos)
        # attr_inds, res = OrderedDict(), OrderedDict()
        # baxter_sampling.add_to_attr_inds_and_res(5, attr_inds, res, cloth, [('pose', cloth_pos)])
        # update_values.append((res, attr_inds))
        #
        # attr_inds, res = OrderedDict(), OrderedDict()
        # baxter_sampling.add_to_attr_inds_and_res(5, attr_inds, res, basket, [('pose', basket_pos)])
        env_mon = EnvironmentMonitor()
        env_mon.basket_pose = basket_pos
        env_mon.cloth_pose = cloth_pos

        update_values = env_mon.update_plan(plan, 5, False)
        result = solver.monitor_update(plan, update_values)

        # self.assertTrue(result)



if __name__ == "__main__":
    unittest.main()
