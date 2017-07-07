import numpy as np
import unittest, time, main
from pma import hl_solver, robot_ll_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.baxter_predicates import BaxterCollides
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.param_setup import ParamSetup
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer
from ros_interface import action_execution
import core.util_classes.baxter_constants as const

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

    def test_basket_domain(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading basket problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/basket_move.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        plan_str = ['0: BASKET_GRASP BAXTER BASKET INIT_TARGET ROBOT_INIT_POSE GRASP_EE_LEFT GRASP_EE_RIGHT PICKUP_POSE', '1: BASKET_PUTDOWN BAXTER BASKET END_TARGET PICKUP_POSE PUTDOWN_EE_LEFT PUTDOWN_EE_RIGHT ROBOT_END_POSE']

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
        result = solver.solve(plan, callback = callback, n_resamples=5)
        end = time.time()

        baxter = plan.params['baxter']
        body = baxter.openrave_body.env_body
        lmanip = body.GetManipulator('left_arm')
        rmanip = body.GetManipulator('right_arm')
        def check(t, vel):
            viewer.draw_plan_ts(plan, t)
            left_t0 = lmanip.GetTransform()[:3,3]
            right_t0 = rmanip.GetTransform()[:3,3]
            viewer.draw_plan_ts(plan, t+1)
            left_t1 = lmanip.GetTransform()[:3,3]
            right_t1 = rmanip.GetTransform()[:3,3]
            left_spend = np.linalg.norm(left_t1 - left_t0) /vel
            right_spend = np.linalg.norm(right_t1 - right_t0) /vel
            print "{}:{}".format(left_spend, baxter.time[:, t+1] - baxter.time[:, t])
            print "{}:{}".format(right_spend, baxter.time[:, t+1] - baxter.time[:, t])

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)

        baxter.time = traj_retiming(plan).reshape((1, plan.horizon))
        print plan.get_failed_preds()
        print "Saving current plan to file basket_plan.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_plan.hdf5", plan)
        import ipdb; ipdb.set_trace()
        """
            Uncomment to execution plan in baxter
        """
        # print "executing plan in Baxter..."
        # for act in plan.actions:
        #     action_execution.execute_action(act)

    def test_laundry_domain(self):
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading laundry problem..."
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/laundry.prob')
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        # plan_str = [
        # '0: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN',
        # '1: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        # '2: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        # '3: BASKET_PUTDOWN BAXTER BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        # '4: MOVETO BAXTER BASKET_PUTDOWN_END ROBOT_END_POSE'
        # # '5: OPEN_DOOR BAXTER WASHER ROBOT_WASHER_BEGIN WASHER_EE ROBOT_WASHER_END WASHER_INIT_POSE WASHER_END_POSE',
        # # '6: MOVETO BAXTER ROBOT_WASHER_END ROBOT_END_POSE',
        # ]

        plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_1',
        '1: CLOTH_GRASP BAXTER CLOTH CLOTH_TARGET_BEGIN_1 CLOTH_GRASP_BEGIN_1 CG_EE_1 CLOTH_GRASP_END_1',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_1 CLOTH_PUTDOWN_BEGIN_1 CLOTH',
        '3: CLOTH_PUTDOWN BAXTER CLOTH CLOTH_TARGET_END_1 CLOTH_PUTDOWN_BEGIN_1 CP_EE_1 CLOTH_PUTDOWN_END_1',
        '4: MOVETO BAXTER CLOTH_PUTDOWN_END_1 ROBOT_END_POSE',
        # '5: BASKET_GRASP BAXTER BASKET INIT_TARGET BASKET_GRASP_BEGIN BG_EE_LEFT BG_EE_RIGHT BASKET_GRASP_END',
        # '6: MOVEHOLDING_BASKET BAXTER BASKET_GRASP_END BASKET_PUTDOWN_BEGIN BASKET',
        # '7: BASKET_PUTDOWN BASKET END_TARGET BASKET_PUTDOWN_BEGIN BP_EE_LEFT BP_EE_RIGHT BASKET_PUTDOWN_END',
        # '8: MOVETO BAXTER BASKET_PUTDOWN_END ROBOT_END_POSE'
        ]

        plan = hls.get_plan(plan_str, domain, problem)
        print "solving basket domain problem..."
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
        velocites[0:19] = 0.5
        baxter = plan.params['baxter']
        ee_times = traj_retiming(plan, velocites)
        baxter.time = ee_times.reshape((1, ee_times.shape[0]))

        print "Saving current plan to file move_to_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("move_to_isolation.hdf5", plan)

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
         '0: BASKET_GRASP BAXTER BASKET INIT_TARGET ROBOT_INIT_POSE BG_EE_LEFT BG_EE_RIGHT ROBOT_END_POSE',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, basket = plan.params['baxter'], plan.params['basket']
        print "solving basket grasp isolation problem..."
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

        print "Saving current plan to file basket_grasp_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_grasp_isolation.hdf5", plan)
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

        print "Saving current plan to file basket_putdown_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_putdown_isolation_plan.hdf5", plan)
        self.assertTrue(result)

    """
    MOVEHOLDING_CLOTH action Isolation
    """
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
        '2: MOVEHOLDING_BASKET BAXTER ROBOT_INIT_POSE ROBOT_END_POSE BASKET',
        ]
        plan = hls.get_plan(plan_str, domain, problem)
        baxter, basket = plan.params['baxter'], plan.params['basket']
        print "solving basket putdown isolation problem..."
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        def callback():
            return viewer

        import ipdb; ipdb.set_trace()
        # offset = np.array([0,0.317,0])
        # lArmPose = baxter.openrave_body.get_ik_from_pose(basket.pose[:, 0]  + offset,[0, np.pi/2, 0],"left_arm")[0]
        # rArmPose = baxter.openrave_body.get_ik_from_pose(basket.pose[:, 0] - offset,[0, np.pi/2, 0],"right_arm")[0]
        # baxter.openrave_body.set_dof({"lArmPose": lArmPose, "rArmPose":rArmPose})
        # basket.openrave_body.set_pose(basket.pose[:, 0], basket.rotation[:, 0])

        start = time.time()
        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = callback, n_resamples=10)
        end = time.time()

        print "Planning finished within {}s, displaying failed predicates...".format(end - start)
        velocites = np.zeros((plan.horizon, ))
        velocites[0:20] = 0.3

        ee_time = traj_retiming(plan, velocites)
        baxter.time = ee_time.reshape((1, ee_time.shape[0]))

        print "Saving current plan to file basket_putdown_isolation.hdf5..."
        serializer = PlanSerializer()
        serializer.write_plan_to_hdf5("basket_putdown_isolation_plan.hdf5", plan)
        self.assertTrue(result)
        import ipdb; ipdb.set_trace()


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
        manip = rave_body.env_body.GetManipulator("left_arm")

        solver = robot_ll_solver.RobotLLSolver()
        result = solver.solve(plan, callback = lambda: None, n_resamples=0)

        robot_poses = solver.pose_suggester(plan, 0)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(manip.GetTransform()[:3,3], ee_pose.value[:, 0] + offset))

        ee_pose = plan.params["cp_ee_1"]
        robot_poses = solver.pose_suggester(plan, 2)
        rave_body.set_dof(robot_poses[0])
        self.assertTrue(np.allclose(manip.GetTransform()[:3,3], ee_pose.value[:, 0] + offset))
        import ipdb; ipdb.set_trace()

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

    def test_washer_position(self):
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

        grasp_rot = np.array([0,np.pi/2,-np.pi/2])
        robot_body = robot.openrave_body
        baskey_body = basket.openrave_body
        washer_body = washer.openrave_body
        offset = [-0.035,0.055,-0.1]
        # -0.035,0.055,-0.1
        tool_link = washer_body.env_body.GetLink("washer_handle")
        washer_handle_pos = tool_link.GetTransform().dot(np.r_[offset, 1])[:3]
        robot_body.set_pose([0,0,np.pi/3])
        high_offset = [0,0,const.APPROACH_DIST*const.EEREACHABLE_STEPS]
        l_arm_pose = robot_body.get_ik_from_pose(washer_handle_pos+high_offset, grasp_rot, "left_arm")[1]
        robot_body.set_dof({'lArmPose': l_arm_pose})
        import ipdb; ipdb.set_trace()

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
        import ipdb; ipdb.set_trace()

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

        import ipdb; ipdb.set_trace()

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
        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    unittest.main()
