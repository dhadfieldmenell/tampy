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

def traj_retiming(plan):
    velocity = np.zeros(plan.horizon)
    velocity[0:15] = 0.25
    velocity[15:25] = 0.04
    velocity[25:55] = 0.15
    velocity[55:65] = 0.04
    velocity[65:80] = 0.25

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
        plan_str = [
        '0: MOVE BAXTER ROBOT_INIT_POSE ROBOT_GRASP_BEGIN',
        '1: BASKET_GRASP BAXTER BASKET INIT_TARGET ROBOT_GRASP_BEGIN GRASP_EE_LEFT GRASP_EE_RIGHT ROBOT_GRASP_END',
        '2: MOVEHOLDING BAXTER ROBOT_GRASP_END ROBOT_PUTDOWN_BEGIN BASKET',
        '3: BASKET_PUTDOWN BAXTER BASKET END_TARGET ROBOT_PUTDOWN_BEGIN PUTDOWN_EE_LEFT PUTDOWN_EE_RIGHT ROBOT_PUTDOWN_END',
        '4: MOVE BAXTER ROBOT_PUTDOWN_END ROBOT_END_POSE'
        # '5: OPEN_DOOR BAXTER WASHER ROBOT_WASHER_BEGIN WASHER_EE ROBOT_WASHER_END WASHER_INIT_POSE WASHER_END_POSE'
        # '6: MOVE BAXTER ROBOT_WASHER_END ROBOT_END_POSE',
        ]

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

        # baxter.time = traj_retiming(plan).reshape((1, plan.horizon))
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
        offset = [0,0.317,0]
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
        offset = [0,0.317,0]
        basket_pos = basket.pose.flatten()
        left_targ = basket_pos + offset + [0,0,5*const.APPROACH_DIST]
        right_targ = basket_pos - offset + [0,0,5*const.APPROACH_DIST]
        grasp_rot = np.array([0,np.pi/2,0])

        robot_body = robot.openrave_body
        baskey_body = basket.openrave_body
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
        import ipdb; ipdb.set_trace()

    def collision_debug_env(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5("debug1.hdf5")
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        robot = plan.params['baxter']
        basket = plan.params['basket']
        viewer.draw_plan_ts(plan, 23)
        viewer.draw_cols_ts(plan, 23)
        preds = [(negated, pred, t) for negated, pred, t in plan.get_failed_preds(priority = 2, tol = 1e-3)]
        pred = preds[0][1]
        print [(pred.get_type()+"_"+str(t), np.max(pred.get_expr(negated=negated).expr.eval(pred.get_param_vector(t)))) for negated, pred, t in preds]

        basket_pos = plan.params['basket'].pose[:, 23]
        offset = [0,0.312,0]
        shift = [0.1,0,0]
        robot_body = robot.openrave_body
        l_arm_pose = robot_body.get_ik_from_pose(basket_pos+shift+offset, [0,np.pi/2,0], "left_arm")[0]
        r_arm_pose = robot_body.get_ik_from_pose(basket_pos+[0,0,-0.2], [0,np.pi/2,0], "right_arm")[0]

        robot.lArmPose[:, 23] = l_arm_pose
        robot.rArmPose[:, 23] = r_arm_pose
        print [(pred.get_type()+"_"+str(t), np.max(pred.get_expr(negated=negated).expr.eval(pred.get_param_vector(t)))) for negated, pred, t in preds]
        viewer.draw_plan_ts(plan, 23)
        viewer.draw_cols_ts(plan, 23)
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
