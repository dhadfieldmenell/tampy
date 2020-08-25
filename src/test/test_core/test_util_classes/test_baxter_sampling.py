import unittest
import time
import main
import numpy as np
from pma import hl_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.internal_repr import parameter
from core.util_classes import  matrix, baxter_predicates, baxter_sampling
from core.util_classes.param_setup import ParamSetup
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from openravepy import Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
import core.util_classes.baxter_constants as const
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    hls = hl_solver.FFSolver(d_c)
    plan = hls.get_plan(['0: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 ROBOT_END_POSE'], domain, problem)
    return domain, problem, params, plan

def planing(env, robot, params, traj, planner):
    t0 = time.time()
    planner=RaveCreatePlanner(env, planner)
    planner.InitPlan(robot, params)
    planner.PlanPath(traj)
    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(traj.GetWaypoint(i),robot,robot.GetActiveDOFIndices())
        traj_list.append(np.round(dofvalues, 3))
    t1 = time.time()
    total = t1-t0
    print("{} Proforms: {}s".format(planner, total))
    return traj_list


class TestBaxterSampling(unittest.TestCase):

    def test_resample_ee_reachable(self):
        domain, problem, params, plan = load_environment("../domains/baxter_domain/baxter.domain", "../domains/baxter_domain/baxter_probs/grasp_1234_1.prob")

        env = problem.env
        objLst = [i[1] for i in list(params.items()) if not i[1].is_symbol()]
        view = OpenRAVEViewer.create_viewer(env)
        view.draw(objLst, 0, 0.7)
        baxter = params['baxter']
        robot_pose = params['robot_init_pose']
        ee_target = params['ee_target0']
        can = params['can0']

        # dof = robot.GetActiveDOFValues()
        pred = baxter_predicates.BaxterEEReachablePos("resample_tester", [baxter, robot_pose, ee_target], ["Robot", "RobotPose", "EEPose"], env)
        # Initialize the trajectory of each parameters
        ee_target.value, ee_target.rotation = can.pose, can.rotation
        baxter.rArmPose = np.zeros((7, 7))
        baxter.lArmPose = np.zeros((7, 7))
        baxter.rGripper = 0.02 * np.ones((1, 7))
        baxter.lGripper = 0.02 * np.ones((1, 7))
        baxter.pose = np.zeros((1, 7))
        baxter._free_attrs['rArmPose'] = np.ones((7,7))
        # Having initial Arm Pose. not supposed to be Ture
        self.assertFalse(pred.test(3))
        val, attr_inds = baxter_sampling.resample_eereachable(pred, False, 3, plan)
        self.assertTrue(pred.test(3))

        # can2 = params['can1']
        # ee_target.value, ee_target.rotation = can2.pose, can2.rotation
        # self.assertFalse(pred.test(3))
        # val, attr_inds = baxter_sampling.resample_eereachable(pred, None, 3, None)
        # import ipdb; ipdb.set_trace()

    def test_resampling_rrt(self):
        domain, problem, params, plan = load_environment("../domains/baxter_domain/baxter.domain", "../domains/baxter_domain/baxter_probs/grasp_1234_1.prob")

        env = problem.env
        objLst = [i[1] for i in list(params.items()) if not i[1].is_symbol()]

        # view = OpenRAVEViewer(env)
        # view.draw(objLst, 0, 0.7)
        baxter = params['baxter']
        startp = params['robot_init_pose']
        endp = params['robot_end_pose']
        can = params['can0']
        pred = baxter_predicates.BaxterObstructs("resample_obstructs_tester", [baxter, startp, endp, can], ["Robot", "RobotPose", "RobotPose", "Can"], env)
        baxter.rArmPose = np.zeros((7, 40))
        baxter.lArmPose = np.zeros((7, 40))
        baxter.rGripper = 0.02 * np.ones((1, 40))
        baxter.lGripper = 0.02 * np.ones((1, 40))
        baxter.pose = np.zeros((1, 40))
        baxter._free_attrs['rArmPose'] = np.ones((7,40))
        # val, attr_inds = pred.resample(False, 8, plan)

        # import ipdb; ipdb.set_trace()

    def test_resample_r_collides(self):
        domain, problem, params, plan = load_environment("../domains/baxter_domain/baxter.domain", "../domains/baxter_domain/baxter_probs/baxter_test_env.prob")
        viewer = OpenRAVEViewer.create_viewer(plan.env)
        viewer.draw_plan_ts(plan, 0)
        baxter = plan.params['baxter']
        obstacle = plan.params['table']
        rcollides_pred = baxter_predicates.BaxterRCollides("test_rcollides", [baxter, obstacle], ["Robot", "Obstacle"], plan.env)
        start_pose = np.array([ 0.4,  0.8,  1. ,  0.4,  0. ,  0. ,  0. ])
        end_pose = np.array([ 0.5  , -0.881,  0.814,  1.669, -2.672,  0.864,  2.308])
        traj = []
        for i in range(7):
            traj.append(np.linspace(start_pose[i], end_pose[i], 40, endpoint=True))
        traj = np.r_[traj]
        baxter.rArmPose = traj
        baxter.lArmPose = np.repeat(baxter.lArmPose[:, 0].reshape((7, 1)), 40, axis = 1)
        baxter.rGripper = np.repeat(baxter.rGripper[:, 0].reshape((1, 1)), 40, axis = 1)
        baxter.lGripper = np.repeat(baxter.lGripper[:, 0].reshape((1, 1)), 40, axis = 1)
        baxter.pose = np.repeat(baxter.pose[:, 0].reshape((1, 1)), 40, axis = 1)
        obstacle.pose = np.repeat(obstacle.pose[:, 0].reshape((3, 1)), 40, axis = 1)
        obstacle.rotation = np.repeat(obstacle.rotation[:, 0].reshape((3, 1)), 40, axis = 1)
        for i in range(1, 30):
            self.assertTrue(rcollides_pred.test(i))

        # TODO Figure out why error occurs at this place
        # rcollides_pred.resample(False, 1, plan)
        # self.assertFalse(rcollides_pred.test(1))
        # import ipdb; ipdb.set_trace()
        # rcollides_pred.resample(False, 2, plan)


    def test_resample_basket_ee_reachable(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_plan')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        robot_pose = plan.params['robot_init_pose']
        ee_left = plan.params['bg_ee_left']
        ee_right = plan.params['bg_ee_right']
        ee_putdown_left = plan.params['bp_ee_left']
        ee_putdown_right = plan.params['bp_ee_right']

        basket = plan.params['basket']

        left_pred = baxter_predicates.BaxterEEReachableLeftVer('test_ee_left', [robot, robot_pose, ee_left], ['Robot', 'RobotPose', 'EEPose'], env=env)
        right_pred = baxter_predicates.BaxterEEReachableRightVer('test_ee_right', [robot, robot_pose, ee_right], ['Robot', 'RobotPose', 'EEPose'], env=env)

        left_pred2 = baxter_predicates.BaxterEEReachableLeftVer('test_ee_left', [robot, robot_pose, ee_putdown_left], ['Robot', 'RobotPose', 'EEPose'], env=env)
        right_pred2 = baxter_predicates.BaxterEEReachableRightVer('test_ee_right', [robot, robot_pose, ee_putdown_right], ['Robot', 'RobotPose', 'EEPose'], env=env)

        basket_pos, offset = basket.pose[:, 24], [0,const.BASKET_OFFSET,0]
        ee_left.value = np.array([basket_pos + offset]).T
        ee_left.rotation = np.array([[0,np.pi/2, 0]]).T
        ee_right.value = np.array([basket_pos - offset]).T
        ee_right.rotation = np.array([[0,np.pi/2, 0]]).T

        self.assertFalse(left_pred.test(24))
        self.assertFalse(right_pred.test(24))

        def resampled_value(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_basket_eereachable_rrt(pred, negated, t, plan)
            self.assertTrue(pred.test(t))

        resampled_value(left_pred, False, 24, plan)
        resampled_value(right_pred, False, 24, plan)
        resampled_value(left_pred2, False, 53, plan)
        resampled_value(right_pred2, False, 53, plan)

    def test_resample_eereachable_ver(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_plan.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        robot_pose = plan.params['robot_init_pose']
        ee_left = plan.params['bg_ee_left']
        ee_right = plan.params['bg_ee_right']
        ee_putdown_left = plan.params['bp_ee_left']
        ee_putdown_right = plan.params['bp_ee_right']
        basket = plan.params['basket']


        left_pred = baxter_predicates.BaxterEEReachableLeftVer('test_ee_left', [robot, robot_pose, ee_left], ['Robot', 'RobotPose', 'EEPose'], env=env)
        right_pred = baxter_predicates.BaxterEEReachableRightVer('test_ee_right', [robot, robot_pose, ee_right], ['Robot', 'RobotPose', 'EEPose'], env=env)

        left_pred2 = baxter_predicates.BaxterEEReachableLeftVer('test_ee_left', [robot, robot_pose, ee_putdown_left], ['Robot', 'RobotPose', 'EEPose'], env=env)
        right_pred2 = baxter_predicates.BaxterEEReachableRightVer('test_ee_right', [robot, robot_pose, ee_putdown_right], ['Robot', 'RobotPose', 'EEPose'], env=env)
        import ipdb; ipdb.set_trace()
        basket_pos, offset = basket.pose[:, 24], [0,const.BASKET_OFFSET,0]
        ee_left.value = np.array([basket_pos + offset]).T
        ee_left.rotation = np.array([[0,np.pi/2, 0]]).T
        ee_right.value = np.array([basket_pos - offset]).T
        ee_right.rotation = np.array([[0,np.pi/2, 0]]).T

        self.assertFalse(left_pred.test(24))
        self.assertFalse(right_pred.test(24))
        self.assertFalse(left_pred2.test(53))
        self.assertFalse(right_pred2.test(53))

        def resample_check(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_eereachable_ver(pred, negated, t, plan)
            self.assertTrue(pred.test(t))

        resample_check(left_pred, False, 24, plan)
        resample_check(right_pred, False, 24, plan)
        resample_check(left_pred2, False, 53, plan)
        resample_check(right_pred2, False, 53, plan)
        import ipdb; ipdb.set_trace()

    def test_resample_ee_approach_retreat(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_approach_retreat.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        robot_pose = plan.params['open_door_begin']
        ee_left = plan.params['open_door_ee']
        washer = plan.params['washer']
        washer.door[:, :] = 0
        approach_pred = baxter_predicates.BaxterEEApproachLeft('test_approach_left', [robot, robot_pose, ee_left], ['Robot', 'RobotPose', 'EEPose'], env=env)

        retreat_pred = baxter_predicates.BaxterEERetreatLeft('test_approach_left', [robot, robot_pose, ee_left], ['Robot', 'RobotPose', 'EEPose'], env=env)
        self.assertFalse(approach_pred.test(10))
        self.assertFalse(retreat_pred.test(20))

        def resampled_value(pred, negated, t, plan, approach = True):
            res, attr_inds = baxter_sampling.resample_washer_ee_approach(pred, negated, t, plan, approach = approach)
            self.assertTrue(pred.test(t))

        resampled_value(approach_pred, False, 10, plan, approach = True)
        resampled_value(retreat_pred, False, 20, plan, approach = False)

    def test_resample_in_gripper(self):
        # TODO resample in gripper doesn't quite work
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_plan')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        robot_pose = plan.params['robot_washer_begin']
        ee_left = plan.params['washer_ee']
        basket = plan.params['basket']

        pred = baxter_predicates.BaxterBasketInGripper('test_in_gripper', [robot, basket], ['Robot', 'Basket'], env=env)
        self.assertFalse(pred.test(30))

        def resampled_value(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_basket_in_gripper(pred, negated, t, plan)
            self.assertTrue(pred.test(t))

        checking_ts = list(range(30, 50))
        for ts in checking_ts:
            resampled_value(pred, False, ts, plan)

    def test_resample_cloth_in_gripper(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('initialized_plan')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        cloth = plan.params['cloth']

        pred = baxter_predicates.BaxterClothInGripperLeft('test_cloth_in_gripper', [robot, cloth], ['Robot', 'Cloth'], env=env)
        self.assertFalse(pred.test(30))

        def resampled_value(pred, negated, t, plan):
            cloth_rot = cloth.rotation[:, t]
            res, attr_inds = baxter_sampling.resample_cloth_in_gripper(pred, negated, t, plan)
            self.assertTrue(pred.test(t))
            self.assertTrue(np.allclose(cloth_rot, cloth.rotation[:, t]))

        for ts in range(30, 49):
            resampled_value(pred, False, ts, plan)

    def test_resample_washer_in_gripper(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('resample_washer_in_gripper_plan.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        washer = plan.params['washer']

        pred = baxter_predicates.BaxterWasherInGripper('test_washer_in_gripper', [robot, washer], ['Robot', 'Washer'], env=env)
        # self.assertFalse(pred.test(10, tol=1e-3))

        res, attr_inds = baxter_sampling.resample_washer_in_gripper(pred, False, 10, plan)

        checking_ts = list(range(10, 20))
        for ts in checking_ts:
            self.assertTrue(pred.test(ts, tol=1e-3))

        import ipdb; ipdb.set_trace()

    def test_resample_washer_approach_in_gripper(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('resample_washer_in_gripper_plan.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        washer = plan.params['washer']
        ee_pose  = plan.params['open_door_ee']
        robot_pose = plan.params['robot_init_pose']

        pred1 = baxter_predicates.BaxterEEApproachLeft('test_washer_in_gripper', [robot, robot_pose, ee_pose], ['Robot', 'RobotPose', "EEPose"], env=env)
        pred2 = baxter_predicates.BaxterWasherInGripper('test_washer_in_gripper', [robot, washer], ['Robot', 'Washer'], env=env)

        res, attr_inds = baxter_sampling.resample_washer_ee_approach(pred1, False, 10, plan, approach=True)
        self.assertTrue(pred2.test(10, tol=1e-3))
        res, attr_inds = baxter_sampling.resample_washer_in_gripper(pred2, False, 10, plan)
        self.assertTrue(pred1.test(10, tol=1e-3))

        import ipdb; ipdb.set_trace()


    def test_resample_basket_moveholding(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_plan')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        robot_pose = plan.params['robot_washer_begin']
        ee_left = plan.params['washer_ee']
        basket = plan.params['basket']

        pred = baxter_predicates.BaxterBasketInGripper('test_in_gripper', [robot, basket], ['Robot', 'Basket'], env=env)
        self.assertFalse(pred.test(30))

        def resampled_value(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_basket_moveholding(pred, negated, t, plan)
            self.assertTrue(pred.test(t))

        for ts in range(30, 50):
            resampled_value(pred, False, ts, plan)

    def test_resample_obstruct(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_obstruct.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        sp = plan.params['cloth_grasp_begin_1']
        ep = plan.params['cloth_grasp_end_1']
        basket = plan.params['basket']
        cloth = plan.params['cloth']
        pred = baxter_predicates.BaxterObstructsHoldingCloth('test_in_gripper', [robot, sp, ep, basket, cloth], ['Robot', 'RobotPose', 'RobotPose', 'Basket', 'Cloth'], env=env)
        self.assertFalse(pred.test(38, negated = True))
        plan.check_cnt_violation(priority = 3, tol = 1e-3)
        def resampled_value(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_basket_obstructs_holding(pred, negated, t, plan)
            self.assertTrue(pred.test(t, negated = negated))

        resampled_value(pred, True, 38, plan)

    def test_resample_obstruct_2(self):
        pd = PlanDeserializer()
        plan = pd.read_from_hdf5('test_resample_obstruct.hdf5')
        env = plan.env
        viewer = OpenRAVEViewer.create_viewer(env)
        viewer.draw_plan_ts(plan, 0)
        robot = plan.params['baxter']
        sp = plan.params['cloth_grasp_begin_1']
        ep = plan.params['cloth_grasp_end_1']
        basket = plan.params['basket']
        offset = np.array([0, baxter_constants.BASKET_OFFSET, 0])
        cloth = plan.params['cloth']
        pred = baxter_predicates.BaxterObstructsHoldingCloth('test_in_gripper', [robot, sp, ep, basket, cloth], ['Robot', 'RobotPose', 'RobotPose', 'Basket', 'Cloth'], env=env)
        self.assertFalse(pred.test(38, negated = True))
        lArmPose = robot.openrave_body.get_ik_from_pose(basket.pose[:, 0] , [0, np.pi/2,0], 'left_arm')[0]
        rArmPose = robot.openrave_body.get_ik_from_pose(basket.pose[:, 0] - offset, [0, np.pi/2,0], 'right_arm')[0]

        robot.lArmPose[:, 38] = lArmPose
        robot.rArmPose[:, 38] = rArmPose
        robot.lGripper[:, 38] = 0.015
        robot.rGripper[:, 38] = 0
        cloth.pose[:, 38] = basket.pose[:, 0]
        self.assertFalse(pred.test(38, negated = True))

        plan.check_cnt_violation(priority = 3, tol = 1e-3)
        def resampled_value(pred, negated, t, plan):
            res, attr_inds = baxter_sampling.resample_basket_obstructs_holding(pred, negated, t, plan)
            self.assertTrue(pred.test(t, negated = negated))

        resampled_value(pred, True, 38, plan)
