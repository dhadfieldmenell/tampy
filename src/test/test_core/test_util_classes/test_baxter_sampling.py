import unittest
import time
import main
import numpy as np
from core.parsing import parse_domain_config, parse_problem_config
from core.internal_repr import parameter
from core.util_classes import box, matrix, baxter_predicates, baxter_sampling
from core.util_classes.param_setup import ParamSetup
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from openravepy import Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle

def load_environment(domain_file, problem_file):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    p_fname = problem_file
    p_c = main.parse_file_to_dict(p_fname)
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    params = problem.init_state.params
    return domain, problem, params

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
    print "{} Proforms: {}s".format(planner, total)
    return traj_list


class TestBaxterSampling(unittest.TestCase):

    def test_resample_ee_reachable(self):
        domain, problem, params = load_environment("../domains/baxter_domain/baxter.domain", "../domains/baxter_domain/baxter_probs/grasp_1234_1.prob")

        env = problem.env
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        view = OpenRAVEViewer(env)
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
        val, attr_inds = baxter_sampling.resample_eereachable(pred, None, 3, None)
        self.assertTrue(pred.test(3))

        # import ipdb; ipdb.set_trace()
        #
        # can2 = params['can1']
        # ee_target.value, ee_target.rotation = can2.pose, can2.rotation
        # self.assertFalse(pred.test(3))
        # val, attr_inds = baxter_sampling.resample_eereachable(pred, None, 3, None)
        # import ipdb; ipdb.set_trace()

    def test_resampling_rrt(self):
        domain, problem, params = load_environment("../domains/baxter_domain/baxter.domain", "../domains/baxter_domain/baxter_probs/grasp_1234_1.prob")

        env = problem.env
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]

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
        val, attr_inds = pred.resample(None, 8, None)

        import ipdb; ipdb.set_trace()
