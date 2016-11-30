import unittest
import time
import main
import numpy as np
from core.parsing import parse_domain_config, parse_problem_config
from core.internal_repr import parameter
from core.util_classes import box, matrix
from core.util_classes.param_setup import ParamSetup
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.baxter_sampling import process_traj
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

class TestBaxter(unittest.TestCase):

    def test_baxter_ik(self):
        env = ParamSetup.setup_env()
        baxter = ParamSetup.setup_baxter()
        can = ParamSetup.setup_green_can(geom = (0.02,0.25))
        baxter_body = OpenRAVEBody(env, baxter.name, baxter.geom)
        can_body = OpenRAVEBody(env, can.name, can.geom)
        baxter_body.set_transparency(0.5)
        can_body.set_transparency(0.5)
        manip = baxter_body.env_body.GetManipulator('right_arm')
        robot = baxter_body.env_body
        can = can_body.env_body
        dof = robot.GetActiveDOFValues()
        #Open the Gripper so there won't be collisions between gripper and can
        dof[9], dof[-1] = 0.02, 0.02
        robot.SetActiveDOFValues(dof)
        iktype = IkParameterizationType.Transform6D
        thetas = np.linspace(0, np.pi*2, 10)
        target_trans = OpenRAVEBody.transform_from_obj_pose([0.9,-0.23,0.93], [0,0,0])
        can_body.env_body.SetTransform(target_trans)
        target_trans[:3,:3]  = target_trans[:3,:3].dot(matrixFromAxisAngle([0, np.pi/2, 0])[:3,:3])
        can_trans = target_trans
        body_trans = np.eye(4)
        """
        To check whether baxter ik model works, uncomment the following
        """
        # env.SetViewer('qtcoin')
        for theta in thetas:
            can_trans[:3,:3] = target_trans[:3,:3].dot(matrixFromAxisAngle([theta,0,0])[:3,:3])
            solution =  manip.FindIKSolutions(IkParameterization(can_trans, iktype), IkFilterOptions.CheckEnvCollisions)
            if len(solution) > 0:
                print "Solution found with pose and rotation:"
                print OpenRAVEBody.obj_pose_from_transform(can_trans)
            else:
                print "Solution not found with pose and rotation:"
                print OpenRAVEBody.obj_pose_from_transform(can_trans)
            for sols in solution:
                dof[10:17] = sols
                robot.SetActiveDOFValues(dof)
                time.sleep(.2)
                body_trans[:3,:3] = can_trans[:3,:3].dot(matrixFromAxisAngle([0, -np.pi/2, 0])[:3,:3])
                can_body.env_body.SetTransform(body_trans)
                self.assertTrue(np.allclose([0.9,-0.23,0.93], manip.GetTransform()[:3,3]))
                # import ipdb; ipdb.set_trace()

    def test_can_world(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter.domain', '../domains/baxter_domain/baxter_probs/grasp_1234_1.prob')

        env = Environment()
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        view = OpenRAVEViewer(env)
        view.draw(objLst, 0, 0.7)
        can_body = view.name_to_rave_body["can0"]
        baxter_body = view.name_to_rave_body["baxter"]
        can = can_body.env_body
        robot = baxter_body.env_body
        dof = robot.GetActiveDOFValues()

        # For Fining Initial Pose in Move Action
        # robot.SetActiveDOFValues(dof)
        # import ipdb; ipdb.set_trace()


        # For Finding Initial Pose in Grasp Action
        # can_pose = OpenRAVEBody.obj_pose_from_transform(can.GetTransform())
        # solution = baxter_body.get_ik_from_pose(can_pose[:3] + np.array([-0.2,0,0]), can_pose[3:], "right_arm")
        # print "Feasible Right Arm Pose Would be:"
        # print np.round(solution[0], 3)
        # print "Coresponding can pose is:"
        # print params["can0"].pose.flatten()
        # dof[-8:-1] = solution[0]
        # robot.SetActiveDOFValues(dof)
        # dof[-8:-1] = baxter_body.get_ik_from_pose(can_pose[:3] + np.array([0,0,0]), can_pose[3:], "right_arm")[0]
        # robot.SetActiveDOFValues(dof)
        # import ipdb;ipdb.set_trace()




    def test_rrt_planner(self):
        # Adopting examples from openrave
        domain, problem, params = load_environment('../domains/baxter_domain/baxter.domain', '../domains/baxter_domain/baxter_probs/grasp_1234_1.prob')

        env = Environment() # create openrave environment
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        view = OpenRAVEViewer(env)
        view.draw(objLst, 0, 0.7)
        can_body = view.name_to_rave_body["can0"]
        baxter_body = view.name_to_rave_body["baxter"]
        can = can_body.env_body
        robot = baxter_body.env_body
        dof = robot.GetActiveDOFValues()

        inds = baxter_body._geom.dof_map['rArmPose']
        r_init = params['robot_init_pose']
        r_end = params['robot_end_pose']

        dof[inds] = r_init.rArmPose.flatten()
        robot.SetActiveDOFValues(dof)

        robot.SetActiveDOFs(inds) # set joints the first 4 dofs

        plan_params = Planner.PlannerParameters()
        plan_params.SetRobotActiveJoints(robot)
        plan_params.SetGoalConfig([ 0.7  , -0.204,  0.862,  1.217,  2.731,  0.665,  2.598]) # set goal to all ones
        # forces parabolic planning with 40 iterations

        traj = RaveCreateTrajectory(env,'')
        # Using openrave built in planner
        trajectory  = {}
        plan_params.SetExtraParameters("""  <_postprocessing planner="parabolicsmoother">
                                                <_nmaxiterations>17</_nmaxiterations>
                                            </_postprocessing>""")
        trajectory["BiRRT"] = planing(env, robot, plan_params, traj, 'BiRRT')  # 3.5s
        # trajectory["BasicRRT"] = planing(env, robot, plan_params, traj, 'BasicRRT') # 0.05s can't run it by its own
        # trajectory["ExplorationRRT"] = planing(env, robot, plan_params, traj, 'ExplorationRRT') # 0.03s

        # plan_params.SetExtraParameters('<range>0.2</range>')
        # Using OMPL planner
        # trajectory["OMPL_RRTConnect"] = planing(env, robot, plan_params, traj, 'OMPL_RRTConnect') # 1.5s
        # trajectory["OMPL_RRT"] = planing(env, robot, plan_params, traj, 'OMPL_RRT') # 10s
        # trajectory["OMPL_RRTstar"] = planing(env, robot, plan_params, traj, 'OMPL_RRTstar') # 10s
        # trajectory["OMPL_TRRT"] = planing(env, robot, plan_params, traj, 'OMPL_TRRT')  # 10s
        # trajectory["OMPL_pRRT"] = planing(env, robot, plan_params, traj, 'OMPL_pRRT') # Having issue, freeze
        # trajectory["OMPL_LazyRRT"] = planing(env, robot, plan_params, traj, 'OMPL_LazyRRT') # 1.5s - 10s unsatble
        # ompl_traj = trajectory["OMPL_RRTConnect"]
        or_traj = trajectory["BiRRT"]

        result = process_traj(or_traj, 20)
        self.assertTrue(len(result) == 20)




    def test_random_init(self):
        domain, problem, params = load_environment('../domains/baxter_domain/baxter.domain', '../domains/baxter_domain/baxter_probs/grasp_1234_1.prob')
        env = Environment() # create openrave environment
        objLst = [i[1] for i in params.items() if not i[1].is_symbol()]
        view = OpenRAVEViewer(env)
        view.draw(objLst, 0, 0.7)
        can_body = view.name_to_rave_body["can0"]
        baxter_body = view.name_to_rave_body["baxter"]
        can = can_body.env_body
        robot = baxter_body.env_body
        dof = robot.GetActiveDOFValues()

        dof_inds = np.r_[dof_map["lArmPose"], dof_map["lGripper"], dof_map["rArmPose"], dof_map["rGripper"]]

        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].flatten()
        active_lb = lb_limit[dof_inds].flatten()
