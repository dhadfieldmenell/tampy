import unittest
from core.util_classes import box, matrix
from core.util_classes.param_setup import ParamSetup
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import OpenRAVEBody
from core.internal_repr import parameter
from openravepy import Environment
import numpy as np
import time

class TestBaxter(unittest.TestCase):

    def test_baxter_ik(self):
        from openravepy import ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
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
        import main
        from core.parsing import parse_domain_config
        from core.parsing import parse_problem_config
        from core.util_classes.viewer import OpenRAVEViewer
        from openravepy import Environment
        from openravepy import ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
        domain_fname = '../domains/baxter_domain/baxter.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        # p_fname = '../domains/baxter_domain/baxter_probs/grasp_1234_1.prob'
        p_fname = '../domains/baxter_domain/baxter_probs/move_1234_3.prob'
        p_c = main.parse_file_to_dict(p_fname)
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        params = problem.init_state.params

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
