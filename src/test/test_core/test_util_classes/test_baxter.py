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

    def test_baxter(self):
        from openravepy import ikfast, IkParameterizationType, IkParameterization, IkFilterOptions, databases, matrixFromAxisAngle
        env = ParamSetup.setup_env()
        env.SetViewer('qtcoin')

        baxter = ParamSetup.setup_baxter()
        can = ParamSetup.setup_green_can(geom = (0.02,0.25))
        can.pose = np.array([1,1,1])
        baxter.pose = np.zeros((1,1))
        baxter.rArmPose = np.array([[0,-0.785,0.785,1.57,-0.785,-0.785,0]]).T

        baxter_body = OpenRAVEBody(env, baxter.name, baxter.geom)
        can_body = OpenRAVEBody(env, can.name, can.geom)
        baxter_body.set_transparency(0.5)
        can_body.set_transparency(0.5)

        can_body.set_pose([.88, -.4,.86],[0,0,0])

        manip = baxter_body.env_body.GetManipulator('right_arm')
        robot = baxter_body.env_body
        dof = robot.GetActiveDOFValues()
        iktype = IkParameterizationType.Transform6D
        target_trans = OpenRAVEBody.transform_from_obj_pose(can.pose, can.rotation)
        rot_mat = matrixFromAxisAngle([0, np.pi/2, 0]).dot(matrixFromAxisAngle([0,0,np.pi]))
        target_trans = target_trans.dot(rot_mat)

        robot_trans1 = np.array([[-0.1957335 ,  0.03309012,  0.98009869,  0.88],
                                [ 0.87043024,  0.46621701,  0.15809141,  0.17],
                                [-0.45170741,  0.88405133, -0.12005693,  1.09 ],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])
        robot_trans2 = np.array([[ -7.74580400e-12,   3.75926422e-01,   9.26649516e-01, 1.01643265e+00],
                                [ -3.14398507e-12,  -9.26649516e-01,   3.75926422e-01, -2.49536286e-01],
                                [  1.00000000e+00,  -1.55431223e-15,   8.35940345e-12,  1.35365985e+00],
                                [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  1.00000000e+00]])
        solution =  manip.FindIKSolutions(IkParameterization(robot_trans1, iktype), IkFilterOptions.CheckEnvCollisions)
        import ipdb; ipdb.set_trace()
        for sol in solution:
            dof[10:17] = sol
            robot.SetActiveDOFValues(dof)
            self.assertTrue(np.allclose(robot_trans1, manip.GetTransform()))
            time.sleep(.2)

        solution =  manip.FindIKSolutions(IkParameterization(robot_trans2, iktype), IkFilterOptions.CheckEnvCollisions)
        for sol in solution:
            dof[10:17] = sol
            robot.SetActiveDOFValues(dof)
            self.assertTrue(np.allclose(robot_trans2, manip.GetTransform()))
            time.sleep(.1)

        import ipdb; ipdb.set_trace()

        """
        To check whether baxter model works, uncomment the following
        """
        print 'done'
        import ipdb; ipdb.set_trace()
