import unittest
from core.util_classes import robot_predicates, baxter_predicates, matrix
from errors_exceptions import PredicateException, ParamValidationException
from core.util_classes.param_setup import ParamSetup
from core.util_classes.openrave_body import OpenRAVEBody
import numpy as np

class TestBaxterPredicates(unittest.TestCase):

    # Begin of the test

    def test_robot_at(self):

        # RobotAt, Robot, RobotPose
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        pred = baxter_predicates.BaxterRobotAt("testRobotAt", [robot, rPose], ["Robot", "RobotPose"])
        self.assertEqual(pred.get_type(), "BaxterRobotAt")
        # Robot and RobotPose are initialized to the same pose
        self.assertTrue(pred.test(0))
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=2)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")
        robot.pose = np.array([[3, 4, 5, 3]])
        rPose.value = np.array([[3, 4, 2, 6]])
        self.assertTrue(pred.is_concrete())
        robot.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        robot.lGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        rPose.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
        robot.rArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        robot.lArmPose = np.array([[0,0,0,0,0,0,0],
                                   [1,2,3,4,5,6,7],
                                   [7,6,5,4,3,2,1],
                                   [0,0,0,0,0,0,0]]).T
        rPose.rArmPose = np.array([[0,0,0,0,0,0,0]]).T
        rPose.lArmPose = np.array([[0,0,0,0,0,0,0]]).T
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=4)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=-1)
        self.assertEqual(cm.exception.message, "Out of range time for predicate 'testRobotAt: (BaxterRobotAt baxter baxter_pose)'.")

        self.assertTrue(pred.test(time=0))
        self.assertFalse(pred.test(time=1))
        self.assertFalse(pred.test(time=2))
        self.assertTrue(pred.test(time=3))

    def test_is_mp(self):
        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterIsMP("test_isMP", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "BaxterIsMP")
        with self.assertRaises(PredicateException) as cm:
            pred.test(time=0)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (BaxterIsMP baxter)' at the timestep.")
        # Getting lowerbound and movement step
        llA_l, lA_m = pred.lower_limit[0:7], pred.joint_step[0:7]
        lrA_l, rA_m = pred.lower_limit[8:15], pred.joint_step[8:15]
        llG_l, lG_m = pred.lower_limit[7], pred.joint_step[7]
        lrG_l, rG_m = pred.lower_limit[15], pred.joint_step[15]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[0,2,3,4,5,6,7]])

        # Arm pose is valid in the timestep: 0,1,2,3
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))

        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))

        # Gripper pose is valid in the timestep: 0,1,3,4,5
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Thus only timestep 1 and 3 are valid
        # import ipdb; ipdb.set_trace()
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertFalse(pred.test(4))
        self.assertFalse(pred.test(5))
        with self.assertRaises(PredicateException) as cm:
            pred.test(6)
        self.assertEqual(cm.exception.message, "Insufficient pose trajectory to check dynamic predicate 'test_isMP: (BaxterIsMP baxter)' at the timestep.")

    def test_within_joint_limit(self):
        robot = ParamSetup.setup_pr2()
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterWithinJointLimit("test_joint_limit", [robot], ["Robot"], test_env)
        self.assertEqual(pred.get_type(), "BaxterWithinJointLimit")
        # Getting lowerbound and movement step
        llA_l, lA_m = pred.lower_limit[0:7], pred.joint_step[0:7]
        lrA_l, rA_m = pred.lower_limit[8:15], pred.joint_step[8:15]
        llG_l, lG_m = pred.lower_limit[7], pred.joint_step[7]
        lrG_l, rG_m = pred.lower_limit[15], pred.joint_step[15]
        # Base pose is valid in the timestep: 1,2,3,4,5
        robot.pose = np.array([[0,2,3,4,5,6,7]])

        # timestep 6 should fail
        robot.rArmPose = np.hstack((lrA_l+rA_m, lrA_l+2*rA_m, lrA_l+3*rA_m, lrA_l+4*rA_m, lrA_l+3*rA_m, lrA_l+5*rA_m, lrA_l+100*rA_m))
        # timestep 1 should fail
        robot.lArmPose = np.hstack((llA_l+lA_m, llA_l-lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m, llA_l+lA_m))
        robot.rGripper = np.matrix([lrG_l, lrG_l+rG_m, lrG_l+2*rG_m, lrG_l+5*rG_m, lrG_l+4*rG_m, lrG_l+3*rG_m, lrG_l+2*rG_m]).reshape((1,7))
        robot.lGripper = np.matrix([llG_l, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m, llG_l+lG_m]).reshape((1,7))
        # Thus timestep 1, 6 should fail
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred.test(2))
        self.assertTrue(pred.test(3))
        self.assertTrue(pred.test(4))
        self.assertTrue(pred.test(5))
        self.assertFalse(pred.test(6))

    def test_in_gripper(self):

        tol = 1e-4
        TEST_GRAD = True
        # InGripper, Robot, Can
        robot = ParamSetup.setup_baxter()
        can = ParamSetup.setup_blue_can()
        test_env = ParamSetup.setup_env()
        # test_env.SetViewer("qtcoin")
        pred = baxter_predicates.BaxterInGripperPos("InGripper", [robot, can], ["Robot", "Can"], test_env)
        pred2 = baxter_predicates.BaxterInGripperRot("InGripper_rot", [robot, can], ["Robot", "Can"], test_env)
        # Since this predicate is not yet concrete
        pred._param_to_body[robot].set_transparency(.7)
        self.assertFalse(pred.test(0))
        can.pose = np.array([[0,0,0]]).T

        # initialized pose value is not right
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        # check the gradient of the implementations (correct)
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        # Now set can's pose and rotation to be the right things
        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        # A new robot arm pose
        robot.rArmPose = np.array([[0,-np.pi/4,np.pi/4,np.pi/2,-np.pi/4,-np.pi/4,0]]).T
        self.assertFalse(pred.test(0))

        # Only the pos is correct, rotation is not yet right
        can.pose = np.array([[1.08769922, -0.31906039,  1.21028557]]).T
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        can.rotation = np.array([[-2.84786534,  0.25268026, -2.98976055]]).T
        self.assertTrue(pred.test(0))
        self.assertTrue(pred2.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        # now rotate robot basepose
        robot.pose = np.array([[np.pi/3]]).T
        self.assertFalse(pred.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        can.pose = np.array([[0.82016401,  0.78244496,  1.21028557]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred2.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)
        can.rotation = np.array([[-1.80066778,  0.25268026, -2.98976055]]).T
        self.assertTrue(pred2.test(0))
        self.assertTrue(pred.test(0))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), True, tol)

    def test_ee_reachable(self):
        # InGripper, Robot, Can

        APPROACH_DIST = 0.05
        RETREAT_DIST = 0.075
        EEREACHABLE_STEPS = 3

        debug = True
        tol = 1e-4
        TEST_GRAD = True
        robot = ParamSetup.setup_baxter()
        test_env = ParamSetup.setup_env()
        rPose = ParamSetup.setup_baxter_pose()
        ee_pose = ParamSetup.setup_pr2_ee_pose()
        if debug == True:
            test_env.SetViewer("qtcoin")
        pred = baxter_predicates.BaxterEEReachablePos("ee_reachable", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        pred2 = baxter_predicates.BaxterEEReachableRot("ee_reachable_rot", [robot, rPose, ee_pose], ["Robot", "RobotPose", "EEPose"], test_env)
        baxter = pred._param_to_body[robot]
        # Since this predicate is not yet concrete
        self.assertFalse(pred.test(0))

        ee_pose.value = np.array([[1.2, -0.1, 0.925]]).T
        ee_pose.rotation = np.array([[0,0,0]]).T
        ee_pos = ParamSetup.setup_green_can()
        ee_body = OpenRAVEBody(test_env, "EE_Pose", ee_pos.geom)
        ee_body.set_pose(ee_pose.value[:, 0], ee_pose.rotation[:, 0])

        robot.lArmPose = np.zeros((7,7))
        robot.lGripper = np.ones((1, 7))*0.02
        robot.rGripper = np.ones((1, 7))*0.02
        robot.pose = np.zeros((1,7))
        robot.rArmPose = np.zeros((7,7))
        # initialized pose value is not right
        self.assertFalse(pred.test(0))

        # Find IK Solution
        # trajectory = []
        # trajectory.append(baxter.get_ik_from_pose([1.2-3*APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])    #s=-3
        # trajectory.append(baxter.get_ik_from_pose([1.2-2*APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])    #s=-2
        # trajectory.append(baxter.get_ik_from_pose([1.2-APPROACH_DIST, -0.1, 0.925], [0,0,0], "right_arm")[0])       #s=-1
        # trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925], [0,0,0], "right_arm")[0])                     #s=0
        # trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+RETREAT_DIST], [0,0,0], "right_arm")[0])        #s=1
        # trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+2*RETREAT_DIST], [0,0,0], "right_arm")[0])      #s=2
        # trajectory.append(baxter.get_ik_from_pose([1.2, -0.1, 0.925+3*RETREAT_DIST], [0,0,0], "right_arm")[0])      #s=3
        # trajectory = np.array(trajectory)

        trajectory = np.array([ [ 1.2       , -0.66775957, -0.17303322,  1.76317334,  0.34777659, -1.07165615, -0.02965195],
                                [ 0.7       , -0.52284073,  0.49408572,  1.6187174 , -2.96184523,  1.11901404,  2.63306623],
                                [ 0.6       , -0.36531823,  0.68122615,  1.40221106, -3.04906318,  1.00349484,  2.4607322 ],
                                [ 0.9       , -0.41957516,  0.18658737,  1.13732271,  0.27464322, -0.76306282, -0.3742298 ],
                                [ 0.8       , -0.54378935,  0.35337336,  1.20946449,  0.29660299, -0.7200256 , -0.53242542],
                                [ 0.5       ,  0.47193719,  2.18166327,  1.19676408, -0.30601498, -0.70770242, -2.06691973],
                                [ 0.5       , -0.55736808,  0.98450067,  1.22984612, -2.71983422,  0.71059423,  1.99023137]])

        robot.rArmPose = trajectory.T
        # Predicate should succeed in the grasping post at t=3,
        # EEreachableRot should always pass since rotation is right all the time
        self.assertFalse(pred.test(0))
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(1))
        self.assertTrue(pred2.test(1))
        self.assertFalse(pred.test(2))
        self.assertTrue(pred2.test(2))
        self.assertTrue(pred.test(3))
        self.assertTrue(pred2.test(3))
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(3), True, tol)

    def test_obstructs(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can

        TOL = 1e-4
        TEST_GRAD = False
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        can = ParamSetup.setup_blue_can()
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterObstructs("test_obstructs", [robot, rPose, rPose, can], ["Robot", "RobotPose", "RobotPose", "Can"], test_env, tol=TOL)
        self.assertEqual(pred.get_type(), "BaxterObstructs")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        can.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        # This gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=5e-2)

        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # This gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), True, 1e-1)

        # Move can to the center of the gripper (touching -> should recognize as collision)
        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # The gradient test below doesn't work because the collision normals in
        # the robot's right gripper already are inaccurate because the can is there.
        # if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)
        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)

        # Move can into the robot arm, should have collision
        can.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # The gradient test below doesn't work because the collision normals for
        # the robot's r_wrist_flex_link are inaccurate because the can is there.
        # if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=1e-1)

    def test_obstructs_holding(self):

        # Obstructs, Robot, RobotPose, RobotPose, Can, Can

        TEST_GRAD = False
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        can = ParamSetup.setup_blue_can("can1", (0.02, 0.25))
        can_held = ParamSetup.setup_blue_can("can2", (0.02,0.25))
        test_env = ParamSetup.setup_env()
        pred = baxter_predicates.BaxterObstructsHolding("test_obstructs", [robot, rPose, rPose, can, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env, debug = True)
        self.assertEqual(pred.get_type(), "BaxterObstructsHolding")

        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        # Move can so that it collide with robot base
        rPose.value = can.pose = np.array([[0],[0],[0]])
        can_held.pose = np.array([[.5],[.5],[0]])
        self.assertTrue(pred.test(0))
        # This Grandient test passes
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move can away so there is no collision
        can.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred.test(0))
        # This Grandient test passes
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move can to the center of the gripper (touching -> should recognize as collision)
        can.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This Gradient test failed, failed Link-> right gripper fingers
        # if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move can away from the gripper, no collision
        can.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This Gradient test passed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        # Move caheldn into the robot arm, should have collision
        can.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient checks failed
        # if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        pred._plot_handles = []

        pred2 = baxter_predicates.BaxterObstructsHolding("test_obstructs_held", [robot, rPose, rPose, can_held, can_held], ["Robot", "RobotPose", "RobotPose", "Can", "Can"], test_env, debug = True)
        rPose.value = can_held.pose = can.pose = np.array([[0],[0],[0]])
        pred._param_to_body[can].set_pose(can.pose, can.rotation)
        self.assertTrue(pred2.test(0))
        can_held.pose = np.array([[0],[0],[-2]])
        self.assertFalse(pred2.test(0))
        # This Grandient test passed
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move can to the center of the gripper (touching -> should allow touching)
        can_held.pose = np.array([[0.96897233, -1.10397558,  1.006976]]).T
        self.assertTrue(pred2.test(0, negated = True))
        self.assertFalse(pred2.test(0))

        # This Gradient test fails ->failed link: l_finger_tip, r_finger_tip, r_gripper_palm
        # if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move can away from the gripper, no collision
        can_held.pose = np.array([[.700,  -.127,   .838]]).T
        self.assertFalse(pred2.test(0))
        self.assertTrue(pred2.test(0, negated = True))
        # This Gradient test passed
        if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

        # Move caheldn into the robot arm, should have collision
        can_held.pose = np.array([[.5, -.6, .9]]).T
        self.assertTrue(pred2.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This Gradient test failed -> failed link: r_gripper_l_finger, r_gripper_r_finger
        # if TEST_GRAD: pred2.expr.expr.grad(pred2.get_param_vector(0), num_check=True, atol=.1)

    def test_r_collides(self):

        # RCollides Robot Obstacle

        TEST_GRAD = False
        robot = ParamSetup.setup_baxter()
        rPose = ParamSetup.setup_baxter_pose()
        table = ParamSetup.setup_box()
        test_env = ParamSetup.setup_env()
        # test_env.SetViewer("qtcoin")
        pred = baxter_predicates.BaxterRCollides("test_r_collides", [robot, table], ["Robot", "Table"], test_env, debug = True)
        # self.assertEqual(pred.get_type(), "RCollides")
        # Since can is not yet defined
        self.assertFalse(pred.test(0))
        table.pose = np.array([[0],[0],[0]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        # Move can so that it collide with robot base
        table.pose = np.array([[0],[0],[1.5]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        # Move can away so there is no collision
        table.pose = np.array([[0],[2],[.75]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[0],[0],[3]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[0],[0],[-0.4]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test failed
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.pose = np.array([[1],[1],[.75]])
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)
        table.rotation = np.array([[.5,.5,-.5]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        table.pose = np.array([[.5],[.5],[2.2]])
        self.assertFalse(pred.test(0))
        self.assertTrue(pred.test(0, negated = True))

        table.pose = np.array([[.5],[1.45],[.5]])
        table.rotation = np.array([[0.8,0,0]]).T
        self.assertTrue(pred.test(0))
        self.assertFalse(pred.test(0, negated = True))
        # This gradient test passed with a box
        if TEST_GRAD: pred.expr.expr.grad(pred.get_param_vector(0), num_check=True, atol=.1)

        """
            Uncomment the following to see the robot
        """
        # pred._param_to_body[table].set_pose(table.pose, table.rotation)
        # import ipdb; ipdb.set_trace()
