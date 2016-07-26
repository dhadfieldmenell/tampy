from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector3d, PR2PoseVector
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2 import PR2
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
import numpy as np
import ctrajoptpy

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""
BASE_MOVE = 1e0
JOINT_MOVE = np.pi/8
dsafe = 1e-1
contact_dist = 0

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = dsafe, debug = False, ind0=0, ind1=1):
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types)

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.distance_from_obj(self.get_param_vector(t))
        self._debug = _debug

    def distance_from_obj(self, x):
        # self._cc.SetContactDistance(self.dsafe + .1)
        # Assuming x is aligned according to the following order:
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot
        # Parse the pose value
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pose, can_rot = x[20:23], x[23:]
        # Set pose of each rave body
        robot = self.params[self.ind0]
        obj = self.params[self.ind1]
        robot_body = self._param_to_body[robot]
        obj_body = self._param_to_body[obj]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        robot_body._set_active_dof_inds()
        obj_body.set_pose(can_pose, can_rot)
        # Make sure two body is in the samd environment
        assert robot_body.env_body.GetEnv() == obj_body.env_body.GetEnv()
        # Setup collision checkers
        self._cc.SetContactDistance(np.Inf)
        collisions = self._cc.BodyVsBody(robot_body.env_body, obj_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, obj_body, collisions)

        return col_val, col_jac

    def _calc_grad_and_val(self, robot_body, obj_body, collisions):

        # jac0 = np.zeros(2)
        # jac1 = np.zeros(2)
        vals = []
        robot_grads = []
        results = []
        for c in collisions:
            # import ipdb; ipdb.set_trace()
            linkA = c.GetLinkAName()
            linkB = c.GetLinkBName()
            linkAParent = c.GetLinkAParentName()
            linkBParent = c.GetLinkBParentName()

            linkRobot = None
            linkObj = None
            sign = 1
            if linkAParent == robot_body.name and linkBParent == obj_body.name:
                ptRobot = c.GetPtA()
                linkRobot = linkA
                sign = -1
                ptObj = c.GetPtB()
                linkObj = linkB
            elif linkBParent == robot_body.name and linkAParent == obj_body.name:
                ptRobot = c.GetPtB()
                linkRobot = linkB
                sign = 1
                ptObj = c.GetPtA()
                linkObj = linkA
            else:
                continue

            distance = c.GetDistance()
            normal = c.GetNormal()
            results.append((ptRobot, ptObj, distance))

            # plotting
            if self._debug:
                # ptRobot[2] = 1.01
                # ptObj[2] = 1.01
                self._plot_collision(ptRobot, ptObj, distance)
                print "pt0 = ", ptRobot
                print "pt1 = ", ptObj
                print "distance = ", distance
                print "normal = ", normal

            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)
            robot_grad = np.dot(sign * normal, robot_jac)
            col_vec = ptRobot - ptObj
            col_vec = col_vec / np.linalg.norm(col_vec)
            obj_jac = np.array([np.dot(col_vec, [1, 0, 0]), np.dot(col_vec, [0, 1, 0]), np.dot(col_vec, [0, 0, 1])])

            obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
            torque = ptObj - obj_pos[:3]
            rot_axies = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
            rot_vec = np.array([np.cross(rot_axies[0], torque), np.cross(rot_axies[1], torque), np.cross(rot_axies[2], torque)])
            obj_jac = np.c_[obj_jac, rot_vec]
            vals.append(self.dsafe - distance)
            robot_grads.append(robot_grad)


            # if there are multiple collisions, use the one with the greatest penetration distance
            # if self.dsafe - distance > val:
            #     chosen_pt0, chosen_pt1 = (pt0, pt1)
            #     chosen_distance = distance
            #     val = self.dsafe - distance
            #     jac0 = -1 * normal[0:2]
            #     jac1 = normal[0:2]

        # if self._debug:
        #     print "options: ", results
        #     print "selected: ", chosen_pt0, chosen_pt1
        #     self._plot_collision(chosen_pt0, chosen_pt1, chosen_distance)

        import ipdb; ipdb.set_trace()

        vals = np.vstack(vals)
        dof_inds = robot_body.dof_inds.tolist()
        robot_grads = np.vstack(robot_grads)[:, dof_inds]
        dim = len(vals)
        val = vals.reshape((dim, 1))
        jac = np.zeros((dim, 26))
        jac[:, range(3,20)] = robot_grads



        return vals, robot_grads

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(1, 0, 0)))
            else:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(0, 0, 0)))

class PosePredicate(ExprPredicate):

    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.05, debug = False, ind0=0, ind1=1):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        super(PosePredicate, self).__init__(name, e, attr_inds, params, expected_param_types)

    def pose_rot_check(self, x):
        # Assuming x is aligned according to the following order:
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot

        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pose, can_rotation = x[20:23], x[23:]
        # Setting pose for each ravebody
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pose, can_rotation)
        # Helper variables that will be used in many places
        obj_trans = obj_body.env_body.GetTransform()
        tool_link = robot.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pose, can_rotation)
        axises = [np.dot(Rz, np.dot(Ry, [1,0,0])), np.dot(Rz, [0,1,0]), [0,0,1]]# axises = [axis_x, axis_y, axis_z]

        # Two function calls return the value and jacobian of each constraints
        pos_val, pos_jac = self.pos_error(obj_trans, robot_trans, axises, arm_joints)
        rot_val, rot_jac = self.rot_error(obj_trans, robot_trans, axises, arm_joints)

        return pos_val, pos_jac, rot_val, rot_jac

    def ee_pose_check(self, x):
        # The x will be formulated to the following
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eeRot
        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        ee_pos, ee_rot = x[20:23], x[23:]
        # Setting pose for the robot
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        # Helper variables that will be used in many places
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        tool_link = robot.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [np.dot(Rz, np.dot(Ry, [1,0,0])), np.dot(Rz, [0,1,0]), [0,0,1]] # axises = [axis_x, axis_y, axis_z]

        pos_val, pos_jac = self.pos_error(obj_trans, robot_trans, axises, arm_joints)
        rot_val, rot_jac = self.rot_error(obj_trans, robot_trans, axises, arm_joints)

        return pos_val, pos_jac, rot_val, rot_jac

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        # Calculate the value and the jacobian regarding displacement between center of gripper and center of can
        gp = np.array([0,0,0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = robot_pos.flatten() - obj_pos.flatten()
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:,2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0],[0],[1]])
        # Calculate object jacobian
        obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((base_jac, torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), obj_jac))

        return (dist_val, dist_jac)

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints):
        # Calculate object transformation, and direction vectors
        local_dir = np.array([0.,0.,1.])
        obj_dir = np.dot(obj_trans[:3,:3], local_dir)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        rot_val = np.dot(obj_dir, world_dir) - 1
        # computing robot's jacobian
        arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir)) for joint in arm_joints]).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir)))
        base_jac = np.array([[0, 0, base_jac]])
        # computing object's jacobian
        obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
        obj_jac = np.r_[[0,0,0], obj_jac].reshape((1, 6))
        # Create final 1x26 jacobian matrix
        rot_jac = np.hstack((base_jac, np.zeros((1, 9)), arm_jac, np.zeros((1,1)), obj_jac))

        return (rot_val, rot_jac)

    def face_up(self, tool_link, arm_joints): # Not used
        # calculate the value and jacobian regarding direction of which the gripper is facing
        local_dir = np.array([0.,0.,1.])
        face_val = tool_link.GetTransform()[:2,:3].dot(local_dir)
        world_dir = tool_link.GetTransform()[:3,:3].dot(local_dir)
        arm_jac = np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()
        face_jac = np.hstack((np.zeros((2, 12)), arm_jac, np.zeros((2, 1)), np.zeros((2, 6))))

        return (face_val, face_jac)

class At(ExprPredicate):

    # At, Can, Target

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.can, self.target = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))]),
                                 (self.target, [("value", np.array([0,1,2], dtype=np.int)),
                                                  ("rotation", np.array([0,1,2], dtype=np.int))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class RobotAt(ExprPredicate):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.r, self.rp = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0,1,2], dtype=np.int)),
                                            ("backHeight", np.array([0], dtype=np.int)),
                                            ("lArmPose", np.array(range(7), dtype=np.int)),
                                            ("lGripper", np.array([0], dtype=np.int)),
                                            ("rArmPose", np.array(range(7), dtype=np.int)),
                                            ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.rp, [("value", np.array([0,1,2], dtype=np.int)),
                                             ("backHeight", np.array([0], dtype=np.int)),
                                             ("lArmPose", np.array(range(7), dtype=np.int)),
                                             ("lGripper", np.array([0], dtype=np.int)),
                                             ("rArmPose", np.array(range(7), dtype=np.int)),
                                             ("rGripper", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(20), -np.eye(20)]
        b ,val = np.zeros((20, 1)), np.zeros((20, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(RobotAt, self).__init__(name, e, attr_inds, params, expected_param_types)

class IsMP(ExprPredicate):

    # IsMP Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.robot, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))])])
        A = np.eye(40) - np.eye(40, k=20) - np.eye(40, k=-20)
        b = np.zeros((40,))
        val = np.vstack((BASE_MOVE*np.ones((3,1)), JOINT_MOVE*np.ones((17,1)), BASE_MOVE*np.ones((3,1)), JOINT_MOVE*np.ones((17,1)))).reshape((40,))
        e = LEqExpr(AffExpr(A, b), val)

        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class IsGP(PosePredicate):

    # IsGP, Robot, RobotPose, Can
    # This predicate only checks whether can pose is at center of gripper

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 3
        self._env = env
        self.robot, self.robot_pose, self.can = params
        attr_inds = OrderedDict([(self.robot_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                   ("backHeight", np.array([0], dtype=np.int)),
                                                   ("lArmPose", np.array(range(7), dtype=np.int)),
                                                   ("lGripper", np.array([0], dtype=np.int)),
                                                   ("rArmPose", np.array(range(7), dtype=np.int)),
                                                   ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[0]
        grad = lambda x: self.pose_rot_check(x)[1]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3, 1)))
        super(IsGP, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

class IsGPRot(PosePredicate):

    # IsGP, Robot, RobotPose, Can
    # This predicate checks whether can has the same rotation axis as that of gripper

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 3
        self._env = env
        self.robot, self.robot_pose, self.can = params
        attr_inds = OrderedDict([(self.robot_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                   ("backHeight", np.array([0], dtype=np.int)),
                                                   ("lArmPose", np.array(range(7), dtype=np.int)),
                                                   ("lGripper", np.array([0], dtype=np.int)),
                                                   ("rArmPose", np.array(range(7), dtype=np.int)),
                                                   ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[2]
        grad = lambda x: self.pose_rot_check(x)[3]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((2, 1)))
        super(IsGPRot, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

class IsPDP(PosePredicate):

    # IsPDP, Robot, RobotPose, Can, Target
    # This predicate only checks whether can pose is at center of gripper

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 4
        self._env = env
        self.robot, self.robot_pose, self.can, self.target = params
        attr_inds = OrderedDict([(self.robot_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                   ("backHeight", np.array([0], dtype=np.int)),
                                                   ("lArmPose", np.array(range(7), dtype=np.int)),
                                                   ("lGripper", np.array([0], dtype=np.int)),
                                                   ("rArmPose", np.array(range(7), dtype=np.int)),
                                                   ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.target, [("value", np.array([0,1,2], dtype=np.int)),
                                                  ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.target: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[0]
        grad = lambda x: self.pose_rot_check(x)[1]

        face_expr = Expr(f, grad)
        e = EqExpr(face_expr, np.zeros((3, 1)))
        super(IsPDP, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=3)

class IsPDPRot(PosePredicate):

    # IsPDP, Robot, RobotPose, Can, target
    # This predicate checks whether can has the same rotation axis as that of gripper

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 4
        self._env = env
        self.robot, self.robot_pose, self.can, self.target = params
        attr_inds = OrderedDict([(self.robot_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                   ("backHeight", np.array([0], dtype=np.int)),
                                                   ("lArmPose", np.array(range(7), dtype=np.int)),
                                                   ("lGripper", np.array([0], dtype=np.int)),
                                                   ("rArmPose", np.array(range(7), dtype=np.int)),
                                                   ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.target, [("value", np.array([0,1,2], dtype=np.int)),
                                                  ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.target: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[2]
        grad = lambda x: self.pose_rot_check(x)[3]

        face_expr = Expr(f, grad)
        e = EqExpr(face_expr, np.zeros((2, 1)))
        super(IsPDPRot, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=3)

class InGripper(PosePredicate):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.can = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[0]
        grad = lambda x: self.pose_rot_check(x)[1]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3,1)))
        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1)

class InGripperRot(PosePredicate):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.can = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.pose_rot_check(x)[2]
        grad = lambda x: self.pose_rot_check(x)[3]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((1,1)))
        super(InGripperRot, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1)

class GraspValid(PosePredicate):

    # GraspValid EEPose Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = OrderedDict([(self.ee_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                 ("rotation", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.target, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                ("rotation", np.array([0, 1, 2], dtype=np.int))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6,1)), np.zeros((6,1))
        e = AffExpr(A, b)
        e = EqExpr(e, val)

        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)

class EEReachable(PosePredicate):

    # EEUnreachable Robot, StartPose, EEPose
    # checks robot.getEEPose = EEPose

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 3
        self._env = env
        self.robot, self.start_pose, self.ee_pose = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.ee_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                 ("rotation", np.array([0, 1, 2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: self.ee_pose_check(x)[0]
        grad = lambda x: self.ee_pose_check(x)[1]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3,1)))
        super(EEReachable, self).__init__(name, e, attr_inds, params, expected_param_types)

class EEReachableRot(PosePredicate):

    # EEUnreachable Robot, StartPose, EEPose
    # checks robot.getEEPose = EEPose

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 3
        self._env = env
        self.robot, self.start_pose, self.ee_pose = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.ee_pose, [("value", np.array([0,1,2], dtype=np.int)),
                                                 ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: self.ee_pose_check(x)[2]
        grad = lambda x: self.ee_pose_check(x)[3]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((1,1)))
        super(EEReachableRot, self).__init__(name, e, attr_inds, params, expected_param_types)

class Stationary(ExprPredicate):

    # Stationary, Can

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.can,  = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryBase(ExprPredicate):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryArm(ExprPredicate):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = OrderedDict([(self.robot, [("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(16), -np.eye(16)]
        b, val = np.zeros((16, 1)), np.zeros((16, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArm, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryW(ExprPredicate):

    # StationaryW, Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

class StationaryNEQ(ExprPredicate):

    # StationaryNEq, Can, Can(Hold)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.can, self.can_held = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                             ("rotation", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.can_held, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                                  ("rotation", np.array([0, 1, 2], dtype=np.int))])])

        if self.c.name == self.c_held.name:
            A = np.zeros((1, 12))
            b = np.zeros((1, 1))
        else:
            A = np.c_[np.eye(6), -np.eye(6)]
            b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, dynamic=True)

# TODO Still in Namo predicate implementation
class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 3
        self._env = env
        self.robot, self.robot_pose, self.can = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.distance_from_obj(x)[0]
        grad_neg = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)

        super(Obstructs, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=2)

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class ObstructsHolding(CollisionPredicate):

    # ObstructsHolding, Robot, RobotPose, Can, Can;
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 4
        self._env = env
        r, rp, obstr, held = params
        self.r = r
        self.obstr = obstr
        self.held = held

        attr_inds = OrderedDict([(r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (obstr, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (held, [("pose", np.array([0, 1], dtype=np.int))])
                                 ])

        self._param_to_body = {r: self.lazy_spawn_or_body(r, r.name, r.geom),
                               obstr: self.lazy_spawn_or_body(obstr, obstr.name, obstr.geom),
                               held: self.lazy_spawn_or_body(held, held.name, held.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.distance_from_obj(x)[0]
        grad_neg = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(ObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types)

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def distance_from_obj(self, x):
        # x = [rpx, rpy, obstrx, obstry, heldx, heldy]
        self._cc.SetContactDistance(np.Inf)
        b0 = self._param_to_body[self.r]
        b1 = self._param_to_body[self.obstr]

        pose_r = x[0:2]
        pose_obstr = x[2:4]

        b0.set_pose(pose_r)
        b1.set_pose(pose_obstr)

        collisions1 = self._cc.BodyVsBody(b0.env_body, b1.env_body)
        col_val1, jac0, jac1 = self._calc_grad_and_val(self.r.name, self.obstr.name, pose_r, pose_obstr, collisions1)

        if self.obstr.name == self.held.name:
            ## add dsafe to col_val1 b/c we're allowed to touch, but not intersect
            col_val1 -= 2*self.dsafe
            val = np.array(col_val1)
            jac = np.r_[jac0, jac1].reshape((1, 4))

        else:
            b2 = self._param_to_body[self.held]
            pose_held = x[4:6]
            b2.set_pose(pose_held)

            collisions2 = self._cc.BodyVsBody(b2.env_body, b1.env_body)
            col_val2, jac2, jac1_ = self._calc_grad_and_val(self.held.name, self.obstr.name, pose_held, pose_obstr, collisions2)

            if col_val1 > col_val2:
                val = np.array(col_val1)
                jac = np.r_[jac0, jac1, np.zeros(2)].reshape((1, 6))
            else:
                val = np.array(col_val2)
                jac = np.r_[np.zeros(2), jac1_, jac2].reshape((1, 6))

        return val, jac


class Collides(CollisionPredicate):

    # Collides Can Wall(Obstacle)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.c, self.w = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        self._param_to_body = {self.c: self.lazy_spawn_or_body(self.c, self.c.name, self.c.geom),
                               self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.distance_from_obj(x)[0]
        def grad_neg(x):
            # print self.distance_from_obj(x)
            return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(Collides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)
        self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class RCollides(CollisionPredicate):

    # RCollides Robot Wall(Obstacle)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.r, self.w = params
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int))]),
                                 (self.w, [("pose", np.array([0, 1], dtype=np.int))])])
        self._param_to_body = {self.r: self.lazy_spawn_or_body(self.r, self.r.name, self.r.geom),
                               self.w: self.lazy_spawn_or_body(self.w, self.w.name, self.w.geom)}

        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        ## so we have an expr for the negated predicate
        def f_neg(x):
            d = self.distance_from_obj(x)[0]
            # if d > 0:
            #     import pdb; pdb.set_trace()
            #     self.distance_from_obj(x)
            return d

        def grad_neg(x):
            # print self.distance_from_obj(x)
            return -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, -val)


        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)

        self.priority = 1

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None
