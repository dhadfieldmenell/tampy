from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector3d, PR2PoseVector
from core.util_classes.viewer import OpenRAVEViewer
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2 import PR2
from core.util_classes.sampling import get_col_free_base_pose_around_target, \
    get_col_free_torso_arm_pose, get_random_theta
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
import numpy as np
import ctrajoptpy
import time, math
from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, IkFilterOptions
import openravepy
from ipdb import set_trace as st

"""
This file implements the classes for pr2 domain specific predicates
"""
BASE_MOVE = 1e0
IN_GRIPPER_COEFF = 1.

EEREACHABLE_COEFF = 1e0
EEREACHABLE_OPT_COEFF = 1e3
EEREACHABLE_ROT_OPT_COEFF = 3e2
INGRIPPER_OPT_COEFF = 3e2
RCOLLIDES_OPT_COEFF = 1e2
OBSTRUCTS_OPT_COEFF = 1e2

GRASP_VALID_COEFF = 1e1
dsafe = 1e-2
contact_dist = 0
can_radius = 0.04
COLLISION_TOL = 1e-2
POSE_TOL = 2e-2

ROBOT_LINKS = 45

TABLE_SAMPLING_RADIUS = 2.0
OBJ_RING_SAMPLING_RADIUS = 0.6

APPROACH_DIST = 0.05
RETREAT_DIST = 0.075

GRIPPER_OPEN_VALUE = 0.5
GRIPPER_CLOSE_VALUE = 0.46

NUM_EEREACHABLE_RESAMPLE_ATTEMPTS = 10

EEREACHABLE_STEPS = 3

MAX_CONTACT_DISTANCE = .1

class CollisionPredicate(ExprPredicate):

    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = dsafe, debug = False, ind0=0, ind1=1, tol=COLLISION_TOL):
        self._debug = debug
        # if self._debug:
        #     self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self._plot_handles = []
        self._cache = {}
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, tol=tol)

    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.robot_obj_collision(self.get_param_vector(t))
        self._debug = _debug

    def plot_collision(self, ptA, ptB, distance):
        handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.001,color=(1,0,0)))
            else:
                handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.001,color=(0,0,0)))
        self._plot_handles.extend(handles)

    def robot_obj_collision(self, x):
        """
            This function is used to calculae collisiosn between Robot and Can
            This function calculates the collision distance gradient associated to it
            x: 26 dimensional list of values aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot
        """
        # Parse the pose value
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        if flattened in self._cache and not self._debug:
            return self._cache[flattened]

        back_height = x[0:1]
        l_arm_pose, l_gripper = x[1:8], x[8:9]
        r_arm_pose, r_gripper = x[9:16], x[16:17]
        base_pose = x[17:20]
        can_pos, can_rot = x[20:23], x[23:]
        # Set pose of each rave body
        robot = self.params[self.ind0]
        obj = self.params[self.ind1]
        robot_body = self._param_to_body[robot]
        obj_body = self._param_to_body[obj]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        robot_body._set_active_dof_inds()
        obj_body.set_pose(can_pos, can_rot)
        # Make sure two body is in the same environment
        assert robot_body.env_body.GetEnv() == obj_body.env_body.GetEnv()
        # Setup collision checkers
        self._cc.SetContactDistance(MAX_CONTACT_DISTANCE)
        collisions = self._cc.BodyVsBody(robot_body.env_body, obj_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, obj_body, collisions)
        # set active dof value back to its original state (For successive function call)
        robot_body._set_active_dof_inds(range(39))
        self._cache[flattened] = (col_val.copy(), col_jac.copy())
        return col_val, col_jac

    def obj_obj_collision(self, x):
        """
            This function calculates collision between object and obstructs
            Assuming object and obstructs all has pose and rotation

            x: 12 dimensional list aligned in the following order:
            CanPose->CanRot->ObstaclePose->ObstacleRot
        """
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        if flattened in self._cache and not self._debug:
            return self._cache[flattened]

        # Parse the pose value
        can_pos, can_rot = x[:3], x[3:6]
        obstr_pos, obstr_rot = x[6:9], x[9:]
        # Set pose of each rave body
        can = self.params[self.ind0]
        obstr = self.params[self.ind1]
        can_body = self._param_to_body[can]
        obstr_body = self._param_to_body[obstr]
        can_body.set_pose(can_pos, can_rot)
        obstr_body.set_pose(obstr_pos, obstr_rot)
        # Make sure two body is in the same environment
        assert can_body.env_body.GetEnv() == obstr_body.env_body.GetEnv()
        # Setup collision checkers
        # self._cc.SetContactDistance(MAX_CONTACT_DISTANCE)
        self._cc.SetContactDistance(np.inf)
        collisions = self._cc.BodyVsBody(can_body.env_body, obstr_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_obj_grad_and_val(can_body, obstr_body, collisions)
        self._cache[flattened] = (col_val.copy(), col_jac.copy())
        return col_val, col_jac

    def robot_obj_held_collision(self, x):
        """
            Similar to robot_obj_collision in CollisionPredicate; however, this function take into account of object holding

            x: 26 dimensional list of values aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot->HeldPose->HeldRot
        """
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        if flattened in self._cache and not self._debug:
            return self._cache[flattened]
        # Parse the pose value
        # obj_body -> self.obstruct
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        can_pos, can_rot = x[20:23], x[23:26]
        held_pose, held_rot = x[26:29], x[29:]
        # Set pose of each rave body
        robot = self.params[self.ind0]
        obj = self.params[self.ind1]
        robot_body = self._param_to_body[robot]
        obj_body = self._param_to_body[obj]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        robot_body._set_active_dof_inds()
        obj_body.set_pose(can_pos, can_rot)
        self._cc.SetContactDistance(MAX_CONTACT_DISTANCE)
        # setup collision between robot and obstruct
        collisions1 = self._cc.BodyVsBody(robot_body.env_body, obj_body.env_body)
        col_val1, col_jac1 = self._calc_grad_and_val(robot_body, obj_body, collisions1)
        num_links = len(robot.geom.col_links)
        col_jac1 = np.c_[col_jac1, np.zeros((num_links,6))]
        # find collision between object and object held
        held_body = self._param_to_body[self.held]
        held_body.set_pose(held_pose, held_rot)
        self._cc.SetContactDistance(np.inf)
        collisions2 = self._cc.BodyVsBody(held_body.env_body, obj_body.env_body)
        col_val2, col_jac2 = self._calc_obj_grad_and_val(held_body, obj_body, collisions2)
        col_jac2 = np.c_[np.zeros((1,20)), col_jac2]
        # Stack these val and jac, and return
        val = np.vstack((col_val1, col_val2))
        jac = np.vstack((col_jac1, col_jac2))
        robot_body._set_active_dof_inds(range(39))
        self._cache[flattened] = (val.copy(), jac.copy())
        return val, jac

    def _calc_grad_and_val(self, robot_body, obj_body, collisions):
        """
            This function is helper function of robot_obj_collision(self, x)
            It calculates collision distance and gradient between each robot's link and object

            robot_body: OpenRAVEBody containing body information of pr2 robot
            obj_body: OpenRAVEBody containing body information of object
            collisions: list of collision objects returned by collision checker
        """
        # Initialization
        links = []
        robot = self.params[self.ind0]
        col_links = robot.geom.col_links
        num_links = len(col_links)
        obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
        rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkRobot, linkObj = None, None
            sign = 0
            if linkAParent == robot_body.name and linkBParent == obj_body.name:
                ptRobot, ptObj = c.GetPtA(), c.GetPtB()
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == robot_body.name and linkAParent == obj_body.name:
                ptRobot, ptObj = c.GetPtB(), c.GetPtA()
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue
            if linkRobot not in col_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)
            grad = np.zeros((1,26))
            grad[:, :20] = np.dot(sign * normal, robot_jac)
            # robot_grad = np.dot(sign * normal, robot_jac).reshape((1,20))
            col_vec = -sign*normal
            # Calculate object pose jacobian
            # obj_jac = np.array([-sign*normal])
            grad[:, 20:23] = np.array([-sign*normal])
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, 23:36] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            # links.append((robot_link_ind, self.dsafe - distance, robot_grad, robot.GetLink(linkRobot)))
            links.append((robot_link_ind, self.dsafe - distance, grad, robot.GetLink(linkRobot)))

            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        # arrange gradients in proper link order
        max_dist = self.dsafe - MAX_CONTACT_DISTANCE
        vals, robot_grads = max_dist*np.ones((num_links,1)), np.zeros((num_links,26))
        links = sorted(links, key = lambda x: x[0])
        vals[:len(links),0] = np.array([link[1] for link in links])
        robot_grads[:len(links), range(26)] = np.array([link[2] for link in links]).reshape((len(links), 26))
        # TODO: remove line below which was added for debugging purposes
        self.links = links
        return vals, robot_grads

    def _calc_obj_grad_and_val(self, obj_body, obstr_body, collisions):
        """
            This function is helper function of robot_obj_collision(self, x) #Used in ObstructsHolding#
            It calculates collision distance and gradient between each robot's link and obstr,
            and between held object and obstr

            obj_body: OpenRAVEBody containing body information of object
            obstr_body: OpenRAVEBody containing body information of obstruction
            collisions: list of collision objects returned by collision checker
        """
        # Initialization
        vals = []
        grads = []
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkObj, linkObstr = None, None
            sign = 1
            if linkAParent == obj_body.name and linkBParent == obstr_body.name:
                ptObj, ptObstr = c.GetPtA(), c.GetPtB()
                linkObj, linkObstr = linkA, linkB
                sign = -1
            elif linkBParent == obj_body.name and linkAParent == obstr_body.name:
                ptObj, ptObstr = c.GetPtB(), c.GetPtA()
                linkObj, linkObstr = linkB, linkA
                sign = 1
            else:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            col_vec = -sign*normal
            # Calculate object pose jacobian
            obj_jac = np.array([normal])
            obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            obj_jac = np.c_[obj_jac, -rot_vec]
            # Calculate obstruct pose jacobian
            obstr_jac = np.array([-normal])
            obstr_pos = OpenRAVEBody.obj_pose_from_transform(obstr_body.env_body.GetTransform())
            torque = ptObstr - obstr_pos[:3]
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obstr_pos[:3], obstr_pos[3:])
            rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            obstr_jac = np.c_[obstr_jac, rot_vec]
            # Constructing gradient matrix
            robot_grad = np.c_[obj_jac, obstr_jac]
            vals.append(self.dsafe - distance)
            grads.append(robot_grad)

            if self._debug:
                self.plot_collision(ptObj, ptObstr, distance)

        vals = np.vstack(vals)
        grads = np.vstack(grads)
        ind = np.argmax(vals)
        val = vals[ind].reshape((1,1))
        grad = grads[ind].reshape((1,12))

        return val, grad

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(1, 0, 0)))
            else:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(0, 0, 0)))

    def test(self, time, negated=False):
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.neg_expr.eval(self.get_param_vector(time), tol=self.tol, negated = (not negated))
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

class PosePredicate(ExprPredicate):

    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.05, debug = False, ind0=0, ind1=1, tol=POSE_TOL, active_range=(0,0)):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self.handle = []
        super(PosePredicate, self).__init__(name, e, attr_inds, params, expected_param_types, tol=tol, active_range=active_range)

    def pose_check(self, x):
        """
            This function is used to check whether:
                object is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot
        """
        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pos, can_rot = x[20:23], x[23:]
        # Setting pose for each ravebody
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pos, can_rot)
        # Helper variables that will be used in many places
        obj_trans = obj_body.env_body.GetTransform()
        tool_link = robot.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pos, can_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]# axises = [axis_z, axis_y, axis_x]
        pos_val, pos_jac = self.pos_error(obj_trans, robot_trans, axises, arm_joints)

        return pos_val, pos_jac

    def rot_check(self, x):
        """
            This function is used to check whether:
                object's rotational axis is parallel to that of robot gripper

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot
        """
        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pos, can_rot = x[20:23], x[23:]
        # Setting pose for each ravebody
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pos, can_rot)
        # Helper variables that will be used in many places
        obj_trans = obj_body.env_body.GetTransform()
        tool_link = robot.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pos, can_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]# axises = [axis_z, axis_y, axis_x]
        rot_val, rot_jac = self.rot_error(obj_trans, robot_trans, axises, arm_joints)

        return rot_val, rot_jac

    def ee_pose_check(self, x):
        """
            This function is used to check whether:
                End effective pose's position is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eRot
        """
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
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        pos_val, pos_jac = self.pos_error(obj_trans, robot_trans, axises, arm_joints)

        return pos_val, pos_jac

    def ee_rot_check(self, x):
        """
            This function is used to check whether:
                End effective pose's rotational axis is parallel to that of robot gripper

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eRot
        """
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
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        rot_val, rot_jac = self.rot_lock(obj_trans, robot_trans, axises, arm_joints)

        return rot_val, rot_jac

    def ee_targ_rot_check(self, x):
        """
            This function is used to check whether:
                End effective pose's rotational axis is parallel to that of the target

            x: 12 dimensional list aligned in following order,
            eePose->eeRot->targPose->targRot
        """
        # Parse the pose values
        ee_pos, ee_rot = x[:3], x[3:6]
        targ_pos, targ_rot = x[6:9], x[9:]
        # Calculate ee_pose and target's transform
        local_dir = np.array([0.,0.,1.])
        ee_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos.reshape((3,)), ee_rot.reshape((3,)))[:3,:3]
        targ_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos.reshape((3,)), targ_rot.reshape((3,)))[:3,:3]
        # Calculate ee_pose and target's direction
        ee_dir = np.dot(ee_trans, local_dir)
        targ_dir = np.dot(targ_trans, local_dir)
        rot_val = np.array([[np.dot(ee_dir, targ_dir) - 1]])
        # Obatin the axises of ee_pose and target
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        ee_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(targ_pos, targ_rot)
        targ_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]
        # Calculate rotational jacobian
        ee_jac = np.array([np.dot(targ_dir, np.cross(axis, ee_dir)) for axis in ee_axises])
        ee_jac = np.r_[[0,0,0], ee_jac].reshape((1, 6))
        targ_jac = np.array([np.dot(ee_dir, np.cross(axis, targ_dir)) for axis in targ_axises])
        targ_jac = np.r_[[0,0,0], targ_jac].reshape((1, 6))
        rot_jac = np.c_[ee_jac, targ_jac]

        return rot_val, rot_jac

    def ee_pose_check_rel_obj(self, x, rel_pt):
        """
            This function is used to check whether:
                End effective pose's position is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eRot
        """
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
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        pos_val, pos_jac = self.pos_error_rel_to_obj(obj_trans, robot_trans, axises, arm_joints, rel_pt)

        return pos_val, pos_jac

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
            This function calculates the value and the jacobian of the displacement between center of gripper and center of object

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        gp = np.array([0,0,0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3,1))
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

    def pos_error_rel_to_obj(self, obj_trans, robot_trans, axises, arm_joints, rel_pt):
        """
            This function calculates the value and the jacobian of the displacement between center of gripper and a point relative to the object

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3,1))
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:,2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0],[0],[1]])
        # Calculate object jacobian
        # obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = -1*np.array([np.cross(axis, obj_pos - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((base_jac, torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), obj_jac))

        return (dist_val, dist_jac)

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
            This function calculates the value and the jacobian of the rotational error between
            robot gripper's rotational axis and object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        local_dir = np.array([0.,0.,1.])
        obj_dir = np.dot(obj_trans[:3,:3], local_dir)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        obj_dir = obj_dir/np.linalg.norm(obj_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        rot_val = np.array([[np.dot(obj_dir, world_dir) - 1]])
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

    def rot_lock(self, obj_trans, robot_trans, axises, arm_joints):
        """
            This function calculates the value and the jacobian of the rotational error between
            robot gripper's rotational axis and object's rotational axis

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        rot_vals = []
        rot_jacs = []
        for local_dir in np.eye(3):
            obj_dir = np.dot(obj_trans[:3,:3], local_dir)
            world_dir = robot_trans[:3,:3].dot(local_dir)
            rot_vals.append(np.array([[np.dot(obj_dir, world_dir) - 1]]))
            # computing robot's jacobian
            arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir)) for joint in arm_joints]).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir)))
            base_jac = np.array([[0, 0, base_jac]])
            # computing object's jacobian
            obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
            obj_jac = np.r_[[0,0,0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix
            rot_jacs.append(np.hstack((base_jac, np.zeros((1, 9)), arm_jac, np.zeros((1,1)), obj_jac)))

        rot_val = np.vstack(rot_vals)
        rot_jac = np.vstack(rot_jacs)

        return (rot_val, rot_jac)

    def face_up(self, tool_link, arm_joints):
        """
            This function checks whether robot gripper is facing up
            This function is not used in any predicates below

            tool_link: link of the robot right gripper
            arm_joints: list of robot joints
        """
        # calculate the value and jacobian regarding direction of which the gripper is facing
        local_dir = np.array([0.,0.,1.])
        face_val = tool_link.GetTransform()[:2,:3].dot(local_dir)
        world_dir = tool_link.GetTransform()[:3,:3].dot(local_dir)
        arm_jac = np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()
        face_jac = np.hstack((np.zeros((2, 12)), arm_jac, np.zeros((2, 1)), np.zeros((2, 6))))

        return (face_val, face_jac)

    def finger_pose_check(self, x):
        """
            This function checks whether finger of robot gripper is touching the object or Not
            val is defined between center of the gripper and left finger tips
            (assuming can is at the center of robot gripper already)

            x: list of 26 dimensional values aligned in following order:
            # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper
        """

        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        # Setting pose for each ravebody
        robot_body = self._param_to_body[self.params[self.ind0]]
        target = self.params[self.ind1]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)

        # Get coordinate points in the finger tips
        l_finger = robot.GetLink("r_gripper_l_finger_tip_link")
        l_finger_pos = OpenRAVEBody.obj_pose_from_transform(l_finger.GetTransform())[:3]
        grip_pos = OpenRAVEBody.obj_pose_from_transform(robot.GetLink("r_gripper_tool_frame").GetTransform())[:3]
        l_tip = l_finger_pos + np.array([0.0205 ,  0.0083,  0.])

        # r_finger = robot.GetLink("r_gripper_r_finger_tip_link")
        # r_finger_pos = OpenRAVEBody.obj_pose_from_transform(r_finger.GetTransform())[:3]
        # r_tip = r_finger_pos + np.array([-0.0087 ,  0.0205,  0.])

        # val corresponds distance between right finger tip to center of gripper
        val = np.linalg.norm(grip_pos - l_tip) - can_radius

        finger_dir =  l_tip - grip_pos.reshape((1,3))
        finger_dir = finger_dir/np.linalg.norm(finger_dir)
        finger_joint = robot.GetJoint('r_gripper_l_finger_joint')
        arm_jac = np.array([np.dot(finger_dir, np.cross(finger_joint.GetAxis(), l_finger_pos.flatten() - finger_joint.GetAnchor()))])
        jac = np.c_[np.zeros((1,19)), arm_jac]

        # self.handle = []
        # self.handle.append(self._env.drawarrow(p1=grip_pos, p2=l_tip, linewidth=.001,color=(1,0,0)))
        # self.handle.append(self._env.drawarrow(p1=r_tip, p2=finger_joint.GetAnchor(), linewidth=.001,color=(0,1,0)))
        # import ipdb; ipdb.set_trace()
        # print val

        return val, jac

    def set_gripper_value(self, x, negated = False):

        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        grab_pose = 0.46
        open_pose = 0.5
        if negated:
            r_gripper = open_pose
        else:
            r_gripper = grab_pose

        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)

        return np.zeros((1,1)), np.zeros((1,20))

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
        self.robot, self.robot_pose = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0,1,2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.robot_pose, [("value", np.array([0,1,2], dtype=np.int)),
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
        self._env = env
        self.robot, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((17,1))
        active_lb = lb_limit[dof_inds].reshape((17,1))
        joint_move = (active_ub-active_lb)/100
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/10
        val = np.vstack((BASE_MOVE*np.ones((3,1)), joint_move, BASE_MOVE*np.ones((3,1)), joint_move))
        A = np.eye(40) - np.eye(40, k=20) - np.eye(40, k=-20)
        b = np.zeros((40,1))
        robot_body._set_active_dof_inds(range(39))
        # Setting attributes for testing
        self.base_step = BASE_MOVE*np.ones((3,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb

        return A, b, val

class WithinJointLimit(ExprPredicate):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                               ("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(WithinJointLimit, self).__init__(name, e, attr_inds, params, expected_param_types)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((17,1))
        active_lb = lb_limit[dof_inds].reshape((17,1))
        joint_move = (active_ub-active_lb)/10
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = np.hstack((np.zeros((17, 3)), -np.eye(17)))
        A_up_limit = np.hstack((np.zeros((17,3)), np.eye(17)))
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((34,1))
        robot_body._set_active_dof_inds(range(39))
        # Setting attributes for testing
        self.base_step = BASE_MOVE*np.ones((3,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

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
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class StationaryBase(ExprPredicate):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0, 1, 2], dtype=np.int))])])

        N_DIM=3

        A = np.c_[np.eye(N_DIM), -np.eye(N_DIM)]
        b, val = np.zeros((N_DIM, 1)), np.zeros((N_DIM, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class StationaryArms(ExprPredicate):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = OrderedDict([(self.robot, [("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int))])])

        N_DIM=16

        A = np.c_[np.eye(N_DIM), -np.eye(N_DIM)]
        b, val = np.zeros((N_DIM, 1)), np.zeros((N_DIM, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArms, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class StationaryW(ExprPredicate):

    # StationaryW, Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                           ("rotation", np.array([0, 1, 2], dtype=np.int))])])
        A = np.c_[np.eye(6), -np.eye(6)]
        b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class StationaryNEq(ExprPredicate):

    # StationaryNEq, Can, Can(Hold)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.can, self.can_held = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                             ("rotation", np.array([0, 1, 2], dtype=np.int))])])

        if self.can.name == self.can_held.name:
            A = np.zeros((1, 12))
            b = np.zeros((1, 1))
        else:
            A = np.c_[np.eye(6), -np.eye(6)]
            b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))

class GraspValid(PosePredicate):

    # GraspValid EEPose Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = OrderedDict([(self.ee_pose, [("value", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.target, [("value", np.array([0, 1, 2], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3,1)), np.zeros((3,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)

class GraspValidRot(PosePredicate):

    # GraspValid EEPose Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = OrderedDict([(self.ee_pose, [("rotation", np.array([1, 2], dtype=np.int))]),
                                 (self.target, [("rotation", np.array([1, 2], dtype=np.int))])])

        # f = lambda x: GRASP_VALID_COEFF*self.ee_targ_rot_check(x)[0]
        # grad = lambda x: GRASP_VALID_COEFF*self.ee_targ_rot_check(x)[1]

        # A = np.c_[np.eye(2), -np.eye(2)]
        A = np.eye(4)
        # b, val = np.zeros((2,1)), np.zeros((2,1))
        b, val = np.zeros((4,1)), np.zeros((4,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        # pos_expr, val = Expr(f, grad), np.zeros((1,1))
        # e = EqExpr(pos_expr, val)
        super(GraspValidRot, self).__init__(name, e, attr_inds, params, expected_param_types)

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

        f = lambda x: IN_GRIPPER_COEFF*self.pose_check(x)[0]
        grad = lambda x: IN_GRIPPER_COEFF*self.pose_check(x)[1]

        self.opt_expr = EqExpr(Expr(lambda x: INGRIPPER_OPT_COEFF * f(x),
                                    lambda x: INGRIPPER_OPT_COEFF*grad(x)),
                                np.zeros((1,1)))


        pos_expr, val = Expr(f, grad), np.zeros((3,1))
        e = EqExpr(pos_expr, val)
        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1)

    def get_expr(self, negated):
        if negated:
            return None
        return self.opt_expr

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

        f = lambda x: IN_GRIPPER_COEFF*self.rot_check(x)[0]
        grad = lambda x: IN_GRIPPER_COEFF*self.rot_check(x)[1]

        pos_expr, val = Expr(f, grad), np.zeros((1,1))
        e = EqExpr(pos_expr, val)

        self.opt_expr = EqExpr(Expr(lambda x: INGRIPPER_OPT_COEFF*f(x),
                                    lambda x: INGRIPPER_OPT_COEFF*grad(x)),
                                np.zeros((1,1)))



        super(InGripperRot, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1)
        # self.priority = 2

    def get_expr(self, negated):
        if negated:
            return None
        return self.opt_expr

class InContact(ExprPredicate):
    # InContact robot EEPose target
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.ee_pose, self.target = params
        attr_inds = OrderedDict([(self.robot, [("rGripper", np.array([0], dtype=np.int))])])

        A = np.eye(1).reshape((1,1))
        b = np.zeros(1).reshape((1,1))

        val = np.array([[GRIPPER_CLOSE_VALUE]])
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = np.array([[GRIPPER_OPEN_VALUE]])
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types)


class InContact2(PosePredicate):
    # InContact robot EEPose target
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.ee_pose, self.target = params
        attr_inds = OrderedDict([(self.robot, [("pose", np.array([0,1,2], dtype=np.int)),
                                            ("backHeight", np.array([0], dtype=np.int)),
                                            ("lArmPose", np.array(range(7), dtype=np.int)),
                                            ("lGripper", np.array([0], dtype=np.int)),
                                            ("rArmPose", np.array(range(7), dtype=np.int)),
                                            ("rGripper", np.array([0], dtype=np.int))]),
                                 (self.target, [("value", np.array([0, 1, 2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: self.finger_pose_check(x)[0]
        grad = lambda x: self.finger_pose_check(x)[1]

        fing_expr, val = Expr(f, grad), np.zeros((1,1))
        e = EqExpr(fing_expr, val)
        super(InContact2, self).__init__(name, e, attr_inds, params, expected_param_types, ind0 = 0, ind1 = 2)

    def finger_pose_check(self, x):

        """
            This function checks whether finger of robot gripper is touching the object or Not
            val is defined between center of the gripper and left finger tips
            (assuming can is at the center of robot gripper already)

            x: list of 26 dimensional values aligned in following order:
            # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->targetPose
        """

        # Parse the pose values
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        target_pose = x[20:23].reshape((1,3))
        # Setting pose for each ravebody
        robot_body = self._param_to_body[self.params[self.ind0]]
        target = self.params[self.ind1]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)

        # Get coordinate points in the finger tips
        l_finger = robot.GetLink("r_gripper_l_finger_tip_link")
        l_finger_pos = OpenRAVEBody.obj_pose_from_transform(l_finger.GetTransform())[:3]
        grip_pos = OpenRAVEBody.obj_pose_from_transform(robot.GetLink("r_gripper_tool_frame").GetTransform())[:3]
        l_tip = l_finger_pos + np.array([0.0205 ,  0.0083,  0.])

        # val corresponds distance between right finger tip to center of gripper
        val = np.linalg.norm(target_pose - l_tip) - can_radius

        finger_dir =  l_tip - target_pose.reshape((1,3))
        finger_dir = finger_dir/np.linalg.norm(finger_dir)
        finger_joint = robot.GetJoint('r_gripper_l_finger_joint')
        arm_jac = np.array([np.dot(finger_dir, np.cross(finger_joint.GetAxis(), l_finger_pos.flatten() - finger_joint.GetAnchor()))])
        obj_jac = np.array([np.dot(finger_dir, axis) for axis in np.eye(3)]).reshape((1,3))
        # import ipdb; ipdb.set_trace()
        jac = np.c_[np.zeros((1,19)), arm_jac, obj_jac]

        # self.handle = []
        # self.handle.append(self._env.drawarrow(p1=grip_pos, p2=l_tip, linewidth=.001,color=(1,0,0)))
        # self.handle.append(self._env.drawarrow(p1=r_tip, p2=finger_joint.GetAnchor(), linewidth=.001,color=(0,1,0)))
        # import ipdb; ipdb.set_trace()
        # print val

        return val, jac

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

        self._steps = EEREACHABLE_STEPS
        self._dim = 26
        # f = lambda x: EEREACHABLE_COEFF*self.ee_pose_check(x)[0]
        # grad = lambda x: EEREACHABLE_COEFF*self.ee_pose_check(x)[1]
        f = lambda x: self.stacked_f(x)
        grad = lambda x: self.stacked_grad(x)

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3*(2*self._steps+1),1)))
        self.opt_expr = EqExpr(Expr(lambda x: EEREACHABLE_OPT_COEFF*f(x),
                                    lambda x: EEREACHABLE_OPT_COEFF*grad(x)),
                                np.zeros((1,1)))
        super(EEReachable, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(-self._steps, self._steps))

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([0, 0, RETREAT_DIST])

    def stacked_f(self, x):
        i = 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(self.ee_pose_check_rel_obj(x[i:i+self._dim], rel_pt)[0])
            i += self._dim
        return np.vstack(tuple(f_res))

    def stacked_grad(self, x):
        f_grad = []
        start, end = self.active_range
        t = (2*self._steps+1)
        k = 3

        grad = np.zeros((k*t, self._dim*t))
        i = 0
        j = 0
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            grad[j:j+k, i:i+self._dim] = self.ee_pose_check_rel_obj(x[i:i+self._dim], rel_pt)[1]
            i += self._dim
            j += k
        return grad

    def get_expr(self, negated):
        if negated:
            return None
        else:
            return self.opt_expr

    def resample(self, negated, t, plan):
        return ee_reachable_resample(self, negated, t, plan)

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
                                 (self.ee_pose, [("value", np.array([0, 1, 2], dtype=np.int)),
                                                 ("rotation", np.array([0, 1, 2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: EEREACHABLE_COEFF*self.ee_rot_check(x)[0]
        grad = lambda x: EEREACHABLE_COEFF*self.ee_rot_check(x)[1]

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3,1)))
        self.opt_expr = EqExpr(Expr(lambda x: EEREACHABLE_ROT_OPT_COEFF * f(x),
                                    lambda x: EEREACHABLE_ROT_OPT_COEFF*grad(x)),
                                np.zeros((1,1)))
        super(EEReachableRot, self).__init__(name, e, attr_inds, params, expected_param_types)

    def get_expr(self, negated):
        if negated:
            return None
        else:
            return self.opt_expr

    def resample(self, negated, t, plan):
        return ee_reachable_resample(self, negated, t, plan)

class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=COLLISION_TOL):
        assert len(params) == 4
        self._env = env
        self.robot, self.startp, self.endp, self.can = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = OrderedDict([(self.robot, [("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int)),
                                               ("pose", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}
        # self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),

        f = lambda x: -self.robot_obj_collision(x)[0]
        grad = lambda x: -self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)
        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Obstructs, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=3, debug=debug, tol=tol)
        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

    def resample(self, negated, t, plan):
        target_pose = self.can.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose, dist=OBJ_RING_SAMPLING_RADIUS)

class ObstructsHolding(CollisionPredicate):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 5
        self._env = env
        self.robot, self.startp, self.endp, self.obstruct, self.held = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = OrderedDict([(self.robot, [("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int)),
                                               ("pose", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.obstruct, [("pose", np.array([0,1,2], dtype=np.int)),
                                                  ("rotation", np.array([0,1,2], dtype=np.int))]),
                                 (self.held, [("pose", np.array([0,1,2], dtype=np.int)),
                                              ("rotation", np.array([0,1,2], dtype=np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstruct: self.lazy_spawn_or_body(self.obstruct, self.obstruct.name, self.obstruct.geom),
                               self.held: self.lazy_spawn_or_body(self.held, self.held.name, self.held.geom)}

        links = len(self.robot.geom.col_links)
        if self.held.name == self.obstruct.name:
            f = lambda x: -self.robot_obj_collision(x)[0] + self.dsafe - 1e-3
            grad = lambda x: -self.robot_obj_collision(x)[1]
            ## so we have an expr for the negated predicate
            f_neg = lambda x: self.robot_obj_collision(x)[0] - self.dsafe + 1e-3
            grad_neg = lambda x: self.robot_obj_collision(x)[1]
            val = np.zeros((links,1))
        else:
            f = lambda x: -self.robot_obj_held_collision(x)[0]
            grad = lambda x: -self.robot_obj_held_collision(x)[1]
            ## so we have an expr for the negated predicate
            f_neg = lambda x: self.robot_obj_held_collision(x)[0]
            grad_neg = lambda x: self.robot_obj_held_collision(x)[1]
            val = np.zeros((links+1,1))

        col_expr, col_expr_neg = Expr(f, grad), Expr(f_neg, grad_neg)
        e, self.neg_expr = LEqExpr(col_expr, val), LEqExpr(col_expr_neg, val)
        self.neg_expr_opt = LEqExpr(get_expr_mult(OBSTRUCTS_OPT_COEFF, col_expr_neg), val)
        super(ObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3, debug = debug)

        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr_opt
        else:
            return None

    def resample(self, negated, t, plan):
        target_pose = self.obstruct.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose,
                                        dist=OBJ_RING_SAMPLING_RADIUS)


class Collides(CollisionPredicate):

    # Collides Can Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.can, self.obstacle = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                             ("rotation", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.obstacle, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                                  ("rotation", np.array([0, 1, 2], dtype=np.int))])])
        self._param_to_body = {self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        f = lambda x: -self.obj_obj_collision(x)[0]
        grad = lambda x: -self.obj_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.obj_obj_collision(x)[0]
        grad_neg = lambda x: self.obj_obj_collision(x)[1]

        col_expr, val = Expr(f, grad), np.zeros((1,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Collides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1, debug=debug)
        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class RCollides(CollisionPredicate):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.obstacle = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = OrderedDict([(self.robot, [("backHeight", np.array([0], dtype=np.int)),
                                               ("lArmPose", np.array(range(7), dtype=np.int)),
                                               ("lGripper", np.array([0], dtype=np.int)),
                                               ("rArmPose", np.array(range(7), dtype=np.int)),
                                               ("rGripper", np.array([0], dtype=np.int)),
                                               ("pose", np.array([0, 1, 2], dtype=np.int))]),
                                 (self.obstacle, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                                  ("rotation", np.array([0, 1, 2], dtype = np.int))])])

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        f = lambda x: -self.robot_obj_collision(x)[0]
        grad = lambda x: -self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.robot_obj_collision(x)[1]

        f_neg_opt = lambda x: RCOLLIDES_OPT_COEFF*self.robot_obj_collision(x)[0]
        grad_neg_opt = lambda x: RCOLLIDES_OPT_COEFF*self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)
        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        col_expr_neg_opt = Expr(f_neg_opt, grad_neg_opt)
        self.neg_expr = LEqExpr(col_expr_neg, -val)
        self.neg_expr_opt = LEqExpr(col_expr_neg_opt, val)

        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)

        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr_opt
        else:
            return None

    def resample(self, negated, t, plan):
        target_pose = self.obstacle.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose,
                                        dist=TABLE_SAMPLING_RADIUS)

def resample_bp_around_target(pred, t, plan, target_pose, dist=OBJ_RING_SAMPLING_RADIUS):
    v = OpenRAVEViewer.create_viewer()

    bp = get_col_free_base_pose_around_target(t, plan, target_pose, pred.robot,
                                        save=True, dist=dist)
    v.draw_plan_ts(plan, t)

    attr_inds = OrderedDict()
    res = []
    robot_attr_name_val_tuples = [('pose', bp)]
    add_to_attr_inds_and_res(t, attr_inds, res, pred.robot,
                            robot_attr_name_val_tuples)
    return np.array(res), attr_inds

def get_expr_mult(coeff, expr):
    new_f = lambda x: coeff*expr.eval(x)
    new_grad = lambda x: coeff*expr.grad(x)
    return Expr(new_f, new_grad)

def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        res.extend(val[inds].flatten().tolist())
        param_attr_inds.append((attr_name, inds, t))
    if param in attr_inds:
        attr_inds[param].extend(param_attr_inds)
    else:
        attr_inds[param] = param_attr_inds

def set_robot_body_to_pred_values(pred, t):
    robot_body = pred._param_to_body[pred.robot]
    robot_body.set_pose(pred.robot.pose[:, t])
    robot_body.set_dof(pred.robot.backHeight[:, t], pred.robot.lArmPose[:, t], pred.robot.lGripper[:, t], pred.robot.rArmPose[:, t], pred.robot.rGripper[:, t])

def plot_transform(env, T, s=0.1):
    """
    Plots transform T in openrave environment.
    S is the length of the axis markers.
    """
    h = []
    x = T[0:3,0]
    y = T[0:3,1]
    z = T[0:3,2]
    o = T[0:3,3]
    h.append(env.drawlinestrip(points=np.array([o, o+s*x]), linewidth=3.0, colors=np.array([(1,0,0),(1,0,0)])))
    h.append(env.drawlinestrip(points=np.array([o, o+s*y]), linewidth=3.0, colors=np.array(((0,1,0),(0,1,0)))))
    h.append(env.drawlinestrip(points=np.array([o, o+s*z]), linewidth=3.0, colors=np.array(((0,0,1),(0,0,1)))))
    return h


def lin_interp_traj(start, end, time_steps):
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps+1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps+1)
        traj[i, :] = traj_row
    return traj

def ee_reachable_resample(pred, negated, t, plan):
    assert not negated
    handles = []
    v = OpenRAVEViewer.create_viewer()

    def target_trans_callback(target_trans):
        handles.append(plot_transform(v.env, target_trans))
        v.draw_plan_ts(plan, t)

    def plot_time_step_callback():
        v.draw_plan_ts(plan, t)
    plot_time_step_callback()

    targets = plan.get_param('GraspValid', 1, {0: pred.ee_pose})
    assert len(targets) == 1
    # confirm target is correct
    target_pose = targets[0].value[:, 0]
    set_robot_body_to_pred_values(pred, t)

    theta = 0
    robot = pred.robot
    robot_body = pred._param_to_body[robot]
    for _ in range(NUM_EEREACHABLE_RESAMPLE_ATTEMPTS):
        # generate collision free base pose
        base_pose = get_col_free_base_pose_around_target(t, plan, target_pose, robot, save=True,
                                                  dist=OBJ_RING_SAMPLING_RADIUS,
                                                  callback=plot_time_step_callback)
        if base_pose is None:
            print "we should always be able to sample a collision-free base pose"
            st()
        # generate collision free arm pose
        target_rot = np.array([get_random_theta(), 0, 0])

        torso_pose, arm_pose = get_col_free_torso_arm_pose(t, target_pose, target_rot,
                                                           robot, robot_body, save=True,
                                                           arm_pose_seed=None,
                                                           callback=target_trans_callback)
        st()
        if torso_pose is None:
            print "we should be able to find an IK"
            continue

        # generate approach IK
        ee_trans = OpenRAVEBody.transform_from_obj_pose(target_pose, target_rot)
        rel_pt = pred.get_rel_pt(-pred._steps)
        target_pose_approach = np.dot(ee_trans, np.r_[rel_pt, 1])[:3]

        torso_pose_approach, arm_pose_approach = get_col_free_torso_arm_pose(
                                                    t, target_pose_approach, target_rot,
                                                    robot, robot_body, save=True,
                                                    arm_pose_seed=arm_pose,
                                                    callback=target_trans_callback)
        st()
        if torso_pose_approach is None:
            continue

        # generate retreat IK
        ee_trans = OpenRAVEBody.transform_from_obj_pose(target_pose, target_rot)
        rel_pt = pred.get_rel_pt(pred._steps)
        target_pose_retreat = np.dot(ee_trans, np.r_[rel_pt, 1])[:3]

        torso_pose_retreat, arm_pose_retreat = get_col_free_torso_arm_pose(
                                                    t, target_pose_retreat, target_rot,
                                                    robot, robot_body, save=True,
                                                    arm_pose_seed=arm_pose,
                                                    callback=target_trans_callback)
        st()
        if torso_pose_retreat is not None:
            break
    else:
        print "we should always be able to sample a collision-free base and arm pose"
        st()

    attr_inds = OrderedDict()
    res = []
    arm_approach_traj = lin_interp_traj(arm_pose_approach, arm_pose, pred._steps)
    torso_approach_traj = lin_interp_traj(torso_pose_approach, torso_pose, pred._steps)
    base_approach_traj = lin_interp_traj(base_pose, base_pose, pred._steps)

    arm_retreat_traj = lin_interp_traj(arm_pose, arm_pose_retreat, pred._steps)
    torso_retreat_traj = lin_interp_traj(torso_pose, torso_pose_retreat, pred._steps)
    base_retreat_traj = lin_interp_traj(base_pose, base_pose, pred._steps)

    arm_traj = np.hstack((arm_approach_traj, arm_retreat_traj[:, 1:]))
    torso_traj = np.hstack((torso_approach_traj, torso_retreat_traj[:, 1:]))
    base_traj = np.hstack((base_approach_traj, base_retreat_traj[:, 1:]))

    # add attributes for approach and retreat
    for ind in range(2*pred._steps+1):
        robot_attr_name_val_tuples = [('rArmPose', arm_traj[:, ind]),
                                      ('backHeight', torso_traj[:, ind]),
                                      ('pose', base_traj[:, ind])]
        add_to_attr_inds_and_res(t+ind-pred._steps, attr_inds, res, pred.robot, robot_attr_name_val_tuples)
    st()

    ee_pose_attr_name_val_tuples = [('value', target_pose),
                                    ('rotation', target_rot)]
    add_to_attr_inds_and_res(t, attr_inds, res, pred.ee_pose, ee_pose_attr_name_val_tuples)
    # v.draw_plan_ts(plan, t)
    v.animate_range(plan, (t-pred._steps, t+pred._steps))
    # check that indexes are correct
    import ipdb; ipdb.set_trace()

    return np.array(res), attr_inds
