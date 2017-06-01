from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2_sampling import get_expr_mult
import core.util_classes.common_constants as const
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from errors_exceptions import PredicateException
from collections import OrderedDict
import numpy as np
import ctrajoptpy
import time

class CollisionPredicate(ExprPredicate):

    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = const.DIST_SAFE, debug = False, ind0=0, ind1=1, tol=const.COLLISION_TOL):
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

        # Set pose of each rave body
        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        obj = self.params[self.ind1]
        obj_body = self._param_to_body[obj]
        can_pos, can_rot = x[-6:-3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)

        # Make sure two body is in the same environment
        assert robot_body.env_body.GetEnv() == obj_body.env_body.GetEnv()
        self.set_active_dof_inds(robot_body, reset=False)
        # Setup collision checkers
        self._cc.SetContactDistance(const.MAX_CONTACT_DISTANCE)
        collisions = self._cc.BodyVsBody(robot_body.env_body, obj_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, obj_body, collisions)
        # set active dof value back to its original state (For successive function call)
        self.set_active_dof_inds(robot_body, reset=True)
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
        # self._cc.SetContactDistance(const.MAX_CONTACT_DISTANCE)
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

        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        can_pos, can_rot = x[-12:-9], x[-9:-6]
        held_pose, held_rot = x[-6:-3], x[-3:]

        obj = self.params[self.ind1]
        obj_body = self._param_to_body[obj]
        obj_body.set_pose(can_pos, can_rot)
        self.set_active_dof_inds(robot_body, reset=False)
        self._cc.SetContactDistance(const.MAX_CONTACT_DISTANCE)
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
        col_jac2 = np.c_[np.zeros((1, self.attr_dim)), col_jac2]
        # Stack these val and jac, and return
        val = np.vstack((col_val1, col_val2))
        jac = np.vstack((col_jac1, col_jac2))
        self.set_active_dof_inds(robot_body, reset=True)
        self._cache[flattened] = (val.copy(), jac.copy())
        return val, jac

    def _calc_grad_and_val(self, robot_body, obj_body, collisions):
        """
            This function is helper function of robot_obj_collision(self, x)
            It calculates collision distance and gradient between each robot's link and object

            robot_body: OpenRAVEBody containing body information of pr2 robot
            obj_body: OpenRAVEBody containing body information of object
            collisions: list of collision objects returned by collision checker
            Note: Needs to provide attr_dim indicating robot pose's total attribute dim
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
            grad = np.zeros((1, self.attr_dim+6))
            grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)
            # robot_grad = np.dot(sign * normal, robot_jac).reshape((1,20))
            col_vec = -sign*normal
            # Calculate object pose jacobian
            # obj_jac = np.array([-sign*normal])
            grad[:, self.attr_dim:self.attr_dim+3] = np.array([-sign*normal])
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, self.attr_dim+3:self.attr_dim+6] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            # links.append((robot_link_ind, self.dsafe - distance, robot_grad, robot.GetLink(linkRobot)))
            links.append((robot_link_ind, self.dsafe - distance, grad, robot.GetLink(linkRobot)))

            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        # arrange gradients in proper link order
        max_dist = self.dsafe - const.MAX_CONTACT_DISTANCE
        vals, robot_grads = max_dist*np.ones((num_links,1)), np.zeros((num_links, self.attr_dim+6))
        links = sorted(links, key = lambda x: x[0])
        vals[:len(links),0] = np.array([link[1] for link in links])
        robot_grads[:len(links), range(self.attr_dim+6)] = np.array([link[2] for link in links]).reshape((len(links), self.attr_dim+6))
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

class PosePredicate(ExprPredicate):

    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = const.DIST_SAFE, debug = False, ind0=0, ind1=1, tol=const.POSE_TOL, active_range=(0,0)):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self.handle = []
        super(PosePredicate, self).__init__(name, e, attr_inds, params, expected_param_types, tol=tol, active_range=active_range)

    def pos_check(self, x):
        """
            This function is used to check whether:
                object is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body)
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        # Set Can Pose
        can_pos, can_rot = x[-6: -3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)
        obj_trans = obj_body.env_body.GetTransform()
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

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Parse the pose values
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body)
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        # Set Can Pose
        can_pos, can_rot = x[-6: -3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)
        obj_trans = obj_body.env_body.GetTransform()
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pos, can_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]# axises = [axis_z, axis_y, axis_x]
        rot_val, rot_jac = self.rot_error(obj_trans, robot_trans, axises, arm_joints)
        return rot_val, rot_jac

    def ee_pose_check_rel_obj(self, x, rel_pt):
        """
            This function is used to check whether:
                End effective pose's position is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eRot

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body

        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body)
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]

        ee_pos, ee_rot = x[-6:-3], x[-3:]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        pos_val, pos_jac = self.pos_error_rel_to_obj(obj_trans, robot_trans, axises, arm_joints, rel_pt)

        return pos_val, pos_jac

    def ee_rot_check(self, x):
        """
            This function is used to check whether:
                End effective pose's rotational axis is parallel to that of robot gripper

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->eePose->eRot

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """

        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body)
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]

        ee_pos, ee_rot = x[-6:-3], x[-3:]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        rot_val, rot_jac = self.rot_lock(obj_trans, robot_trans, axises, arm_joints)

        return rot_val, rot_jac

    def both_arm_pos_check(self, x):
        """
            This function is used to check whether:
                basket is at both robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")

        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]
        # Set Can Pose
        can_pos, can_rot = x[-6: -3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)
        obj_trans = obj_body.env_body.GetTransform()
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pos, can_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]# axises = [axis_z, axis_y, axis_x]

        l_pos_val, l_pos_jac = self.pos_error(obj_trans, robot_left_trans, axises, left_arm_joints, [0.317,0,0], "left")
        r_pos_val, r_pos_jac = self.pos_error(obj_trans, robot_right_trans, axises, right_arm_joints, [-0.317,0,0], "right")

        pos_val = np.vstack([l_pos_val, r_pos_val])
        pos_jac = np.vstack([l_pos_jac, r_pos_jac])
        return pos_val, pos_jac

    def both_arm_rot_check(self, x):
        """
            This function is used to check whether:
                object is at robot gripper's center

            x: 26 dimensional list aligned in following order,
            BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        obj_body = self._param_to_body[self.params[self.ind1]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")

        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]
        # Set Can Pose
        can_pos, can_rot = x[-6: -3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)
        obj_trans = obj_body.env_body.GetTransform()
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(can_pos, can_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]# axises = [axis_z, axis_y, axis_x]

        l_rot_val, l_rot_jac = self.rot_error(obj_trans, robot_left_trans, axises, left_arm_joints, "left")
        r_rot_val, r_rot_jac = self.rot_error(obj_trans, robot_right_trans, axises, right_arm_joints, "right")

        rot_val = np.vstack([l_rot_val, r_rot_val])
        rot_jac = np.vstack([l_rot_jac, r_rot_jac])
        return rot_val, rot_jac

    def vel_check(self, x):
        """
        Check whether end effector are within range
        """
        jac = np.zeros((12, 40))
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        # Set poses and Get transforms

        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]

        left_pose_rot =robot_left_trans[:3,3]
        left_arm_jac = np.array([np.cross(joint.GetAxis(), left_pose_rot[:3] - joint.GetAnchor()) for joint in left_arm_joints]).T.copy()
        left_base_jac = np.cross(np.array([0, 0, 1]), left_pose_rot[:3] - np.zeros((3,))).reshape((3,))

        jac[0:3, 0:7] = -left_arm_jac
        jac[0:3, 16] = -left_base_jac
        jac[0:3, 17:20] = -np.eye(3)

        jac[3:6, 0:7] = left_arm_jac
        jac[3:6, 16] = left_base_jac
        jac[3:6, 17:20] = -np.eye(3)

        right_pose_rot =robot_right_trans[:3,3]
        right_arm_jac = np.array([np.cross(joint.GetAxis(), right_pose_rot[:3] - joint.GetAnchor()) for joint in right_arm_joints]).T.copy()
        right_base_jac = np.cross(np.array([0, 0, 1]), right_pose_rot[:3] - np.zeros((3,))).reshape((3,))

        jac[6:9, 8:15] = -right_arm_jac
        jac[6:9, 16] = -right_base_jac
        jac[9:12, 8:15] = right_arm_jac
        jac[9:12, 16] = right_base_jac

        self.set_robot_poses(x[20:37], robot_body)
        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
        # Added here just in case
        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]

        left_new_pose_rot =robot_left_trans[:3,3]
        left_new_arm_jac = np.array([np.cross(joint.GetAxis(), left_new_pose_rot[:3] - joint.GetAnchor()) for joint in left_arm_joints]).T.copy()
        left_new_base_jac = np.cross(np.array([0, 0, 1]), left_new_pose_rot[:3] - np.zeros((3,))).reshape((3,))
        jac[0:3, 20:27] = left_new_arm_jac
        jac[0:3, 36] = left_new_base_jac
        jac[3:6, 20:27] = -left_new_arm_jac
        jac[3:6, 36] = -left_new_base_jac

        right_new_pose_rot = robot_right_trans[:3,3]
        right_new_arm_jac = np.array([np.cross(joint.GetAxis(), right_new_pose_rot[:3] - joint.GetAnchor()) for joint in right_arm_joints]).T.copy()
        right_new_base_jac = np.cross(np.array([0, 0, 1]), right_new_pose_rot[:3] - np.zeros((3,))).reshape((3,))
        jac[6:9, 28:35] = right_new_arm_jac
        jac[6:9, 36] = right_new_base_jac
        jac[6:9, 37:40] = -np.eye(3)
        jac[9:12, 28:35] = -right_new_arm_jac
        jac[9:12, 36] = -right_new_base_jac
        jac[9:12, 37:40] = -np.eye(3)

        dist_left = (left_new_pose_rot - left_pose_rot - x[17:20].flatten()).reshape((3,1))
        dist_left_rev = (left_pose_rot - left_new_pose_rot - x[17:20].flatten()).reshape((3,1))
        dist_right = (right_new_pose_rot - right_pose_rot - x[37:40].flatten()).reshape((3,1))
        dist_right_rev = (right_pose_rot - right_new_pose_rot - x[37:40].flatten()).reshape((3,1))

        val = np.vstack([dist_left, dist_left_rev, dist_right, dist_right_rev])
        return val, jac

class At(ExprPredicate):
    """
        Format: # At, Can, Target

        Non-robot related
    """
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
        self.spacial_anchor = True

class RobotAt(ExprPredicate):
    """
        Format: RobotAt, Robot, RobotPose

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.robot, self.robot_pose = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b ,val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(RobotAt, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class IsMP(ExprPredicate):
    """
        Format: IsMP Robot (Just the Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            setup_mov_limit_check[Function]: function that sets constraint matrix
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = self.attr_inds
        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
        self.spacial_anchor = False

class WithinJointLimit(ExprPredicate):
    """
        Format: WithinJointLimit Robot

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            setup_mov_limit_check[Function]: function that sets constraint matrix
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, = params
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(WithinJointLimit, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = False

class Stationary(ExprPredicate):
    """
        Format: Stationary, Can

        Non-robot related
    """
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.can,  = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1,2], dtype=np.int)),
                                             ("rotation", np.array([0,1,2], dtype=np.int))])])

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
        self.spacial_anchor = False

class StationaryBase(ExprPredicate):
    """
        Format: StationaryBase, Robot (Only Robot Base)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
        self.spacial_anchor = False

class StationaryArms(ExprPredicate):
    """
        Format: StationaryArms, Robot (Only Robot Arms)

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.robot,  = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArms, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
        self.spacial_anchor = False

class StationaryW(ExprPredicate):
    """
        Format: StationaryW, Obstacle

        Non-robot related
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.w, = params
        attr_inds = OrderedDict([(self.w, [("pose", np.array([0, 1, 2], dtype=np.int)),
                                           ("rotation", np.array([0, 1, 2], dtype=np.int))])])
        A = np.c_[np.eye(6), -np.eye(6)]
        b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1))
        self.spacial_anchor = False

class StationaryNEq(ExprPredicate):
    """
        Format: StationaryNEq, Can, Can(Hold)

        Non-robot related
    """
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
        self.spacial_anchor = False

class GraspValid(ExprPredicate):
    """
        Format: GraspValid EEPose Target

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class InContact(ExprPredicate):
    """
        Format: InContact robot EEPose(Right Arm) target

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
            GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
            GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.eye(1).reshape((1,1))
        b = np.zeros(1).reshape((1,1))

        val = np.array([[self.GRIPPER_CLOSE]])
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = np.array([[self.GRIPPER_OPEN]])
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContact, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class InContacts(ExprPredicate):
    """
        Format: InContact Robot EEPose(Left Arm) EEPose(Right Arm) Target

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]: dimension of robot attribute
            GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
            GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.left_ee, self.right_ee, self.target = params
        attr_inds = self.attr_inds

        A = np.eye(2).reshape((2,2))
        b = np.zeros((2,1))

        val = np.array([[self.GRIPPER_CLOSE, self.GRIPPER_CLOSE]]).T
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = np.array([[self.GRIPPER_OPEN, self.GRIPPER_OPEN]]).T
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContacts, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class InGripper(PosePredicate):
    """
        Format: InGripper, Robot, Can

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate value
            eval_grad[Function]:Function returns predicate gradient
            coeff[Float]:In Gripper coeffitions, used during optimazation
            opt_coeff[Float]:In Gripper coeffitions, used during optimazation
    """
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.can = params
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.coeff*self.eval_f(x)
        grad = lambda x: self.coeff*self.eval_grad(x)

        self.opt_expr = EqExpr(Expr(lambda x: self.opt_coeff * f(x),
                                    lambda x: self.opt_coeff*grad(x)),
                                np.zeros((1,1)))

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim,1))
        e = EqExpr(pos_expr, val)
        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1)
        self.spacial_anchor = True

    def get_expr(self, negated):
        if negated:
            return None
        return self.opt_expr

class EEReachable(PosePredicate):
    """
        Format: EEUnreachable Robot, StartPose, EEPose

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            get_robot_info[Function]:Function that returns robot's transformations and arm indices
            eval_f[Function]:Function returns predicate value
            eval_grad[Function]:Function returns predicate gradient
            coeff[Float]:EEReachable coeffitions, used during optimazation
            opt_coeff[Float]:EEReachable coeffitions, used during optimazation
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        assert len(params) == 3
        self._env = env
        self.robot, self.start_pose, self.ee_pose = params
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        self._steps = steps
        f = lambda x: self.coeff*self.eval_f(x)
        grad = lambda x: self.coeff*self.eval_grad(x)

        pos_expr = Expr(f, grad)
        e = EqExpr(pos_expr, np.zeros((3*(2*self._steps+1),1)))
        self.opt_expr = EqExpr(Expr(lambda x: self.opt_coeff*f(x),
                                    lambda x: self.opt_coeff*grad(x)),
                                np.zeros((1,1)))
        super(EEReachable, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(-self._steps, self._steps))
        self.spacial_anchor = True

    def get_expr(self, negated):
        if negated:
            return None
        else:
            return self.opt_expr

class Obstructs(CollisionPredicate):
    """
        Format: Obstructs, Robot, RobotPose, RobotPose, Can

        Robot related

        Requires:
            attr_inds[OrderedDict]: robot attribute indices
            attr_dim[Int]:number of attribute in robot's full pose
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            coeff[Float]:EEReachable coeffitions, used during optimazation
            neg_coeff[Float]:EEReachable coeffitions, used during optimazation
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.COLLISION_TOL):
        assert len(params) == 4
        self._env = env
        self.robot, self.startp, self.endp, self.can = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.coeff*self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff*self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff*self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff*self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)
        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Obstructs, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=3, debug=debug, tol=tol)
        self.priority = 2
        self.spacial_anchor = False

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class ObstructsHolding(CollisionPredicate):
    """
        Format: ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

        Robot related

        Requires:
            attr_dim[Int]:number of attribute in robot's full pose
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            OBSTRUCTS_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimazation problem
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 5
        self._env = env
        self.robot, self.startp, self.endp, self.obstruct, self.held = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstruct: self.lazy_spawn_or_body(self.obstruct, self.obstruct.name, self.obstruct.geom),
                               self.held: self.lazy_spawn_or_body(self.held, self.held.name, self.held.geom)}

        links = len(self.robot.geom.col_links)
        if self.held.name == self.obstruct.name:
            col_fn, offset = self.robot_obj_collision, const.DIST_SAFE - const.COLLISION_TOL
            val = np.zeros((links,1))
        else:
            col_fn, offset = self.robot_obj_held_collision, 0
            val = np.zeros((links+1,1))

        f = lambda x: self.coeff * (col_fn(x)[0] - offset)
        grad = lambda x: self.coeff * col_fn(x)[1]
        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * (col_fn(x)[0] - offset)
        grad_neg = lambda x: self.neg_coeff * col_fn(x)[1]

        col_expr, col_expr_neg = Expr(f, grad), Expr(f_neg, grad_neg)
        e, self.neg_expr = LEqExpr(col_expr, val), LEqExpr(col_expr_neg, val)
        self.neg_expr_opt = LEqExpr(get_expr_mult(self.OBSTRUCTS_OPT_COEFF, col_expr_neg), val)
        super(ObstructsHolding, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=3, debug = debug)
        self.spacial_anchor = False
        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr_opt
        else:
            return None

class Collides(CollisionPredicate):
    """
        Format: Collides Item Item

        Non-robot related
    """
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
        self.spacial_anchor = False

    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None

class RCollides(CollisionPredicate):
    """
        Format: RCollides Robot Obstacle

        Robot related

        Requires:
            attr_dim[Int]:number of attribute in robot's full pose
            attr_inds[OrderedDict]: robot attribute indices
            set_robot_poses[Function]:Function that sets robot's poses
            set_active_dof_inds[Function]:Function that sets robot's active dof indices
            RCOLLIDES_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimazation problem
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.obstacle = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom),
                               self.obstacle: self.lazy_spawn_or_body(self.obstacle, self.obstacle.name, self.obstacle.geom)}

        f = lambda x: self.coeff * self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff * self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[1]

        f_neg_opt = lambda x: self.opt_coeff*self.robot_obj_collision(x)[0]
        grad_neg_opt = lambda x: self.opt_coeff*self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)
        val = np.zeros((links,1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        col_expr_neg_opt = Expr(f_neg_opt, grad_neg_opt)
        self.neg_expr = LEqExpr(col_expr_neg, val)
        self.neg_expr_opt = LEqExpr(col_expr_neg_opt, val)

        super(RCollides, self).__init__(name, e, attr_inds, params,
                                        expected_param_types, ind0=0, ind1=1)
        self.spacial_anchor = False
        self.priority = 2

    def get_expr(self, negated):
        if negated:
            return self.neg_expr_opt
        else:
            return None

class BasketLevel(ExprPredicate):
    '''
    Format: BasketLevel Basket
    '''
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.array([[np.pi/2], [0], [np.pi/2]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BasketLevel, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = False

class ObjectWithinRotLimit(ExprPredicate):
    '''
    Format: ObjectWithinRotLimit Object
    '''
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[np.eye(self.attr_dim)]
        b, val = np.array([[np.pi], [np.pi], [np.pi]]), np.array([[2*np.pi], [2*np.pi], [2*np.pi]])
        pos_expr = AffExpr(A, b)
        e = LEqExpr(pos_expr, val)
        super(ObjectWithinRotLimit, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = False

class EERetiming(PosePredicate):
    """
        Format: EERetiming Robot EEVel

        Robot related
        Requires:
        self.attr_inds
        self.coeff
        self.eval_f
        self.eval_grad
        self.eval_dim
    """
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 2
        self._env = env
        self.robot, self.ee_vel = params
        attr_inds = self.attr_inds

        self._param_to_body = {self.robot: self.lazy_spawn_or_body(self.robot, self.robot.name, self.robot.geom)}

        f = lambda x: self.coeff*self.eval_f(x)
        grad = lambda x: self.coeff*self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim,1))
        e = EqExpr(pos_expr, val)
        super(EERetiming, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1, active_range = (0,1))
        self.spacial_anchor = False
