from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2_sampling import get_expr_mult
from core.util_classes.param_setup import ParamSetup
import core.util_classes.common_constants as const
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from errors_exceptions import PredicateException
from collections import OrderedDict
import numpy as np

import pybullet as p

import itertools
import sys
import traceback
import time


class CollisionPredicate(ExprPredicate):

    # @profile
    def __init__(
        self,
        name,
        e,
        attr_inds,
        params,
        expected_param_types,
        dsafe=const.DIST_SAFE,
        debug=False,
        ind0=0,
        ind1=1,
        tol=const.COLLISION_TOL,
        priority=0,
    ):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self._plot_handles = []
        # self._cache = {}
        super(CollisionPredicate, self).__init__(
            name, e, attr_inds, params, expected_param_types, tol=tol, priority=priority
        )

    # @profile
    def robot_self_collision(self, x):
        """
        This function is used to calculae collisiosn between Robot and Can
        This function calculates the collision distance gradient associated to it
        x: 26 dimensional list of values aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot
        """
        # Parse the pose value
        self._plot_handles = []
        flattened = tuple(x.flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        # Set pose of each rave body
        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        self.set_active_dof_inds(robot_body, reset=False)
        # Setup collision checkers
        collisions = p.getClosestPoints(
            robot_body.body_id, robot_body.body_id, const.MAX_CONTACT_DISTANCE
        )

        # Calculate value and jacobian
        col_val, col_jac = self._calc_self_grad_and_val(robot_body, collisions)
        # set active dof value back to its original state (For successive function call)
        self.set_active_dof_inds(robot_body, reset=True)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        # import ipdb; ipdb.set_trace()
        return col_val, col_jac

    # @profile
    def robot_obj_collision(self, x):
        """
        This function is used to calculae collisiosn between Robot and Can
        This function calculates the collision distance gradient associated to it
        x: 26 dimensional list of values aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot
        """
        # Parse the pose value
        self._plot_handles = []
        flattened = tuple(x.flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        # Set pose of each rave body
        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        obj = self.params[self.ind1]
        obj_body = self._param_to_body[obj]
        can_pos, can_rot = x[-6:-3], x[-3:]
        obj_body.set_pose(can_pos, can_rot)

        # Make sure two body is in the same environment
        collisions = p.getClosestPoints(
            robot_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE
        )
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, obj_body, collisions)
        # set active dof value back to its original state (For successive function call)
        self.set_active_dof_inds(robot_body, reset=True)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        # import ipdb; ipdb.set_trace()
        return col_val, col_jac

    # @profile
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
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

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
        collisions = p.getClosestPoints(
            can_body.body_id, obstr_body.body_id, const.MAX_CONTACT_DISTANCE
        )
        # Calculate value and jacobian
        col_val, col_jac = self._calc_obj_grad_and_val(can_body, obstr_body, collisions)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        return col_val, col_jac

    # @profile
    def robot_obj_held_collision(self, x):
        """
        Similar to robot_obj_collision in CollisionPredicate; however, this function take into account of object holding

        x: 26 dimensional list of values aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose->CanRot->HeldPose->HeldRot
        """
        self._plot_handles = []
        flattened = tuple(x.round(5).flatten())
        # cache prevents plotting
        # if flattened in self._cache and not self._debug:
        #     return self._cache[flattened]

        robot = self.params[self.ind0]
        robot_body = self._param_to_body[robot]
        self.set_robot_poses(x, robot_body)

        can_pos, can_rot = x[-12:-9], x[-9:-6]
        held_pose, held_rot = x[-6:-3], x[-3:]

        obj = self.params[self.ind1]
        obj_body = self._param_to_body[obj]
        obj_body.set_pose(can_pos, can_rot)
        collisions1 = p.getClosestPoints(
            robot_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE
        )

        col_val1, col_jac1 = self._calc_grad_and_val(robot_body, obj_body, collisions1)

        col_jac1 = np.c_[col_jac1, np.zeros((len(self.col_link_pairs), 6))]
        # find collision between object and object held
        held_body = self._param_to_body[self.obj]
        held_body.set_pose(held_pose, held_rot)
        collisions2 = p.getClosestPoints(
            held_body.body_id, obj_body.body_id, const.MAX_CONTACT_DISTANCE
        )

        col_val2, col_jac2 = self._calc_obj_held_grad_and_val(
            robot_body, held_body, obj_body, collisions2
        )

        # Stack these val and jac, and return
        val = np.vstack((col_val1, col_val2))
        jac = np.vstack((col_jac1, col_jac2))
        self.set_active_dof_inds(robot_body, reset=True)
        # self._cache[flattened] = (val.copy(), jac.copy())
        return val, jac

    # def _old_calc_grad_and_val(self, robot_body, obj_body, collisions):
    #     """
    #         This function is helper function of robot_obj_collision(self, x)
    #         It calculates collision distance and gradient between each robot's link and object

    #         robot_body: OpenRAVEBody containing body information of pr2 robot
    #         obj_body: OpenRAVEBody containing body information of object
    #         collisions: list of collision objects returned by collision checker
    #         Note: Needs to provide attr_dim indicating robot pose's total attribute dim
    #     """
    #     # Initialization
    #     links = []
    #     robot = self.params[self.ind0]
    #     col_links = robot.geom.col_links
    #     num_links = len(col_links)
    #     obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
    #     Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
    #     rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
    #     for c in collisions:
    #         # Identify the collision points
    #         linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
    #         linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
    #         linkRobot, linkObj = None, None
    #         sign = 0
    #         if linkAParent == robot_body.name and linkBParent == obj_body.name:
    #             ptRobot, ptObj = c.GetPtA(), c.GetPtB()
    #             linkRobot, linkObj = linkA, linkB
    #             sign = -1
    #         elif linkBParent == robot_body.name and linkAParent == obj_body.name:
    #             ptRobot, ptObj = c.GetPtB(), c.GetPtA()
    #             linkRobot, linkObj = linkB, linkA
    #             sign = 1
    #         else:
    #             continue
    #         if linkRobot not in col_links:
    #             continue
    #         # Obtain distance between two collision points, and their normal collision vector
    #         distance = c.GetDistance()
    #         normal = c.GetNormal()
    #         # Calculate robot jacobian
    #         robot = robot_body.env_body
    #         robot_link_ind = robot.GetLink(linkRobot).GetIndex()
    #         robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptObj)
    #         grad = np.zeros((1, self.attr_dim+6))
    #         grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)
    #         # robot_grad = np.dot(sign * normal, robot_jac).reshape((1,20))
    #         col_vec = -sign*normal
    #         # Calculate object pose jacobian
    #         # obj_jac = np.array([-sign*normal])
    #         grad[:, self.attr_dim:self.attr_dim+3] = np.array([-sign*normal])
    #         torque = ptObj - obj_pos[:3]
    #         # Calculate object rotation jacobian
    #         rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
    #         # obj_jac = np.c_[obj_jac, rot_vec]
    #         grad[:, self.attr_dim+3:self.attr_dim+6] = rot_vec
    #         # Constructing gradient matrix
    #         # robot_grad = np.c_[robot_grad, obj_jac]
    #         # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
    #         # links.append((robot_link_ind, self.dsafe - distance, robot_grad, robot.GetLink(linkRobot)))
    #         links.append((robot_link_ind, self.dsafe - distance, grad, robot.GetLink(linkRobot)))

    #         if self._debug:
    #             self.plot_collision(ptRobot, ptObj, distance)

    #     # arrange gradients in proper link order
    #     max_dist = self.dsafe - const.MAX_CONTACT_DISTANCE
    #     vals, robot_grads = max_dist*np.ones((num_links,1)), np.zeros((num_links, self.attr_dim+6))
    #     links = sorted(links, key = lambda x: x[0])
    #     vals[:len(links),0] = np.array([link[1] for link in links])
    #     robot_grads[:len(links), range(self.attr_dim+6)] = np.array([link[2] for link in links]).reshape((len(links), self.attr_dim+6))
    #     # TODO: remove line below which was added for debugging purposes
    #     self.links = links
    #     return vals, robot_grads

    # @profile
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
        obj = self.params[self.ind1]
        col_links = robot.geom.col_links
        obj_links = obj.geom.col_links
        obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
        rot_axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]
        link_pair_to_col = {}
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 0
            if linkAParent == robot_body.body_id and linkBParent == obj_body.body_id:
                ptRobot, ptObj = c.positionOnA, c.positionOnB
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == robot_body.body_id and linkAParent == obj_body.body_id:
                ptRobot, ptObj = c.positionOnB, c.positionOnA
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            if linkRobot not in col_links or linkObj not in obj_links:
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.contactDistance
            normal = c.contactNormalOnB  # Pointing towards A
            n_jnts = p.getNumJoints(robot_body.body_id)
            jnts = p.getJointStates(list(range(n_jnts)))[0]
            robot_jac, robot_ang_jac = p.calculateJacobian(
                robot_body.body_id,
                linkRobot,
                ptRobot,
                objPositions=jnts,
                objVelocities=np.zeros(n_jnts),
                objAccelerations=np.zeros(n_jnts),
            )

            grad = np.zeros((1, self.attr_dim + 6))
            grad[:, : self.attr_dim] = np.dot(sign * normal, robot_jac)
            col_vec = -sign * normal
            # Calculate object pose jacobian
            grad[:, self.attr_dim : self.attr_dim + 3] = col_vec
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]]
            )
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, self.attr_dim + 3 : self.attr_dim + 6] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot, linkObj)] = [
                self.dsafe - distance,
                grad,
                None,
                None,
            ]
            # import ipdb; ipdb.set_trace()
            # if self._debug:
            #     self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, self.attr_dim + 6)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape(
            (len(greds), self.attr_dim + 6)
        )

    # @profile
    def _calc_self_grad_and_val(self, robot_body, collisions):
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
        link_pair_to_col = {}
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 0
            if linkAParent == robot_body.body_id and linkBParent == obj_body.body_id:
                ptRobot1, ptRobot2 = c.positionOnA, c.positionOnB
                linkRobot1, linkRobot2 = linkA, linkB
                sign = -1
            else:
                continue

            if linkRobot1 not in col_links or linkRobot2 not in col_links:
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.contactDistance
            normal = c.contactNormalOnB  # Pointing towards A
            n_jnts = p.getNumJoints(robot_body.body_id)
            jnts = p.getJointStates(list(range(n_jnts)))[0]
            robot_jac, robot_ang_jac = p.calculateJacobian(
                robot_body.body_id,
                linkRobot1,
                ptRobot1,
                objPositions=jnts,
                objVelocities=np.zeros(n_jnts),
                objAccelerations=np.zeros(n_jnts),
            )

            grad = np.zeros((1, self.attr_dim))
            grad[:, : self.attr_dim] = np.dot(sign * normal, robot_jac)

            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot1, linkRobot2)] = [
                self.dsafe - distance,
                grad,
                None,
                None,
            ]
            # import ipdb; ipdb.set_trace()
            # if self._debug:
            #     self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, self.attr_dim)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape(
            (len(greds), self.attr_dim)
        )

    # @profile
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

        held_links = self.obj.geom.col_links
        obs_links = self.obstacle.geom.col_links

        link_pair_to_col = {}
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 0
            if linkAParent == obj_body.body_id and linkBParent == obstr_body.body_id:
                ptObj, ptObstr = c.positionOnA, c.positionOnB
                linkObj, linkObstr = linkA, linkB
                sign = -1
            elif linkBParent == obj_body.body_id and linkAParent == obstr_body.body_id:
                ptObj, ptObstr = c.positionOnB, c.positionOnA
                linkObj, linkObstr = linkB, linkA
                sign = 1
            else:
                continue

            if linkObj not in held_links or linkObstr not in obs_links:
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.contactDistance
            normal = c.contactNormalOnB  # Pointing towards A

            col_vec = -sign * normal
            # Calculate object pose jacobian
            obj_jac = np.array([normal])
            obj_pos = OpenRAVEBody.obj_pose_from_transform(
                obj_body.env_body.GetTransform()
            )
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
            rot_axises = [
                [0, 0, 1],
                np.dot(Rz, [0, 1, 0]),
                np.dot(Rz, np.dot(Ry, [1, 0, 0])),
            ]
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]]
            )
            obj_jac = np.c_[obj_jac, -rot_vec]
            # Calculate obstruct pose jacobian
            obstr_jac = np.array([-normal])
            obstr_pos = OpenRAVEBody.obj_pose_from_transform(
                obstr_body.env_body.GetTransform()
            )
            torque = ptObstr - obstr_pos[:3]
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obstr_pos[:3], obstr_pos[3:])
            rot_axises = [
                [0, 0, 1],
                np.dot(Rz, [0, 1, 0]),
                np.dot(Rz, np.dot(Ry, [1, 0, 0])),
            ]
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]]
            )
            obstr_jac = np.c_[obstr_jac, rot_vec]
            # Constructing gradient matrix
            robot_grad = np.c_[obj_jac, obstr_jac]

            link_pair_to_col[(linkObj, linkObstr)] = [self.dsafe - distance, robot_grad]
            # if self._debug:
            #     self.plot_collision(ptObj, ptObstr, distance)

        vals, grads = [], []
        for robot_link, obj_link in self.obj_obj_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, 12)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            grads.append(col_infos[1])

        vals = np.vstack(vals)
        grads = np.vstack(grads)
        return vals, grads

    def _calc_obj_held_grad_and_val(self, robot_body, obj_body, obstr_body, collisions):
        """
        This function is helper function of robot_obj_collision(self, x) #Used in ObstructsHolding#
        It calculates collision distance and gradient between each robot's link and obstr,
        and between held object and obstr

        obj_body: OpenRAVEBody containing body information of object
        obstr_body: OpenRAVEBody containing body information of obstruction
        collisions: list of collision objects returned by collision checker
        """
        # Initialization
        robot_links = self.robot.geom.col_links
        held_links = self.obj.geom.col_links
        obs_links = self.obstacle.geom.col_links

        # TODO: Make a more generalized approach for any robot
        l_arm_joints = [31, 32, 33, 34, 35, 37, 38]
        r_arm_joints = [13, 14, 15, 16, 17, 19, 20]
        l_anchors = []
        r_anchors = []
        for jnt_id in l_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            l_anchors.append((parent_frame_pos + parent_pos, axis))
        for jnt_id in r_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            r_anchors.append((parent_frame_pos + parent_pos, axis))

        l_diff = np.linalg.norm(
            obj_body.env_body.GetTransform()[:3, 3] - l_ee_trans[:3, 3]
        )
        r_diff = np.linalg.norm(
            obj_body.env_body.GetTransform()[:3, 3] - r_ee_trans[:3, 3]
        )
        arm = "left"
        if r_diff < l_diff:
            arm = "right"

        link_pair_to_col = {}
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.linkIndexA, c.linkIndexB
            linkAParent, linkBParent = c.bodyUniqueIdA, c.bodyUniqueIdB
            sign = 1
            if linkAParent == robot_body.body_id and linkBParent == obj_body.body_id:
                ptRobot, ptObj = c.positionOnA, c.positionOnB
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == robot_body.body_id and linkAParent == obj_body.body_id:
                ptRobot, ptObj = c.positionOnB, c.positionOnA
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            if linkObj not in held_links or linkObstr not in obs_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            grad = np.zeros((1, self.attr_dim + 12))
            distance = c.contactDistance
            normal = c.contactNormalOnB

            # Calculate robot joint jacobian
            if arm == "left":
                l_arm_jac = np.array(
                    [np.cross(a[1], ptObj - a[0]) for a in l_anchors]
                ).T.copy()
                grad[:, :7] = np.dot(sign * normal, l_arm_jac)
            elif arm == "right":
                r_arm_jac = -np.array(
                    [np.cross(a[1], ptObj - a[0]) for joint in r_anchors]
                ).T.copy()
                grad[:, 8:15] = np.dot(sign * normal, r_arm_jac)

            # Calculate obstruct pose jacobian
            obstr_jac = -sign * normal
            obstr_pos = OpenRAVEBody.obj_pose_from_transform(
                obstr_body.env_body.GetTransform()
            )
            torque = ptObstr - obstr_pos[:3]
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obstr_pos[:3], obstr_pos[3:])
            rot_axises = [
                [0, 0, 1],
                np.dot(Rz, [0, 1, 0]),
                np.dot(Rz, np.dot(Ry, [1, 0, 0])),
            ]
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), obstr_jac) for axis in rot_axises]]
            )
            grad[:, self.attr_dim : self.attr_dim + 3] = obstr_jac
            grad[:, self.attr_dim + 3 : self.attr_dim + 6] = rot_vec

            # Calculate object_held pose jacobian
            obj_jac = sign * normal
            obj_pos = OpenRAVEBody.obj_pose_from_transform(
                obj_body.env_body.GetTransform()
            )
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
            rot_axises = [
                [0, 0, 1],
                np.dot(Rz, [0, 1, 0]),
                np.dot(Rz, np.dot(Ry, [1, 0, 0])),
            ]
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), obj_jac) for axis in rot_axises]]
            )
            grad[:, self.attr_dim + 6 : self.attr_dim + 9] = obj_jac
            grad[:, self.attr_dim + 9 : self.attr_dim + 12] = rot_vec

            link_pair_to_col[(linkObj, linkObstr)] = [self.dsafe - distance, grad]
            # if self._debug:
            #     self.plot_collision(ptObj, ptObstr, distance)

        vals, grads = [], []
        for robot_link, obj_link in self.obj_obj_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, 18)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            grads.append(col_infos[1])

        vals = np.vstack(vals)
        grads = np.vstack(grads)
        return vals, grads

    # @profile
    def test(self, time, negated=False, tol=None):
        if tol is None:
            tol = self.tol
        # This test is overwritten so that collisions can be calculated correctly
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'." % self)
        try:
            return self.neg_expr.eval(
                self.get_param_vector(time), tol=tol, negated=(not negated)
            )
        except IndexError as err:
            ## this happens with an invalid time
            traceback.print_exception(*sys.exc_info())
            raise PredicateException("Out of range time for predicate '%s'." % self)

    # @profile
    def plot_cols(self, env, t):
        _debug = self._debug
        self._env = env
        self._debug = True
        self.robot_obj_collision(self.get_param_vector(t))
        self._debug = _debug

    # @profile
    def plot_collision(self, ptA, ptB, distance):
        handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                handles.append(
                    self._env.drawarrow(
                        p1=ptA, p2=ptB, linewidth=0.001, color=(1, 0, 0)
                    )
                )
            else:
                handles.append(
                    self._env.drawarrow(
                        p1=ptA, p2=ptB, linewidth=0.001, color=(0, 0, 0)
                    )
                )
        self._plot_handles.extend(handles)


class PosePredicate(ExprPredicate):

    # @profile
    def __init__(
        self,
        name,
        e,
        attr_inds,
        params,
        expected_param_types,
        dsafe=const.DIST_SAFE,
        debug=False,
        ind0=0,
        ind1=1,
        tol=const.POSE_TOL,
        active_range=(0, 0),
        priority=0,
    ):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        self.handle = []
        super(PosePredicate, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            tol=tol,
            active_range=active_range,
            priority=priority,
        )

    # @profile
    def robot_obj_kinematics(self, x):
        """
        This function is used to check whether End Effective pose's position is at robot gripper's center

        Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        robot_body = self.robot.openrave_body
        # Setting the poses for forward kinematics to work
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = arm_inds

        ee_pos, ee_rot = x[-6:-3], x[-3:]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]  # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints

    # @profile
    def get_arm_jac(self, arm_jac, base_jac, obj_jac, arm):
        if not arm == "right" and not arm == "left":
            assert PredicateException("Invalid Arm Specified")

        dim = arm_jac.shape[0]
        if arm == "left":
            jacobian = np.hstack(
                (arm_jac, np.zeros((dim, 1)), np.zeros((dim, 8)), base_jac, obj_jac)
            )
        elif arm == "right":
            jacobian = np.hstack(
                (np.zeros((dim, 8)), arm_jac, np.zeros((dim, 1)), base_jac, obj_jac)
            )
        return jacobian

    # @profile
    def rel_ee_pos_check_f(self, x, rel_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rel_pos_error_f(obj_trans, robot_trans, rel_pt)

    # @profile
    def rel_ee_pos_check_jac(self, x, rel_pt):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rel_pos_error_jac(
            obj_trans, robot_trans, axises, arm_joints, rel_pt
        )

    # @profile
    def rel_pos_error_f(self, obj_trans, robot_trans, rel_pt):
        """
        This function calculates the value of the displacement between center of gripper and a point relative to the object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        rel_pt: offset between your target point and object's pose
        """
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos - obj_pos).reshape((3, 1))
        return dist_val

    # @profile
    def rel_pos_error_jac(self, obj_trans, robot_trans, axises, arm_joints, rel_pt):
        """
        This function calculates the jacobian of the displacement between center of gripper and a point relative to the object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        rel_pt: offset between your target point and object's pose
        """
        gp = rel_pt
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        # Calculate the joint jacobian
        arm_jac = []
        for jnt_id in arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            arm_jac.apend(np.cross(axis, robot_pos - (parent_pos + parent_frame_pos)))
        arm_jac = np.array(arm_jac).T
        # Calculate jacobian for the robot base
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos).reshape((3, 1))
        # Calculate object jacobian
        obj_jac = (
            -1
            * np.array(
                [np.cross(axis, obj_pos - obj_trans[:3, 3]) for axis in axises]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm)

        return dist_jac

    # @profile
    def ee_rot_check_f(self, x, offset=np.eye(3)):
        """
        This function is used to check whether End Effective pose's rotational axis is parallel to that of robot gripper

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rot_lock_f(obj_trans, robot_trans, offset)

    # @profile
    def ee_rot_check_jac(self, x):
        """
        This function is used to check whether End Effective pose's rotational axis is parallel to that of robot gripper

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints)

    # @profile
    def rot_lock_f(self, obj_trans, robot_trans, offset=np.eye(3)):
        """
        This function calculates the value of the angle
        difference between robot gripper's rotational axis and
        object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        """
        rot_vals = []
        local_dir = np.eye(3)
        for i in range(3):
            obj_dir = np.dot(obj_trans[:3, :3], local_dir[i])
            world_dir = robot_trans[:3, :3].dot(local_dir[i])
            rot_vals.append([np.dot(obj_dir, world_dir) - offset[i].dot(local_dir[i])])

        rot_val = np.vstack(rot_vals)
        return rot_val

    # @profile
    def rot_lock_jac(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the jacobian of the angle
        difference between robot gripper's rotational axis and
        object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        rot_jacs = []
        for local_dir in np.eye(3):
            obj_dir = np.dot(obj_trans[:3, :3], local_dir)
            world_dir = robot_trans[:3, :3].dot(local_dir)
            # computing robot's jacobian
            arm_jac = []
            for jnt_id in arm_joints:
                info = p.getJointInfo(jnt_id)
                parent_id = info[-1]
                parent_frame_pos = info[14]
                axis = info[13]
                parent_info = p.getLinkState(robot_body.body_id, parent_id)
                parent_pos = parent_info[0]
                arm_jac.apend(np.dot(obj_dir, np.cross(axis, world_dir)))
            arm_jac = np.array(arm_jac).T

            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0, 0, 1], world_dir)))
            base_jac = base_jac.reshape((1, 1))
            # computing object's jacobian
            obj_jac = np.array(
                [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
            )
            obj_jac = np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix
            rot_jacs.append(self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm))
        rot_jac = np.vstack(rot_jacs)
        return rot_jac

    # @profile
    def pos_check_f(self, x, rel_pt=np.zeros((3,))):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rel_pos_error_f(obj_trans, robot_trans, rel_pt)

    # @profile
    def pos_check_jac(self, x, rel_pt=np.zeros((3,))):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        return self.rel_pos_error_jac(
            obj_trans, robot_trans, axises, arm_joints, rel_pt
        )

    # @profile
    def rot_check_f(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        local_dir = np.array([0.0, 0.0, 1.0])

        return self.rot_error_f(obj_trans, robot_trans, local_dir)

    # @profile
    def rot_check_jac(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        local_dir = np.array([0.0, 0.0, 1.0])

        return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints, local_dir)

    # @profile
    def rot_error_f(self, obj_trans, robot_trans, local_dir, robot_dir=None):
        """
        This function calculates the value of the rotational error between
        robot gripper's rotational axis and object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        if robot_dir is None:
            robot_dir = local_dir
        obj_dir = np.dot(obj_trans[:3, :3], local_dir)
        world_dir = robot_trans[:3, :3].dot(robot_dir)
        obj_dir = obj_dir / np.linalg.norm(obj_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]])
        return rot_val

    # @profile
    def rot_error_jac(
        self, obj_trans, robot_trans, axises, arm_joints, local_dir, robot_dir=None
    ):
        """
        This function calculates the jacobian of the rotational error between
        robot gripper's rotational axis and object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        if robot_dir is None:
            robot_dir = local_dir
        obj_dir = np.dot(obj_trans[:3, :3], local_dir)
        world_dir = robot_trans[:3, :3].dot(robot_dir)
        obj_dir = obj_dir / np.linalg.norm(obj_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        sign = np.sign(np.dot(obj_dir, world_dir))
        # computing robot's jacobian
        arm_jac = []
        for jnt_id in arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            arm_jac.apend(np.dot(obj_dir, np.cross(axis, sign * world_dir)))
        arm_jac = np.array(arm_jac).T

        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = sign * np.array(
            np.dot(obj_dir, np.cross([0, 0, 1], world_dir))
        ).reshape((1, 1))
        # computing object's jacobian
        obj_jac = np.array(
            [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
        )
        obj_jac = sign * np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
        # Create final 1x23 jacobian matrix
        rot_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm)
        return rot_jac

    # @profile
    def both_arm_pos_check_f(self, x):
        """
        This function is used to check whether:
            basket is at both robot gripper's center

        x: 26 dimensional list aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, "left")
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, "right")
        l_arm_joints = l_arm_inds
        r_arm_joints = r_arm_inds

        rel_pt = np.array([0, 2 * const.BASKET_OFFSET, 0])
        # rel_pt = np.array([0, 2*const.BASKET_NARROW_OFFSET,0])
        l_pos_val = self.rel_pos_error_f(r_ee_trans, l_ee_trans, rel_pt)
        rel_pt = np.array([0, -2 * const.BASKET_OFFSET, 0])
        # rel_pt = np.array([0, -2*const.BASKET_NARROW_OFFSET,0])
        r_pos_val = self.rel_pos_error_f(l_ee_trans, r_ee_trans, rel_pt)
        rel_pt = np.array([const.BASKET_OFFSET, self.grip_offset, 0])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        obj_pos_val = self.rel_pos_error_f(obj_trans, l_ee_trans, rel_pt)
        # import ipdb; ipdb.set_trace()
        return np.vstack([l_pos_val, r_pos_val, obj_pos_val])

    # @profile
    def both_arm_pos_check_jac(self, x):
        """
        This function is used to check whether:
            basket is at both robot gripper's center

        x: 26 dimensional list aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """

        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, "left")
        l_arm_joints = l_arm_inds
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, "right")
        r_arm_joints = r_arm_inds
        # left_arm_focused
        rel_pt = np.array([0, 2 * const.BASKET_OFFSET, 0])
        # rel_pt = np.array([0,2*const.BASKET_NARROW_OFFSET,0])
        robot_pos = l_ee_trans[:3, 3]
        obj_pos = np.dot(r_ee_trans, np.r_[rel_pt, 1])[:3]

        l_arm_jac = []
        for jnt_id in l_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            l_arm_jac.apend(np.cross(axis, robot_pos - (parent_pos + parent_frame_pos)))
        l_arm_jac = np.array(l_arm_jac).T

        r_arm_jac = []
        for jnt_id in r_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            r_arm_jac.apend(np.cross(axis, obj_pos - (parent_pos + parent_frame_pos)))
        r_arm_jac = -np.array(r_arm_jac).T

        l_pos_jac = np.hstack(
            [l_arm_jac, np.zeros((3, 1)), r_arm_jac, np.zeros((3, 8))]
        )
        # right_arm_focused
        rel_pt = np.array([0, -2 * const.BASKET_OFFSET, 0])
        # rel_pt = np.array([0,-2*const.BASKET_NARROW_OFFSET,0])
        robot_pos = r_ee_trans[:3, 3]
        obj_pos = np.dot(l_ee_trans, np.r_[rel_pt, 1])[:3]

        l_arm_jac = []
        for jnt_id in l_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            l_arm_jac.apend(np.cross(axis, obj_pos - (parent_pos + parent_frame_pos)))
        l_arm_jac = -np.array(l_arm_jac).T

        r_arm_jac = []
        for jnt_id in r_arm_joints:
            info = p.getJointInfo(jnt_id)
            parent_id = info[-1]
            parent_frame_pos = info[14]
            axis = info[13]
            parent_info = p.getLinkState(robot_body.body_id, parent_id)
            parent_pos = parent_info[0]
            r_arm_jac.apend(np.cross(axis, robot_pos - (parent_pos + parent_frame_pos)))
        r_arm_jac = np.array(r_arm_jac).T

        r_pos_jac = np.hstack(
            [l_arm_jac, np.zeros((3, 1)), r_arm_jac, np.zeros((3, 8))]
        )

        self.arm = "left"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6:-3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([const.BASKET_OFFSET, self.grip_offset, 0])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        obj_pos_jac = self.rel_pos_error_jac(
            obj_trans, l_ee_trans, axises, arm_joints, rel_pt
        )

        return np.vstack([l_pos_jac, r_pos_jac, obj_pos_jac])

    # @profile
    def both_arm_rot_check_f(self, x):
        """
        This function is used to check whether:
            object is at robot gripper's center

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        offset = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]])
        # offset = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6:-3], x[-3:])
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        l_rot_val = self.rot_lock_f(obj_trans, robot_trans, offset)
        self.arm = "right"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        r_rot_val = self.rot_lock_f(obj_trans, robot_trans, offset)

        return np.vstack([l_rot_val, r_rot_val])

    # @profile
    def both_arm_rot_check_jac(self, x):
        """
        This function is used to check whether:
            object is at robot gripper's center

        x: 26 dimensional list aligned in following order,
        BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->canPose->canRot

        Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6:-3], x[-3:])
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        l_rot_jac = self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints)
        self.arm = "right"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        r_rot_jac = self.rot_lock_jac(obj_trans, robot_trans, axises, arm_joints)

        return np.vstack([l_rot_jac, r_rot_jac])

    # @profile
    # def vel_check(self, x):
    # """
    # Check whether end effector are within range
    # """
    # jac = np.zeros((12, 40))
    # robot_body = self._param_to_body[self.params[self.ind0]]
    # robot = robot_body.env_body
    # # Set poses and Get transforms

    # left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
    # right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]

    # left_pose_rot =robot_left_trans[:3,3]
    # left_arm_jac = np.array([np.cross(joint.GetAxis(), left_pose_rot[:3] - joint.GetAnchor()) for joint in left_arm_joints]).T.copy()
    # left_base_jac = np.cross(np.array([0, 0, 1]), left_pose_rot[:3] - np.zeros((3,))).reshape((3,))

    # jac[0:3, 0:7] = -left_arm_jac
    # jac[0:3, 16] = -left_base_jac
    # jac[0:3, 17:20] = -np.eye(3)

    # jac[3:6, 0:7] = left_arm_jac
    # jac[3:6, 16] = left_base_jac
    # jac[3:6, 17:20] = -np.eye(3)

    # right_pose_rot =robot_right_trans[:3,3]
    # right_arm_jac = np.array([np.cross(joint.GetAxis(), right_pose_rot[:3] - joint.GetAnchor()) for joint in right_arm_joints]).T.copy()
    # right_base_jac = np.cross(np.array([0, 0, 1]), right_pose_rot[:3] - np.zeros((3,))).reshape((3,))

    # jac[6:9, 8:15] = -right_arm_jac
    # jac[6:9, 16] = -right_base_jac
    # jac[9:12, 8:15] = right_arm_jac
    # jac[9:12, 16] = right_base_jac

    # self.set_robot_poses(x[20:37], robot_body)
    # robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
    # robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
    # # Added here just in case
    # left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
    # right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]

    # left_new_pose_rot =robot_left_trans[:3,3]
    # left_new_arm_jac = np.array([np.cross(joint.GetAxis(), left_new_pose_rot[:3] - joint.GetAnchor()) for joint in left_arm_joints]).T.copy()
    # left_new_base_jac = np.cross(np.array([0, 0, 1]), left_new_pose_rot[:3] - np.zeros((3,))).reshape((3,))
    # jac[0:3, 20:27] = left_new_arm_jac
    # jac[0:3, 36] = left_new_base_jac
    # jac[3:6, 20:27] = -left_new_arm_jac
    # jac[3:6, 36] = -left_new_base_jac

    # right_new_pose_rot = robot_right_trans[:3,3]
    # right_new_arm_jac = np.array([np.cross(joint.GetAxis(), right_new_pose_rot[:3] - joint.GetAnchor()) for joint in right_arm_joints]).T.copy()
    # right_new_base_jac = np.cross(np.array([0, 0, 1]), right_new_pose_rot[:3] - np.zeros((3,))).reshape((3,))
    # jac[6:9, 28:35] = right_new_arm_jac
    # jac[6:9, 36] = right_new_base_jac
    # jac[6:9, 37:40] = -np.eye(3)
    # jac[9:12, 28:35] = -right_new_arm_jac
    # jac[9:12, 36] = -right_new_base_jac
    # jac[9:12, 37:40] = -np.eye(3)

    # dist_left = (left_new_pose_rot - left_pose_rot - x[17:20].flatten()).reshape((3,1))
    # dist_left_rev = (left_pose_rot - left_new_pose_rot - x[17:20].flatten()).reshape((3,1))
    # dist_right = (right_new_pose_rot - right_pose_rot - x[37:40].flatten()).reshape((3,1))
    # dist_right_rev = (right_pose_rot - right_new_pose_rot - x[37:40].flatten()).reshape((3,1))

    # val = np.vstack([dist_left, dist_left_rev, dist_right, dist_right_rev])
    # return val, jac


class At(ExprPredicate):
    """
    Format: # At, Can, Target

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict(
            [
                (
                    self.obj,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
                (
                    self.target,
                    [
                        ("value", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
            ]
        )

        # A = np.c_[np.eye(6), -np.eye(6)]
        # b, val = np.zeros((6, 1)), np.zeros((6, 1))
        # aff_e = AffExpr(A, b)
        # e = EqExpr(aff_e, val)

        A = np.c_[np.r_[np.eye(6), -np.eye(6)], np.r_[-np.eye(6), np.eye(6)]]
        b, val = np.zeros((12, 1)), np.ones((12, 1)) * 1e-2
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(At, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class AtPose(ExprPredicate):
    """
    Format: # At, Can, Target

    Non-robot related
    """

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict(
            [
                (self.obj, [("pose", np.array([0, 1, 2], dtype=np.int))]),
                (self.target, [("value", np.array([0, 1, 2], dtype=np.int))]),
            ]
        )

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b, val = np.zeros((6, 1)), np.ones((6, 1)) * 1e-3
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(AtPose, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class Near(ExprPredicate):
    """
    Format: # At, Can, Target

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict(
            [
                (
                    self.obj,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
                (
                    self.target,
                    [
                        ("value", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
            ]
        )

        # A = np.c_[np.eye(6), -np.eye(6)]
        # b, val = np.zeros((6, 1)), np.zeros((6, 1))
        # aff_e = AffExpr(A, b)
        # e = EqExpr(aff_e, val)

        A = np.c_[np.r_[np.eye(6), -np.eye(6)], np.r_[-np.eye(6), np.eye(6)]]
        b, val = np.zeros((12, 1)), np.ones((12, 1)) * 1e-1
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(At, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class HLAnchor(ExprPredicate):
    """
    Format: # HLAnchor, RobotPose, RobotPose

    Non-robot related
    Should Always return True
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        attr_inds = self.attr_inds

        A = np.zeros((self.attr_dim, self.attr_dim))
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLAnchor, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class RobotAt(ExprPredicate):
    """
    Format: RobotAt, Robot, RobotPose

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.robot, self.robot_pose = params

        # A = np.c_[np.r_[np.eye(self.attr_dim), -np.eye(self.attr_dim)], np.r_[-np.eye(self.attr_dim), np.eye(self.attr_dim)]]
        # b, val = np.zeros((self.attr_dim*2, 1)), np.ones((self.attr_dim*2, 1))*1e-3
        # aff_e = AffExpr(A, b)
        # e = LEqExpr(aff_e, val)

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(RobotAt, self).__init__(
            name, e, self.attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class IsMP(ExprPredicate):
    """
    Format: IsMP Robot (Just the Robot Base)

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        setup_mov_limit_check[Function]: function that sets constraint matrix
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        (self.robot,) = params
        ## constraints  |x_t - x_{t+1}| < dmove
        ## ==> x_t - x_{t+1} < dmove, -x_t + x_{t+a} < dmove
        attr_inds = self.attr_inds
        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(IsMP, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class WithinJointLimit(ExprPredicate):
    """
    Format: WithinJointLimit Robot

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        setup_mov_limit_check[Function]: function that sets constraint matrix
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        (self.robot,) = params
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        A, b, val = self.setup_mov_limit_check()
        e = LEqExpr(AffExpr(A, b), val)
        super(WithinJointLimit, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = False


class Stationary(ExprPredicate):
    """
    Format: Stationary, Can

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        (self.obj,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.obj,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                )
            ]
        )

        A = np.c_[np.eye(6), -np.eye(6)]
        b, val = np.zeros((6, 1)), np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class StationaryBase(ExprPredicate):
    """
    Format: StationaryBase, Robot (Only Robot Base)

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        (self.robot,) = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryBase, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class StationaryArms(ExprPredicate):
    """
    Format: StationaryArms, Robot (Only Robot Arms)

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        (self.robot,) = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryArms, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class StationaryW(ExprPredicate):
    """
    Format: StationaryW, Obstacle

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        (self.w,) = params
        attr_inds = OrderedDict(
            [
                (
                    self.w,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                )
            ]
        )
        A = np.c_[np.eye(6), -np.eye(6)]
        b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryW, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class StationaryNEq(ExprPredicate):
    """
    Format: StationaryNEq, Can, Can(Hold)

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.obj, self.obj_held = params
        attr_inds = OrderedDict(
            [
                (
                    self.obj,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                )
            ]
        )

        if self.obj.name == self.obj_held.name:
            A = np.zeros((1, 12))
            b = np.zeros((1, 1))
        else:
            A = np.c_[np.eye(6), -np.eye(6)]
            b = np.zeros((6, 1))
        e = EqExpr(AffExpr(A, b), b)
        super(StationaryNEq, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )
        self.spacial_anchor = False


class GraspValid(ExprPredicate):
    """
    Format: GraspValid EEPose Target

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(GraspValid, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class InContact(ExprPredicate):
    """
    Format: InContact Robot

    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
        GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
        GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params
        attr_inds = self.attr_inds

        A = np.eye(1).reshape((1, 1))
        b = np.zeros(1).reshape((1, 1))

        val = np.array([[self.GRIPPER_CLOSE]])
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = np.array([[self.GRIPPER_OPEN]])
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContact, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class InContacts(ExprPredicate):
    """
    Format: InContact Robot
    Robot related

    Requires:
        attr_inds[OrderedDict]: robot attribute indices
        attr_dim[Int]: dimension of robot attribute
        GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
        GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params
        attr_inds = self.attr_inds

        A = np.eye(2).reshape((2, 2))
        b = np.zeros((2, 1))

        val = np.array([[self.GRIPPER_CLOSE, self.GRIPPER_CLOSE]]).T
        aff_expr = AffExpr(A, b)
        e = EqExpr(aff_expr, val)

        aff_expr = AffExpr(A, b)
        val = np.array([[self.GRIPPER_OPEN, self.GRIPPER_OPEN]]).T
        self.neg_expr = EqExpr(aff_expr, val)

        super(InContacts, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class InGripper(PosePredicate):
    """
    Format: InGripper, Robot, Item

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

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot, self.obj = params

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
            self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom),
        }

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(InGripper, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            priority=2,
        )
        self.spacial_anchor = True


class AlmostInGripper(PosePredicate):
    """
    Format: AlmostInGripper, Robot, Item

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

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot, self.obj = params

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
            self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom),
        }

        e = LEqExpr(Expr(self.eval_f, self.eval_grad), self.max_dist.reshape(-1, 1))
        super(AlmostInGripper, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            priority=2,
        )
        self.spacial_anchor = True


class EEAt(PosePredicate):
    """
    Format: EEAt, Robot
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self._env = env
        (self.robot,) = params

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(EEAt, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            priority=2,
        )
        self.spacial_anchor = True


class GripperAt(PosePredicate):
    """
    Format: GripperAt, Robot, EEPose
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot, self.pose = params

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(GripperAt, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            priority=2,
        )
        self.spacial_anchor = True


class EEGraspValid(PosePredicate):

    # EEGraspValid EEPose Washer

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.ee_pose, self.robot = params

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(EEGraspValid, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            priority=0,
        )
        self.spacial_anchor = True


class EEReachable(PosePredicate):
    """
    Format: EEReachable Robot, StartPose, EEPose

    Robot related

    Requires:
        attr_inds[OrderedDict]: attribute indices for constructing x
        set_robot_poses[Function]:Function that sets robot's poses
        get_robot_info[Function]:Function that returns robot's transformations and arm indices
        eval_f[Function]:Function returns predicate's value
        eval_grad[Function]:Function returns predicate's gradient
        coeff[Float]:pose coeffitions
        rot_coeff[Float]:rotation coeffitions
    """

    # @profile
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        active_range=(-const.EEREACHABLE_STEPS, const.EEREACHABLE_STEPS),
        env=None,
        debug=False,
    ):
        assert len(params) == 3
        self._env = env
        self.robot, self.start_pose, self.ee_pose = params
        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }
        pos_expr = Expr(self.eval_f, self.eval_grad)
        e = EqExpr(pos_expr, np.zeros((self.eval_dim, 1)))

        super(EEReachable, self).__init__(
            name,
            e,
            self.attr_inds,
            params,
            expected_param_types,
            active_range=active_range,
            priority=1,
        )
        self.spacial_anchor = True

    # @profile
    def stacked_f(self, x):
        """
        Stacking values of all EEReachable timesteps in following order:
        pos_val(t - EEReachableStep)
        pos_val(t - EEReachableStep + 1)
        ...
        pos_val(t)
        rot_val(t)
        pos_val(t + 1)
        ...
        pos_val(t + EEReachableStep)
        """
        i, index = 0, 0
        f_res = []
        start, end = self.active_range
        offset = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        for s in range(start, end + 1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(
                self.coeff * self.rel_ee_pos_check_f(x[i : i + self.attr_dim], rel_pt)
            )
            if s == 0:
                f_res.append(
                    self.rot_coeff
                    * self.ee_rot_check_f(x[i : i + self.attr_dim], offset)
                )
            i += self.attr_dim

        return np.vstack(f_res)

    # @profile
    def stacked_grad(self, x):
        """
        Stacking jacobian of all EEReachable timesteps in following order:
        pos_jac(t - EEReachableStep)
        pos_jac(t - EEReachableStep + 1)
        ...
        pos_jac(t)
        rot_jac(t)
        pos_jac(t + 1)
        ...
        pos_jac(t + EEReachableStep)
        """
        start, end = self.active_range
        dim, step = 3, end + 1 - start
        i, j = 0, 0
        grad = np.zeros((dim * step + 3, self.attr_dim * step))
        for s in range(start, end + 1):
            rel_pt = self.get_rel_pt(s)
            grad[
                j : j + dim, i : i + self.attr_dim
            ] = self.coeff * self.rel_ee_pos_check_jac(x[i : i + self.attr_dim], rel_pt)
            j += dim
            if s == 0:
                grad[
                    j : j + 3, i : i + self.attr_dim
                ] = self.rot_coeff * self.ee_rot_check_jac(x[i : i + self.attr_dim])
                j += dim
            i += self.attr_dim

        return grad


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

    # @profile
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.COLLISION_TOL,
    ):
        assert len(params) == 4
        self._env = env
        self.robot, self.startp, self.endp, self.obstacle = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
            self.obstacle: self.lazy_spawn_or_body(
                self.obstacle, self.obstacle.name, self.obstacle.geom
            ),
        }

        f = lambda x: self.coeff * self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff * self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)

        self.col_link_pairs = [
            x
            for x in itertools.product(
                self.robot.geom.col_links, self.obstacle.geom.col_links
            )
        ]
        self.col_link_pairs = sorted(self.col_link_pairs)

        val = np.zeros((len(self.col_link_pairs), 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Obstructs, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=3,
            debug=debug,
            tol=tol,
            priority=3,
        )
        self.spacial_anchor = False

    # @profile
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

    # @profile
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.COLLISION_TOL,
    ):
        assert len(params) == 5
        self._env = env
        self.robot, self.startp, self.endp, self.obstacle, self.obj = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
            self.obstacle: self.lazy_spawn_or_body(
                self.obstacle, self.obstacle.name, self.obstacle.geom
            ),
            self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom),
        }

        # self.col_link_pairs = [x for x in itertools.product(self.robot.geom.col_links, self.obstacle.geom.col_links)]
        def exclude_f(c):
            c = list(c)
            return (
                ("left_gripper_r_finger_tip" in c or "left_gripper_r_finger" in c)
                and "short_1" in c
            ) or (
                ("right_gripper_l_finger_tip" in c or "right_gripper_l_finger" in c)
                and "short_2" in c
            )

        self.col_link_pairs = [
            x
            for x in itertools.product(
                self.robot.geom.col_links, self.obstacle.geom.col_links
            )
            if not exclude_f(x)
        ]
        self.col_link_pairs = sorted(self.col_link_pairs)

        self.obj_obj_link_pairs = [
            x
            for x in itertools.product(
                self.obj.geom.col_links, self.obstacle.geom.col_links
            )
        ]
        self.obj_obj_link_pairs = sorted(self.obj_obj_link_pairs)

        if self.obj.name == self.obstacle.name:
            links = len(self.col_link_pairs)

            col_fn, offset = (
                self.robot_obj_collision,
                const.DIST_SAFE - const.COLLISION_TOL,
            )
            val = np.zeros((links, 1))
        else:
            links = len(self.col_link_pairs) + len(self.obj_obj_link_pairs)
            col_fn, offset = self.robot_obj_held_collision, 0
            val = np.zeros((links, 1))

        f = lambda x: self.coeff * (col_fn(x)[0] - offset)
        grad = lambda x: self.coeff * col_fn(x)[1]
        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * (col_fn(x)[0] - offset)
        grad_neg = lambda x: self.neg_coeff * col_fn(x)[1]

        col_expr, col_expr_neg = Expr(f, grad), Expr(f_neg, grad_neg)
        e, self.neg_expr = LEqExpr(col_expr, val), LEqExpr(col_expr_neg, val)
        super(ObstructsHolding, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=3,
            debug=debug,
            tol=tol,
            priority=3,
        )
        self.spacial_anchor = False

    # @profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class Collides(CollisionPredicate):
    """
    Format: Collides Item Item

    Non-robot related
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.obj, self.obstacle = params
        attr_inds = OrderedDict(
            [
                (
                    self.obj,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
                (
                    self.obstacle,
                    [
                        ("pose", np.array([0, 1, 2], dtype=np.int)),
                        ("rotation", np.array([0, 1, 2], dtype=np.int)),
                    ],
                ),
            ]
        )
        self._param_to_body = {
            self.obj: self.lazy_spawn_or_body(self.obj, self.obj.name, self.obj.geom),
            self.obstacle: self.lazy_spawn_or_body(
                self.obstacle, self.obstacle.name, self.obstacle.geom
            ),
        }

        self.obj_obj_link_pairs = [
            x
            for x in itertools.product(
                self.obj.geom.col_links, self.obstacle.geom.col_links
            )
        ]
        self.obj_obj_link_pairs = sorted(self.obj_obj_link_pairs)

        links = len(self.obj_obj_link_pairs)

        f = lambda x: self.coeff * self.obj_obj_collision(x)[0]
        grad = lambda x: self.coeff * self.obj_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.obj_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.obj_obj_collision(x)[1]

        col_expr, val = Expr(f, grad), np.zeros((1, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(Collides, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            debug=debug,
            priority=3,
        )
        self.spacial_anchor = False

    # @profile
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

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot, self.obstacle = params
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            ),
            self.obstacle: self.lazy_spawn_or_body(
                self.obstacle, self.obstacle.name, self.obstacle.geom
            ),
        }

        f = lambda x: self.coeff * self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff * self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)

        self.col_link_pairs = [
            x
            for x in itertools.product(
                self.robot.geom.col_links, self.obstacle.geom.col_links
            )
        ]
        self.col_link_pairs = sorted(self.col_link_pairs)
        links = len(self.col_link_pairs)

        val = np.zeros((links, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(RCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=1, priority=3
        )
        self.spacial_anchor = False

    # @profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class RSelfCollides(CollisionPredicate):
    """
    Format: RCollides Robot

    Robot related

    Requires:
        attr_dim[Int]:number of attribute in robot's full pose
        attr_inds[OrderedDict]: robot attribute indices
        set_robot_poses[Function]:Function that sets robot's poses
        set_active_dof_inds[Function]:Function that sets robot's active dof indices
        RCOLLIDES_OPT_COEFF[Float]: Obstructs_holding coeffitions, used during optimazation problem
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self._env = env
        self.robot = params[0]
        # attr_inds for the robot must be in this exact order do to assumptions
        # in OpenRAVEBody's _set_active_dof_inds and the way OpenRAVE's
        # CalculateActiveJacobian for robots work (base pose is always last)
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        f = lambda x: self.coeff * self.robot_self_collision(x)[0]
        grad = lambda x: self.coeff * self.robot_self_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.robot_self_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.robot_self_collision(x)[1]

        col_expr = Expr(f, grad)

        self.col_link_pairs = [
            x
            for x in itertools.product(
                self.robot.geom.col_links, self.robot.geom.col_links
            )
        ]
        self.col_link_pairs = sorted(self.col_link_pairs)
        links = len(self.col_link_pairs)

        val = np.zeros((links, 1))
        e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

        super(RSelfCollides, self).__init__(
            name, e, attr_inds, params, expected_param_types, ind0=0, ind1=0, priority=3
        )
        self.spacial_anchor = False

    # @profile
    def get_expr(self, negated):
        if negated:
            return self.neg_expr
        else:
            return None


class BasketLevel(ExprPredicate):
    """
    Format: BasketLevel Basket
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[np.eye(self.attr_dim)]
        A[0, 0] = 0
        b, val = np.zeros((self.attr_dim, 1)), np.array([[0], [0], [np.pi / 2]])
        # b, val = np.zeros((self.attr_dim,1)), np.array([[np.pi/2], [0], [np.pi/2]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BasketLevel, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )

        self.spacial_anchor = False


class ObjectWithinRotLimit(ExprPredicate):
    """
    Format: ObjectWithinRotLimit Object
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.r_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = (
            np.zeros((self.attr_dim * 2, 1)),
            np.array([[np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]]).T,
        )
        pos_expr = AffExpr(A, b)
        e = LEqExpr(pos_expr, val)
        super(ObjectWithinRotLimit, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = False


class GrippersLevel(PosePredicate):
    """
    Format: GrippersLevel Robot
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 1
        self._env = env
        self.robot = params[0]
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        f = lambda x: self.coeff * self.eval_f(x)
        grad = lambda x: self.coeff * self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim, 1))
        e = EqExpr(pos_expr, val)

        super(GrippersLevel, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=3
        )
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

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot, self.ee_vel = params
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot: self.lazy_spawn_or_body(
                self.robot, self.robot.name, self.robot.geom
            )
        }

        f = lambda x: self.coeff * self.eval_f(x)
        grad = lambda x: self.coeff * self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim, 1))
        e = EqExpr(pos_expr, val)
        super(EERetiming, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            ind0=0,
            ind1=1,
            active_range=(0, 1),
            priority=3,
        )
        self.spacial_anchor = False


class ObjRelPoseConstant(ExprPredicate):
    """
    Format: ObjRelPoseConstant Basket Cloth
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        attr_inds = self.attr_inds
        A = np.c_[
            np.eye(self.attr_dim),
            -np.eye(self.attr_dim),
            -np.eye(self.attr_dim),
            np.eye(self.attr_dim),
        ]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(ObjRelPoseConstant, self).__init__(
            name,
            e,
            attr_inds,
            params,
            expected_param_types,
            active_range=(0, 1),
            priority=-2,
        )

        self.spacial_anchor = False


class IsPushing(PosePredicate):
    """
    Format: IsPushing Robot Robot
    """

    # @profile
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 2
        self._env = env
        self.robot1 = params[0]
        self.robot2 = params[1]
        attr_inds = self.attr_inds

        self._param_to_body = {
            self.robot1: self.lazy_spawn_or_body(
                self.robot1, self.robot1.name, self.robot1.geom
            ),
            self.robot2: self.lazy_spawn_or_body(
                self.robot2, self.robot2.name, self.robot2.geom
            ),
        }

        f = lambda x: self.coeff * self.eval_f(x)
        grad = lambda x: self.coeff * self.eval_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.eval_dim, 1))
        e = EqExpr(pos_expr, val)

        super(IsPushing, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=1
        )
        self.spacial_anchor = False


# class CloseGripper(ExprPredicate):
#     """
#         Format: InContact Robot

#         Robot related

#         Requires:
#             attr_inds[OrderedDict]: robot attribute indices
#             attr_dim[Int]: dimension of robot attribute
#             GRIPPER_CLOSE[Float]: Constants, specifying gripper value when gripper is closed
#             GRIPPER_OPEN[Float]: Constants, specifying gripper value when gripper is open
#     """
#     #@profile
#     def __init__(self, name, params, expected_param_types, env=None, debug=False):
#         self._env = env
#         self.robot = params
#         attr_inds = self.attr_inds

#         A = np.eye(1).reshape((1,1))
#         b = np.zeros(1).reshape((1,1))

#         val = np.array([[self.GRIPPER_CLOSE]])
#         aff_expr = AffExpr(A, b)
#         e = LEqExpr(aff_expr, val)

#         aff_expr = AffExpr(-1*A, b)
#         val = np.array([[-1*self.GRIPPER_CLOSE]])
#         self.neg_expr = LEqExpr(aff_expr, val)

#         super(CloseGripper, self).__init__(name, e, attr_inds, params, expected_param_types, priority = -2)
#         self.spacial_anchor = True
