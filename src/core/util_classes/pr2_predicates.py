from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.matrix import Vector3d, PR2PoseVector
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.pr2 import PR2
from sco.expr import Expr, AffExpr, EqExpr
from collections import OrderedDict
import numpy as np
import ctrajoptpy

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

class CollisionPredicate(ExprPredicate):
    def __init__(self, name, e, attr_inds, params, expected_param_types, dsafe = 0.05, debug = False, ind0=0, ind1=1):
        self._debug = debug
        if self._debug:
            self._env.SetViewer("qtcoin")
        self._cc = ctrajoptpy.GetCollisionChecker(self._env)
        self.dsafe = dsafe
        self.ind0 = ind0
        self.ind1 = ind1
        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types)

    def distance_from_obj(self, x):
        # self._cc.SetContactDistance(self.dsafe + .1)
        self._cc.SetContactDistance(np.Inf)
        p0 = self.params[self.ind0]
        p1 = self.params[self.ind1]
        b0 = self._param_to_body[p0]
        b1 = self._param_to_body[p1]
        pose0 = x[0:3]
        pose1 = x[3:6]
        b0.set_pose(pose0)
        b1.set_pose(pose1)

        collisions = self._cc.BodyVsBody(b0.env_body, b1.env_body)

        col_val, jac0, jac1 = self._calc_grad_and_val(p0.name, p1.name, pose0, pose1, collisions)
        val = np.array([col_val])
        jac = np.r_[jac0, jac1].reshape((1, 6))
        return val, jac


    def _calc_grad_and_val(self, name0, name1, pose0, pose1, collisions):
        val = -1 * float("inf")
        jac0 = None
        jac1 = None
        for c in collisions:
            linkA = c.GetLinkAParentName()
            linkB = c.GetLinkBParentName()

            if linkA == name0 and linkB == name1:
                pt0 = c.GetPtA()
                pt1 = c.GetPtB()
            elif linkB == name0 and linkA == name1:
                pt0 = c.GetPtB()
                pt1 = c.GetPtA()
            else:
                continue

            distance = c.GetDistance()
            normal = c.GetNormal()

            # plotting
            if self._debug:
                pt0[2] = 1.01
                pt1[2] = 1.01
                self._plot_collision(pt0, pt1, distance)
                print "pt0 = ", pt0
                print "pt1 = ", pt1
                print "distance = ", distance

            # if there are multiple collisions, use the one with the greatest penetration distance
            if self.dsafe - distance > val:
                val = self.dsafe - distance
                jac0 = -1 * normal[0:2]
                jac1 = normal[0:2]

        return val, jac0, jac1

    def gripper_can_displacement(self, obj_body, arm_joints, tool_link):
        # Calculate the value and the jacobian regarding displacement between center of gripper and center of can
        robot_pos = tool_link.GetTransform()[:3, 3]
        obj_trans = obj_body.env_body.GetTransform()
        obj_trans[2,3] = obj_trans[2,3] + self.dsafe
        obj_pos = obj_trans[:3,3]
        dist_val = robot_pos.flatten() - obj_pos.flatten()
        # Calculate the joint jacobian, and create the giant 3x20 matrix corresponding 20 pose values in the robot
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        # Calculate jacobian for the robot base

        base_jac = np.eye(3)
        base_jac[2,2] = 0
        dist_jac = np.hstack((base_jac, np.zeros((3, 9)), arm_jac, np.zeros((3, 1)), np.eye(3)))

        return (dist_val, dist_jac)

    def face_up(self, tool_link, arm_joints):
        # calculate the value and jacobian regarding direction of which the gripper is facing
        local_dir = np.array([0.,0.,1.])
        face_val = tool_link.GetTransform()[:2,:3].dot(local_dir)
        # Calculate the joint jacobian with respect to the gripper direction
        world_dir = tool_link.GetTransform()[:3,:3].dot(local_dir)
        # Originally in the planopt codebase, it only creates 2x7 matrix -> Don't know the reason why
        # face_rarm_jac = np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()
        arm_jac = np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()



        face_jac = np.hstack((np.zeros((2, 12)), arm_jac, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1))))
        return (face_val, face_jac)

    def _plot_collision(self, ptA, ptB, distance):
        self.handles = []
        if not np.allclose(ptA, ptB, atol=1e-3):
            if distance < 0:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(1, 0, 0)))
            else:
                self.handles.append(self._env.drawarrow(p1=ptA, p2=ptB, linewidth=.01, color=(0, 0, 0)))

class At(ExprPredicate):

    # At, Can, Location

    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.can, self.targ = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1,2], dtype=np.int))]),
                                 (self.targ, [("value", np.array([0,1,2], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))
        val = np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types)

class RobotAt(ExprPredicate):

    # RobotAt, Robot, RobotPose -> Every pose value of robot matches that of robotPose

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
        b = np.zeros((20, 1))
        val = np.zeros((20, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(RobotAt, self).__init__(name, e, attr_inds, params, expected_param_types)

class IsGP(CollisionPredicate):

    # IsGP, Robot, RobotPose, Can
    # 1. Center of can is at center of gripper Done
    # 2. gripper must face up Done
    # 3. There is no collision between gripper and can (Maybe a safety distance dsafe between robot and can)
    # MAYBE NOT 3

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
                                 (self.can, [("pose", np.array([0,1,2], dtype=np.int))])])
        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.can: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((3, 1))
        e = EqExpr(col_expr, val)
        super(IsGP, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

    def distance_from_obj(self, x):
        # Setting pose for each ravebody
        # Assuming x is aligned according to the following order:
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose
        robot_body = self._param_to_body[self.robot_pose]
        obj_body = self._param_to_body[self.can]
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pose = x[20:]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pose)
        # Helper variables that will be used in many places
        tool_link = robot.GetLink("r_gripper_tool_frame")
        rarm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        rarm_joints = [robot.GetJointFromDOFIndex(ind) for ind in rarm_inds]
        # Two function call returns value and jacobian of each requirement
        dist_val, dist_jac = self.gripper_can_displacement(obj_body, rarm_joints, tool_link)
        face_val, face_jac = self.face_up(tool_link, rarm_joints)
        tool_link = robot.GetLink("r_gripper_tool_frame")
        rarm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        rarm_joints = [robot.GetJointFromDOFIndex(ind) for ind in rarm_inds]

        # So far, only make IsGP test on displacement
        return (dist_val, dist_jac)

        #return the one that has the biggest error
        # if np.linalg.norm(dist_val) > np.linalg.norm(face_val):
        #     return (dist_val, dist_jac)
        # else:
        #     return (face_val, face_jac)

class IsPDP(CollisionPredicate):

    # IsPDP, Robot, RobotPose, Can, Location
    # same as IsGP

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 4
        self._env = env
        self.robot, self.robot_pose, self.can, self.location = params
        attr_inds = {self.robot_pose: [("value", np.array([0,1,2], dtype=np.int)),
                                        ("backHeight", np.array([0], dtype=np.int)),
                                        ("lArmPose", np.array(range(7), dtype=np.int)),
                                        ("lGripper", np.array([0], dtype=np.int)),
                                        ("rArmPose", np.array(range(7), dtype=np.int)),
                                        ("rGripper", np.array([0], dtype=np.int))],
                     self.location: [("value", np.array([0,1,2], dtype=np.int))]}
        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.location: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((3, 1))
        e = EqExpr(col_expr, val)
        super(IsPDP, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

    def distance_from_obj(self, x):
        # Setting pose for each ravebody
        # Assuming x is aligned according to the following order:
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose
        robot_body = self._param_to_body[self.robot_pose]
        obj_body = self._param_to_body[self.location]
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pose = x[20:]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pose)
        # Helper variables that will be used in many places
        tool_link = robot.GetLink("r_gripper_tool_frame")
        rarm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        rarm_joints = [robot.GetJointFromDOFIndex(ind) for ind in rarm_inds]
        # Two function call returns value and jacobian of each requirement
        base_link = robot.GetLink("base_link")

        dist_val, dist_jac = self.gripper_can_displacement(obj_body, rarm_joints, tool_link)
        face_val, face_jac = self.face_up(tool_link, rarm_joints)

        #So far make IsPDP only test on facing
        return (face_val, face_jac)

        # return the one that has the biggest error
        # if np.linalg.norm(dist_val) > np.linalg.norm(face_val):
        #     return (dist_val, dist_jac)
        # else:
        #     return (face_val, face_jac)

class IsGPError(CollisionPredicate):

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        assert len(params) == 4
        self._env = env
        self.robot, self.robot_pose, self.can, self.location = params
        attr_inds = {self.robot_pose: [("value", np.array([0,1,2], dtype=np.int)),
                                        ("backHeight", np.array([0], dtype=np.int)),
                                        ("lArmPose", np.array(range(7), dtype=np.int)),
                                        ("lGripper", np.array([0], dtype=np.int)),
                                        ("rArmPose", np.array(range(7), dtype=np.int)),
                                        ("rGripper", np.array([0], dtype=np.int))],
                     self.location: [("value", np.array([0,1,2], dtype=np.int))]}
        self._param_to_body = {self.robot_pose: self.lazy_spawn_or_body(self.robot_pose, self.robot_pose.name, self.robot.geom),
                               self.location: self.lazy_spawn_or_body(self.can, self.can.name, self.can.geom)}

        f = lambda x: self.distance_from_obj(x)[0]
        grad = lambda x: self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((3, 1))
        e = EqExpr(col_expr, val)
        super(IsPDP, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)

    def distance_from_obj(self, x):
        # Setting pose for each ravebody
        # Assuming x is aligned according to the following order:
        # BasePose->BackHeight->LeftArmPose->LeftGripper->RightArmPose->RightGripper->CanPose
        robot_body = self._param_to_body[self.robot_pose]
        obj_body = self._param_to_body[self.location]
        base_pose, back_height = x[0:3], x[3]
        l_arm_pose, l_gripper = x[4:11], x[11]
        r_arm_pose, r_gripper = x[12:19], x[19]
        can_pose = x[20:]
        robot = robot_body.env_body
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)
        obj_body.set_pose(can_pose)
        # Helper variables that will be used in many places
        tool_link = robot.GetLink("r_gripper_tool_frame")
        rarm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        rarm_joints = [robot.GetJointFromDOFIndex(ind) for ind in rarm_inds]
        # Two function call returns value and jacobian of each requirement
        dist_val, dist_jac = self.gripper_can_displacement(obj_body, rarm_joints, tool_link)
        face_val, face_jac = self.face_up(tool_link, rarm_joints)

        #return the one that has the bigger error
        if np.linalg.norm(dist_val) > np.linalg.norm(face_val):
            return (dist_val, dist_jac)
        else:
            return (face_val, face_jac)





class InGripper(ExprPredicate):

    # InGripper, Robot, Can, Grasp

    def __init__(self, name, params, expected_param_types):
        self.robot, self.can, self.grasp = params
        attr_inds = {self.robot: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     self.can: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     self.grasp: [("value", np.array([0, 1, 2], dtype=np.int))]}
        # want x0 - x2 = x4, x1 - x3 = x5
        A = np.c_[np.eye(3), -np.eye(3)]
        b = np.zeros((3, 1))

        e = AffExpr(A, b)
        e = EqExpr(e, np.zeros((3,1)))

        super(InGripper, self).__init__(name, e, attr_inds, params, expected_param_types)

    def test(self, time = 0):
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.expr.eval(self.get_param_vector(time))
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

    def pos_error(self,traj,debug=False, analytical=False):
        gp = np.array([0,0,.125])
        obj_K = 6
        obj_offset = self.T*self.K

        dim = 3
        val = np.zeros((self.T*dim, 1))
        jac = np.zeros((val.size, traj.size))

        for t in range(self.T):
            xt = traj[self.K*t:self.K*(t+1)]
            self.robot.set_pose(self.env, xt)

            ot = traj[obj_K*t + obj_offset:obj_K*(t+1) + obj_offset]
            self.obj.set_pose(self.env, ot)
            W_T_O = self.obj.get_transform(self.env)
            obj_pos = np.dot(W_T_O, np.r_[gp, 1])[:3]
            # obj_pos = ot[3:6].flatten() + gp
            # target[2] += 0.125
            self.hl_action.plot()


            robot_body = self.robot.get_env_body(self.env)
            # inds = robot_body.GetActiveDOFIndices()
            rarm_inds = robot_body.GetManipulator('rightarm').GetArmIndices()
            rarm_joints = [robot_body.GetJointFromDOFIndex(ind) for ind in rarm_inds]


            tool_link = robot_body.GetLink("r_gripper_tool_frame")
            link_ind = tool_link.GetIndex()

            robot_pos = tool_link.GetTransform()
            robot_pos = robot_pos[:3, 3]
            val_t = robot_pos.flatten() - obj_pos.flatten()

            rarm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in rarm_joints]).T.copy()
            base_jac = np.eye(3)
            base_jac[2,2] = 0
            bodypart_jac = {"rightarm": rarm_jac, "base": base_jac}
            robot_jac_t = self.robot.jac_from_bodypart_jacs(bodypart_jac, 3)

            # if np.isclose(r_wrist_flex_joint.GetValue(0), 0.) and analytical:
            #     import ipdb; ipdb.set_trace()
            if debug:
                handles = []
                r_wrist_flex_joint = robot_body.GetJoint('r_wrist_flex_joint')
                point = r_wrist_flex_joint.GetAnchor()
                print np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in [r_wrist_flex_joint]]).T.copy()
                handles.append(self.hl_action.hl_plan.env.plot3(points=point,pointsize=25.0,colors=(0,0,1,0.2)))
                handles.append(self.hl_action.hl_plan.env.plot3(points=robot_pos,pointsize=25.0,colors=(0,1,1,0.2)))
                handles.append(self.hl_action.hl_plan.env.plot3(points=point + 0.1*r_wrist_flex_joint.GetAxis(),pointsize=25.0,colors=(0,1,1,0.2)))
                print "joint value:", r_wrist_flex_joint.GetValue(0)
                self.env.UpdatePublishedBodies()
                import ipdb; ipdb.set_trace()

            # import ipdb; ipdb.set_trace()

            axises = self.obj.get_axises(ot)
            obj_jac_t = -1*np.array([np.cross(axis, obj_pos - W_T_O[:3,3].flatten()) for axis in axises]).T
            obj_jac_t = np.c_[obj_jac_t, -np.eye(3)]
            # import ipdb; ipdb.set_trace()
            jac[dim*t:dim*(t+1), self.K*t:self.K*(t+1)] = robot_jac_t
            jac[dim*t:dim*(t+1), obj_K*t+obj_offset:obj_K*(t+1)+obj_offset] = obj_jac_t
            # import ipdb; ipdb.set_trace()
            val[dim*t:dim*(t+1)] = val_t.reshape((dim,1))
            # print "val: ", val

        return (10*val, 10*jac)

    def rot_error(self,traj,debug=False):
        dim = 1
        obj_K = 6
        obj_offset = self.T*self.K

        val = np.zeros((self.T*dim, 1))
        jac = np.zeros((val.size, traj.size))

        for t in range(self.T):
            xt = traj[self.K*t:self.K*(t+1)]
            self.robot.set_pose(self.env, xt)

            ot = traj[obj_K*t + obj_offset:obj_K*(t+1) + obj_offset]
            self.obj.set_pose(self.env, ot)
            W_T_O = self.obj.get_transform(self.env)
            self.hl_action.plot()

            robot_body = self.robot.get_env_body(self.env)

            tool_link = robot_body.GetLink("r_gripper_tool_frame")
            link_ind = tool_link.GetIndex()

            local_dir = np.array([0.,0.,1.])
            obj_dir = np.dot(W_T_O[:3,:3], local_dir)
            world_dir = tool_link.GetTransform()[:3,:3].dot(local_dir)

            # computing robot's jacobian
            rarm_inds = robot_body.GetManipulator('rightarm').GetArmIndices()
            rarm_joints = [robot_body.GetJointFromDOFIndex(ind) for ind in rarm_inds]
            rarm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir)) for joint in rarm_joints]).T.copy()
            rarm_jac = rarm_jac.reshape((1, len(rarm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir)))
            base_jac = np.array([[0, 0, base_jac]])
            bodypart_jac = {"rightarm": rarm_jac, "base": base_jac}
            robot_jac_t = self.robot.jac_from_bodypart_jacs(bodypart_jac, 1)

            r_wrist_flex_joint = robot_body.GetJoint('r_wrist_flex_joint')
            if debug:
                print "joint value:", r_wrist_flex_joint.GetValue(0)
                import ipdb; ipdb.set_trace()
            # computing object's jacobian
            axises = self.obj.get_axises(ot)
            obj_jac_t = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
            obj_jac_t = np.r_[obj_jac_t, [0,0,0]]

            jac[dim*t:dim*(t+1), self.K*t:self.K*(t+1)] = robot_jac_t
            jac[dim*t:dim*(t+1), obj_K*t+obj_offset:obj_K*(t+1)+obj_offset] = obj_jac_t
            val[dim*t:dim*(t+1)] = np.dot(obj_dir, world_dir) + 1

        return (val, jac)


class Obstructs(CollisionPredicate):

    # Obstructs, Robot, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        assert len(params) == 3
        self._env = env
        r, rp, c = params
        attr_inds = {r: [("pose", np.array([0, 1, 2], dtype=np.int))],
                     rp: [],
                     c: [("pose", np.array([0, 1, 2], dtype=np.int))]}
        self._param_to_body = {r: self.lazy_spawn_or_body(r, r.name, r.geom),
                               rp: self.lazy_spawn_or_body(rp, rp.name, r.geom),
                               c: self.lazy_spawn_or_body(c, c.name, c.geom)}
        f = lambda x: -self.distance_from_obj(x)[0]
        grad = lambda x: -self.distance_from_obj(x)[1]

        col_expr = Expr(f, grad)
        val = np.zeros((1,1))
        e = LEqExpr(col_expr, val)
        super(Obstructs, self).__init__(name, e, attr_inds, params, expected_param_types, ind0=1, ind1=2)
