from core.util_classes import robot_predicates
from core.util_classes.common_predicates import ExprPredicate
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import ctrajoptpy
from collections import OrderedDict
from openravepy import DOFAffine, Environment, quatRotateDirection, matrixFromQuat
from openravepy import (
    matrixFromAxisAngle,
    IkParameterization,
    IkParameterizationType,
    IkFilterOptions,
    Planner,
    RaveCreatePlanner,
    RaveCreateTrajectory,
    matrixFromAxisAngle,
    CollisionReport,
    RaveCreateCollisionChecker,
)
import numpy as np
import core.util_classes.hsr_constants as const
from core.util_classes.items import Box, Can, Sphere
from core.util_classes.param_setup import ParamSetup

# Attribute map used in hsr domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {
    "Robot": (
        ("arm", np.array(list(range(5)), dtype=np.int)),
        ("gripper", np.array([0], dtype=np.int)),
        ("pose", np.array([0, 1, 2], dtype=np.int)),
    ),
    "RobotPose": (
        ("arm", np.array(list(range(5)), dtype=np.int)),
        ("gripper", np.array([0], dtype=np.int)),
        ("value", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Rotation": [("value", np.array([0], dtype=np.int))],
    "Can": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "EEPose": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Target": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Table": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Obstacle": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Basket": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "BasketTarget": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "Washer": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
        ("door", np.array([0], dtype=np.int)),
    ),
    "WasherPose": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
        ("door", np.array([0], dtype=np.int)),
    ),
    "Cloth": (
        ("pose", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "ClothTarget": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "CanTarget": (
        ("value", np.array([0, 1, 2], dtype=np.int)),
        ("rotation", np.array([0, 1, 2], dtype=np.int)),
    ),
    "EEVel": (("value", np.array([0], dtype=np.int))),
    "Region": [("value", np.array([0, 1], dtype=np.int))],
}


def lin_interp_traj(start, end, time_steps):
    """
    This helper function returns a linear trajectory from start pose to end pose
    """
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps + 1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps + 1)
        traj[i, :] = traj_row
    return traj


def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    # param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]


"""
    Movement Constraints Family
"""


class HSRAt(robot_predicates.At):
    pass


class HSRClothAt(robot_predicates.At):
    pass


class HSRCanAt(robot_predicates.At):
    pass


class HSRClothAtPose(robot_predicates.AtPose):
    pass


class HSRRobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 9
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        super(HSRRobotAt, self).__init__(name, params, expected_param_types, env)


class HSRStacked(robot_predicates.ExprPredicate):

    # Stacked, Bottom, Top

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][0]]),
                (params[1], [ATTRMAP[params[1]._type][0]]),
            ]
        )
        self.attr_dim = 12
        self.bottom_can = params[0]
        self.top_can = params[1]
        A = np.r_[
            np.c_[np.eye(3), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 3))],
            np.c_[np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))],
            np.c_[np.zeros((3, 9)), np.eye(3)],
        ]

        b = np.zeros((self.attr_dim / 2, 1))
        val = np.array(
            [
                [0],
                [0],
                [self.bottom_can.geom.height / 2 + self.top_can.geom.height / 2],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ]
        )
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(HSRStacked, self).__init__(
            name, e, self.attr_inds, params, expected_param_types, priority=-2
        )


class HSRCansStacked(HSRStacked):
    pass


class HSRTargetOnTable(HSRStacked):
    pass


class HSRTargetsStacked(HSRStacked):
    pass


class HSRTargetCanStacked(HSRStacked):
    pass


class HSRIsMP(robot_predicates.IsMP):

    # IsMP Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        self.dof_cache = None
        super(HSRIsMP, self).__init__(name, params, expected_param_types, env, debug)

    # @profile
    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        dof_map = robot_body._geom.dof_map
        dof_inds = dof_map["arm"]
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((len(dof_inds), 1))
        active_lb = lb_limit[dof_inds].reshape((len(dof_inds), 1))
        joint_move = (active_ub - active_lb) / const.JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= const.BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/const.JOINT_MOVE_FACTOR
        val = np.vstack(
            (
                joint_move,
                2 * np.ones((1, 1)),
                const.BASE_MOVE * np.ones((const.BASE_DIM, 1)),
                joint_move,
                2 * np.ones((1, 1)),
                const.BASE_MOVE * np.ones((const.BASE_DIM, 1)),
            )
        )
        A = (
            np.eye(len(val))
            - np.eye(len(val), k=len(val) / 2)
            - np.eye(len(val), k=-len(val) / 2)
        )
        b = np.zeros((len(val), 1))
        self.base_step = const.BASE_MOVE * np.ones((const.BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val


class HSRWithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        super(HSRWithinJointLimit, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        dof_map = robot_body._geom.dof_map
        dof_inds = np.r_[dof_map["arm"], dof_map["gripper"]]
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((const.JOINT_DIM, 1))
        active_lb = lb_limit[dof_inds].reshape((const.JOINT_DIM, 1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(const.JOINT_DIM)
        A_up_limit = np.eye(const.JOINT_DIM)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2 * const.JOINT_DIM, 1))
        joint_move = (active_ub - active_lb) / const.JOINT_MOVE_FACTOR
        self.base_step = const.BASE_MOVE * np.ones((const.BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val


class HSRStationary(robot_predicates.Stationary):
    pass


class HSRStationaryCloth(robot_predicates.Stationary):
    pass


class HSRStationaryNeqCloth(robot_predicates.StationaryNEq):
    pass


class HSRStationaryCan(robot_predicates.Stationary):
    pass


class HSRStationaryNeqCan(robot_predicates.StationaryNEq):
    pass


class HSRStationaryWasher(robot_predicates.StationaryBase):

    # HSRStationaryWasher, Washer (Only pose, rotation)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][:2])])
        self.attr_dim = 6
        super(HSRStationaryWasher, self).__init__(
            name, params, expected_param_types, env
        )


class HSRStationaryWasherDoor(robot_predicates.StationaryBase):

    # HSRStationaryWasher, Washer (Only pose, rotation)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][2:])])
        self.attr_dim = 1
        super(HSRStationaryWasherDoor, self).__init__(
            name, params, expected_param_types, env
        )


class HSRStationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][-1:])])
        self.attr_dim = const.BASE_DIM
        super(HSRStationaryBase, self).__init__(name, params, expected_param_types, env)


class HSRStationaryBasePos(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][-1:])])
        self.attr_dim = const.BASE_DIM
        super(HSRStationaryBase, self).__init__(name, params, expected_param_types, env)


class HSRStationaryArm(robot_predicates.StationaryArms):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict(
            [(params[0], ["pose", np.array([0, 1], dtype=np.int)])]
        )
        self.attr_dim = const.JOINT_DIM + 1  # Keep gripper stationary as well
        super(HSRStationaryArms, self).__init__(name, params, expected_param_types, env)


class HSRStationaryRollJoints(robot_predicates.StationaryArms):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [("arm", np.array([2, 4]))])])
        self.attr_dim = 2
        super(HSRStationaryArms, self).__init__(name, params, expected_param_types, env)


class HSRStationaryEndJoints(robot_predicates.StationaryArms):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [("arm", np.array([1, 2, 3]))])])
        self.attr_dim = 3
        super(HSRStationaryEndJoints, self).__init__(
            name, params, expected_param_types, env
        )


class HSRStationaryWrist(robot_predicates.StationaryArms):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [("arm", np.array([4]))])])
        self.attr_dim = 1
        super(HSRStationaryWrist, self).__init__(
            name, params, expected_param_types, env
        )


class HSRStationaryLiftJoint(robot_predicates.StationaryArms):

    # StationaryArm, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [("arm", np.array([0]))])])
        self.attr_dim = 1
        super(HSRStationaryArms, self).__init__(name, params, expected_param_types, env)


class HSRStationaryW(robot_predicates.StationaryW):
    pass


class HSRStationaryNEq(robot_predicates.StationaryNEq):
    pass


"""
    Grasping Pose Constraints Family
"""


class HSRGraspValid(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][0]]),
                (params[1], [ATTRMAP[params[1]._type][0]]),
            ]
        )
        self.attr_dim = 3

        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim, 1)), np.zeros((self.attr_dim, 1))
        val[2] = const.HAND_DIST
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(HSRGraspValid, self).__init__(
            name, e, attr_inds, params, expected_param_types, priority=-2
        )
        self.spacial_anchor = True


class HSRCanGraspValid(HSRGraspValid):
    pass


class HSREEGraspValid(robot_predicates.EEGraspValid):

    # HSREEGraspValid EEPose Washer
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        self.coeff = const.EEGRASP_VALID_COEFF
        self.rot_coeff = const.EEGRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 3
        self.rel_pt = np.zeros((3,))
        self.rot_dir = np.array([0, 0, 1])
        super(HSREEGraspValid, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def set_washer_poses(self, x, washer_body):
        pose, rotation = x[-7:-4], x[-4:-1]
        door = x[-1]
        washer_body.set_pose(pose, rotation)
        washer_body.set_dof({"door": door})

    def get_washer_info(self, washer_body):
        tool_link = washer_body.env_body.GetLink("washer_handle")
        washer_trans = tool_link.GetTransform()
        washer_inds = [0]
        return washer_trans, washer_inds

    # @profile
    def washer_obj_kinematics(self, x):
        """
        This function is used to check whether End Effective pose's position is at robot gripper's center

        Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        # Setting the poses for forward kinematics to work
        self.set_washer_poses(x, robot_body)
        robot_trans, arm_inds = self.get_washer_info(robot_body)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(x[-7:-4], x[-4:-1])
        axises = [[0, 0, 1], np.dot(Rz, [0, 1, 0]), np.dot(Rz, np.dot(Ry, [1, 0, 0]))]

        ee_pos, ee_rot = x[:3], x[3:6]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        obj_axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]  # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return robot_trans, obj_trans, axises, obj_axises, arm_joints

    # @profile
    def washer_ee_check_f(self, x, rel_pt):
        (
            washer_trans,
            obj_trans,
            axises,
            obj_axises,
            arm_joints,
        ) = self.washer_obj_kinematics(x)
        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos - obj_pos).reshape((3, 1))
        return dist_val

    # @profile
    def washer_ee_check_jac(self, x, rel_pt):
        (
            washer_trans,
            obj_trans,
            axises,
            obj_axises,
            arm_joints,
        ) = self.washer_obj_kinematics(x)

        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]

        joint_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        washer_jac = np.array(
            [np.cross(axis, robot_pos - x[-7:-4, 0]) for axis in axises]
        ).T

        obj_jac = (
            -1
            * np.array(
                [np.cross(axis, obj_pos - obj_trans[:3, 3]) for axis in axises]
            ).T
        )
        dist_jac = np.hstack(
            [-np.eye(3), obj_jac, np.eye(3), washer_jac, 1 * joint_jac]
        )
        return dist_jac

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_f(x, self.rel_pt)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_jac(x, self.rel_pt)])


class HSRCloseGripper(robot_predicates.InContact):

    # HSRCloseGripperLeft Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        super(HSRCloseGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )


class HSROpenGripper(HSRCloseGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(HSROpenGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.expr = self.neg_expr


"""
Obstructs & Coliides Constraints
"""


class HSRObstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.DIST_SAFE,
    ):
        self.attr_dim = 9
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[3], list(ATTRMAP[params[3]._type])),
            ]
        )
        super(HSRObstructs, self).__init__(
            name, params, expected_param_types, env, debug, tol
        )
        self.dsafe = const.DIST_SAFE

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(
                const.COLLISION_DOF_INDICES, DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

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
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkRobot1, linkRobot2 = None, None
            sign = 0
            if linkAParent == robot_body.name and linkBParent == robot_body.name:
                ptRobot1, ptRobot2 = c.GetPtA(), c.GetPtB()
                linkRobot1, linkRobot2 = linkA, linkB
                sign = -1
            else:
                continue

            if linkRobot1 not in col_links or linkRobot2 not in col_links:
                continue

            if (
                not (linkRobot1.startswith("right") or linkRobot1.startswith("left"))
                or linkRobot1 == linkRobot2
                or linkRobot1.endswith("upper_shoulder")
                or linkRobot1.endswith("lower_shoulder")
                or linkRobot2.startswith("right")
                or linkRobot2.startswith("left")
            ):
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot1).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot1)

            grad = np.zeros((1, self.attr_dim))
            grad[:, : self.attr_dim - 3] = np.dot(sign * normal, robot_jac)
            col_vec = sign * normal

            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot1, linkRobot2)] = [
                self.dsafe - distance,
                grad,
                robot.GetLink(linkRobot1),
                robot.GetLink(linkRobot2),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot1, ptRobot2, distance)

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

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)

            grad = np.zeros((1, self.attr_dim + 6))
            grad[:, : self.attr_dim - 3] = np.dot(sign * normal, robot_jac)
            col_vec = -sign * normal
            grad[:, self.attr_dim - 3 : self.attr_dim - 1] = -col_vec[:2]
            grad[:, self.attr_dim - 1] = 0  # Don't try to rotate away
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
                robot.GetLink(linkRobot),
                robot.GetLink(linkObj),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

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


class HSRObstructsCan(HSRObstructs):
    pass


class HSRObstructsWasher(HSRObstructs):
    """
    This collision checks the washer as a solid cube
    """

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.DIST_SAFE,
    ):
        self.attr_dim = 9
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[3], list(ATTRMAP[params[3]._type])),
            ]
        )
        super(HSRObstructs, self).__init__(
            name, params, expected_param_types, env, debug, tol
        )
        self.dsafe = 1e-2  # const.DIST_SAFE
        # self.test_env = Environment()
        # self._cc = ctrajoptpy.GetCollisionChecker(self.test_env)
        # self._param_to_body = {}
        self.true_washer_body = self._param_to_body[params[3]]
        self._param_to_body[params[3]] = [
            OpenRAVEBody(self._env, "washer_obstruct", Box([0.375, 0.375, 0.375])),
            OpenRAVEBody(self._env, "obstruct_door", Can(0.35, 0.05)),
            OpenRAVEBody(
                self._env,
                "obstruct_handle",
                Sphere(
                    0.08,
                ),
            ),
        ]
        self._param_to_body[params[3]][0].set_pose([0, 0, 0])
        self._param_to_body[params[3]][1].set_pose([0, 0, 0])
        self._param_to_body[params[3]][2].set_pose([0, 0, 0])

        f = lambda x: self.coeff * self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff * self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff * self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)

        val = np.zeros((len(self.col_link_pairs), 1))
        # e = LEqExpr(col_expr, val)

        col_expr_neg = Expr(f_neg, grad_neg)
        self.neg_expr = LEqExpr(col_expr_neg, val)

    def robot_obj_collision(self, x):
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

        washer = self.params[self.ind1]
        washer_body = self._param_to_body[washer][0]
        washer_door = self._param_to_body[washer][1]
        washer_handle = self._param_to_body[washer][2]

        washer_pos, washer_rot, door_val = x[-7:-4], x[-4:-1], x[-1]
        self.true_washer_body.set_pose(washer_pos, washer_rot)
        self.true_washer_body.set_dof({"door": x[-1]})
        rot = washer_rot[0]
        x_offset = np.sin(rot) * 0.1
        y_offset = -np.cos(rot) * 0.1
        washer_body.set_pose(washer_pos - [[x_offset], [y_offset], [0]], washer_rot)
        door_trans = self.true_washer_body.env_body.GetLink(
            "washer_door"
        ).GetTransform()
        washer_door_pos = door_trans.dot([0, 0.025, 0, 1])[:3]
        washer_door.set_pose(
            washer_door_pos, [washer_rot[0] + [np.pi / 2 + x[-1]], np.pi / 2, 0]
        )
        handle_trans = self.true_washer_body.env_body.GetLink(
            "washer_handle"
        ).GetTransform()
        washer_handle.set_pose(handle_trans[:3, 3] + [0, 0, 0.02])

        # Make sure two body is in the same environment
        assert robot_body.env_body.GetEnv() == washer_body.env_body.GetEnv()
        self.set_active_dof_inds(robot_body, reset=False)
        # Setup collision checkers
        self._cc.SetContactDistance(const.MAX_CONTACT_DISTANCE)
        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, washer_body, collisions)

        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_door.env_body)
        door_col_val, door_col_jac = self._calc_grad_and_val(
            robot_body, washer_door, collisions
        )

        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_handle.env_body)
        handle_col_val, handle_col_jac = self._calc_grad_and_val(
            robot_body, washer_handle, collisions
        )
        # set active dof value back to its original state (For successive function call)
        self.set_active_dof_inds(robot_body, reset=True)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        washer_body.set_pose([0, 0, 0])
        washer_door.set_pose([0, 0, 0])
        washer_handle.set_pose([0, 0, 0])
        return (
            col_val + door_col_val + handle_col_val,
            col_jac + door_col_jac + handle_col_jac,
        )
        # return np.vstack([col_val, door_col_val, handle_col_val]), np.vstack([col_jac, door_col_jac, handle_col_jac])

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

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)

            grad = np.zeros((1, self.attr_dim + 7))
            grad[:, :5] = np.dot(sign * normal, robot_jac)
            grad[:, 6:8] = sign * normal

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
                robot.GetLink(linkRobot),
                robot.GetLink(linkObj),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, self.attr_dim + 7)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape(
            (len(greds), self.attr_dim + 7)
        )


class HSRObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.DIST_SAFE,
    ):
        self.attr_dim = 9
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[3], list(ATTRMAP[params[3]._type])),
                (params[4], list(ATTRMAP[params[4]._type])),
            ]
        )
        super(HSRObstructsHolding, self).__init__(
            name, params, expected_param_types, env, debug, tol
        )
        self.dsafe = const.DIST_SAFE

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(
                const.COLLISION_DOF_INDICES, DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

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

        body = robot_body.env_body
        manip = body.GetManipulator("arm")
        ee_trans = manip.GetTransform(), manip.GetTransform()
        arm_inds = self.robot.geom.dof_map["arm"]
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

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

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)

            grad = np.zeros((1, self.attr_dim + 6))
            grad[:, 6:8] = sign * normal[:2]
            arm_jac = np.array(
                [
                    np.cross(joint.GetAxis(), ptObj - joint.GetAnchor())
                    for joint in arm_joints
                ]
            ).T.copy()
            grad[:, 1:5] = np.dot(sign * normal, arm_jac)[1:]
            grad[:, 0] = sign * normal[2]

            obj_jac = -sign * normal
            obj_pos = OpenRAVEBody.obj_pose_from_transform(
                obj_body.env_body.GetTransform()
            )
            torque = ptObj - obj_pos[:3]
            Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
            rot_axises = [
                [0, 0, 1],
                np.dot(Rz, [0, 1, 0]),
                np.dot(Rz, np.dot(Ry, [1, 0, 0])),
            ]
            rot_vec = np.array(
                [[np.dot(np.cross(axis, torque), obj_jac) for axis in rot_axises]]
            )
            grad[:, self.attr_dim : self.attr_dim + 3] = obj_jac
            grad[:, self.attr_dim + 3 : self.attr_dim + 6] = rot_vec

            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot, linkObj)] = [
                self.dsafe - distance,
                grad,
                robot.GetLink(linkRobot),
                robot.GetLink(linkObj),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

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

        body = robot_body.env_body
        manip = body.GetManipulator("arm")
        ee_trans = manip.GetTransform(), manip.GetTransform()
        arm_inds = self.robot.geom.dof_map["arm"]

        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        diff = np.linalg.norm(obj_body.env_body.GetTransform()[:3, 3] - ee_trans[:3, 3])

        link_pair_to_col = {}
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
            if linkObj not in held_links or linkObstr not in obs_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            grad = np.zeros((1, self.attr_dim + 12))
            distance = c.GetDistance()
            normal = c.GetNormal()

            # Calculate robot joint jacobian

            grad[:, 6:8] = sign * normal[:2]
            arm_jac = np.array(
                [
                    np.cross(joint.GetAxis(), ptObj - joint.GetAnchor())
                    for joint in arm_joints
                ]
            ).T.copy()
            grad[:, 1:5] = np.dot(sign * normal, arm_jac)[1:]
            grad[:, 0] = sign * normal[2]

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
            if self._debug:
                self.plot_collision(ptObj, ptObstr, distance)

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


class HSRObstructsHoldingCan(HSRObstructsHolding):
    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9])
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

        manip = robot_body.env_body.GetManipulator("arm")
        pos = manip.GetTransform()[:3, 3]
        x[-6:-3] = pos.reshape(x[-6:-3].shape)


class HSRCollides(robot_predicates.Collides):

    # Collides Item Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = -const.COLLIDE_COEFF
        self.neg_coeff = const.COLLIDE_COEFF
        super(HSRCollides, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = const.COLLIDES_DSAFE


class HSRRCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 9
        self.dof_cache = None
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        super(HSRRCollides, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = const.RCOLLIDES_DSAFE

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose([x[6, 0], x[7, 0], x[8, 0]])
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(
                const.COLLISION_DOF_INDICES, DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

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
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkRobot1, linkRobot2 = None, None
            sign = 0
            if linkAParent == robot_body.name and linkBParent == robot_body.name:
                ptRobot1, ptRobot2 = c.GetPtA(), c.GetPtB()
                linkRobot1, linkRobot2 = linkA, linkB
                sign = -1
            else:
                continue

            if linkRobot1 not in col_links or linkRobot2 not in col_links:
                continue

            if (
                not (linkRobot1.startswith("right") or linkRobot1.startswith("left"))
                or linkRobot1 == linkRobot2
                or linkRobot1.endswith("upper_shoulder")
                or linkRobot1.endswith("lower_shoulder")
                or linkRobot2.startswith("right")
                or linkRobot2.startswith("left")
            ):
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot1).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot1)

            grad = np.zeros((1, self.attr_dim))
            grad[:, : self.attr_dim - 3] = np.dot(sign * normal, robot_jac)
            col_vec = sign * normal

            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot1, linkRobot2)] = [
                self.dsafe - distance,
                grad,
                robot.GetLink(linkRobot1),
                robot.GetLink(linkRobot2),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot1, ptRobot2, distance)

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

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)

            grad = np.zeros((1, self.attr_dim + 6))
            grad[:, : self.attr_dim - 3] = np.dot(sign * normal, robot_jac)
            col_vec = -sign * normal
            grad[:, self.attr_dim - 3 : self.attr_dim - 1] = -col_vec[:2]
            grad[:, self.attr_dim - 1] = 0  # Don't try to rotate away
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
                robot.GetLink(linkRobot),
                robot.GetLink(linkObj),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

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

    def resample(self, negated, t, plan):
        # Variable that needs to added to BoundExpr and latter pass to the planner
        JOINT_STEP = 20
        STEP_DECREASE_FACTOR = 1.5
        ATTEMPT_SIZE = 7
        LIN_SAMP_RANGE = 5

        attr_inds = OrderedDict()
        res = OrderedDict()
        robot, rave_body = self.robot, self._param_to_body[self.robot]
        body = rave_body.env_body
        manip = body.GetManipulator("arm")
        arm_inds = manip.GetArmIndices()
        lb_limit, ub_limit = body.GetDOFLimits()
        step_factor = JOINT_STEP
        joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds]) / step_factor
        base_step = np.array([0.05, 0.05, 0.0])
        if self.obstacle.pose[0, t] > self.robot.pose[0, t]:
            base_step[0] *= -1
        if self.obstacle.pose[1, t] > self.robot.pose[1, t]:
            base_step[1] *= -1
        original_arm_pose, arm_pose = robot.arm[:, t].copy(), robot.arm[:, t].copy()
        original_pose, pose = robot.pose[:, t].copy(), robot.pose[:, t].copy()
        rave_body.set_pose([pose[0], pose[1], pose[2]])
        rave_body.set_dof(
            {
                "arm": robot.arm[:, t].flatten(),
                "gripper": robot.gripper[:, t].flatten(),
            }
        )

        ## Determine the range we should resample
        pred_list = [
            act_pred["active_timesteps"]
            for act_pred in plan.actions[0].preds
            if act_pred["pred"].spacial_anchor == True
        ]
        start, end = 0, plan.horizon - 1
        for action in plan.actions:
            if action.active_timesteps[0] <= t and action.active_timesteps[1] > t:
                start, end = action.active_timesteps
                for act_pred in plan.actions[0].preds:
                    if act_pred["pred"].spacial_anchor == True:
                        if (
                            act_pred["active_timesteps"][0]
                            + act_pred["pred"].active_range[0]
                            > t
                        ):
                            end = min(
                                end,
                                act_pred["active_timesteps"][0]
                                + act_pred["pred"].active_range[0],
                            )
                        if (
                            act_pred["active_timesteps"][1]
                            + act_pred["pred"].active_range[1]
                            < t
                        ):
                            start = max(
                                start,
                                act_pred["active_timesteps"][1]
                                + act_pred["pred"].active_range[1],
                            )

        desired_end_pose = robot.arm[:, end]
        current_end_pose = robot.arm[:, t]
        col_report = CollisionReport()
        collisionChecker = RaveCreateCollisionChecker(plan.env, "pqp")
        count = 1
        while (
            body.CheckSelfCollision()
            or collisionChecker.CheckCollision(body, report=col_report)
            or col_report.minDistance <= pred.dsafe
        ):
            step_sign = np.ones(len(arm_inds))
            step_sign[
                np.random.choice(len(arm_inds), len(arm_inds) / 2, replace=False)
            ] = -1
            # Ask in collision pose to randomly move a step, hopefully out of collision
            arm_pose = original_arm_pose + np.multiply(step_sign, joint_step)
            pose = original_pose + base_step
            rave_body.set_dof({"arm": arm_pose})
            rave_body.set_pose([pose[0], pose[1], 0])
            # arm_pose = body.GetActiveDOFValues()[arm_inds]
            if not count % ATTEMPT_SIZE:
                step_factor = step_factor / STEP_DECREASE_FACTOR
                joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds]) / step_factor
            count += 1

            if count > 25:
                return None, None

        add_to_attr_inds_and_res(
            t, attr_inds, res, robot, [("arm", arm_pose), ("pose", pose)]
        )
        robot._free_attrs["arm"][:, t] = 0

        start, end = max(start, t - LIN_SAMP_RANGE), min(t + LIN_SAMP_RANGE, end)
        rcollides_traj = np.hstack(
            [
                lin_interp_traj(robot.arm[:, start], arm_pose, t - start),
                lin_interp_traj(arm_pose, robot.arm[:, end], end - t)[:, 1:],
            ]
        ).T
        base_rcollides_traj = np.hstack(
            [
                lin_interp_traj(robot.pose[:, start], pose, t - start),
                lin_interp_traj(pose, robot.pose[:, end], end - t)[:, 1:],
            ]
        ).T
        i = start + 1
        for traj in rcollides_traj[1:-1]:
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [("arm", traj)])
            i += 1
        i = start + 1
        for traj in base_rcollides_traj[1:-1]:
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [("pose", traj)])
            i += 1

        return res, attr_inds


class HSRRSelfCollides(robot_predicates.RSelfCollides):

    # RCollides Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 9
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        super(HSRRSelfCollides, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = const.RCOLLIDES_DSAFE

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose([x[6, 0], x[7, 0], x[8, 0]])
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(
                const.COLLISION_DOF_INDICES, DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

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
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkRobot1, linkRobot2 = None, None
            sign = 0
            if linkAParent == robot_body.name and linkBParent == robot_body.name:
                ptRobot1, ptRobot2 = c.GetPtA(), c.GetPtB()
                linkRobot1, linkRobot2 = linkA, linkB
                sign = -1
            else:
                continue

            if linkRobot1 not in col_links or linkRobot2 not in col_links:
                continue

            if (
                not (linkRobot1.startswith("right") or linkRobot1.startswith("left"))
                or linkRobot1 == linkRobot2
                or linkRobot1.endswith("upper_shoulder")
                or linkRobot1.endswith("lower_shoulder")
                or linkRobot2.startswith("right")
                or linkRobot2.startswith("left")
            ):
                continue

            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot1).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot1)

            grad = np.zeros((1, self.attr_dim))
            grad[:, : self.attr_dim - 3] = np.dot(sign * normal, robot_jac)
            col_vec = sign * normal

            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot1, linkRobot2)] = [
                self.dsafe - distance,
                grad,
                robot.GetLink(linkRobot1),
                robot.GetLink(linkRobot2),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot1, ptRobot2, distance)

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


class HSRCollidesWasher(HSRRCollides):
    """
    This collision checks the full mock-up as a set of its individual parts
    """

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 9
        self.dof_cache = None
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type][:-1])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        super(HSRRCollides, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = 1e-2  # const.RCOLLIDES_DSAFE

    def robot_obj_collision(self, x):
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
        washer_pos, washer_rot = x[-7:-4], x[-4:-1]
        obj_body.set_pose(washer_pos, washer_rot)
        obj_body.set_dof({"door": x[-1]})

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
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        return col_val, col_jac

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

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)

            grad = np.zeros((1, self.attr_dim + 7))
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
                robot.GetLink(linkRobot),
                robot.GetLink(linkObj),
            ]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get(
                (robot_link, obj_link),
                [
                    self.dsafe - const.MAX_CONTACT_DISTANCE,
                    np.zeros((1, self.attr_dim + 7)),
                    None,
                    None,
                ],
            )
            vals.append(col_infos[0])
            if len(col_infos[1][0]) == 6:
                col_infos[1] = np.c_[col_infos[1], [[0]]]
                import ipdb

                ipdb.set_trace()
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape(
            (len(greds), self.attr_dim + 7)
        )

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(
                const.COLLISION_DOF_INDICES, DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        base_pose = x[16]
        robot_body.set_pose([x[6, 0], x[7, 0], x[8, 0]])
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})


"""
EEReachable Family
"""


class HSREEReachable(robot_predicates.EEReachable):
    def __init__(
        self,
        name,
        params,
        expected_param_types,
        active_range=(-const.EEREACHABLE_STEPS, const.EEREACHABLE_STEPS),
        env=None,
        debug=False,
    ):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[2], list(ATTRMAP[params[2]._type])),
            ]
        )
        self.attr_dim = 15
        self.coeff = const.EEREACHABLE_COEFF
        self.rot_coeff = const.EEREACHABLE_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 3 + 3 * (1 + (active_range[1] - active_range[0]))
        self.arm = "center"
        super(HSREEReachable, self).__init__(
            name, params, expected_param_types, active_range, env, debug
        )

    def get_arm_jac(self, arm_jac, base_jac, obj_jac, arm):
        dim = arm_jac.shape[0]
        jacobian = np.hstack((arm_jac, np.zeros((dim, 1)), base_jac, obj_jac))
        return jacobian

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step * np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step * np.array([0, 0, const.RETREAT_DIST])

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    # @profile
    def get_robot_info(self, robot_body, arm):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = self.robot.geom.dof_map["arm"]
        return robot_trans, arm_inds

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
        for s in range(start, end + 1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(
                self.coeff * self.rel_ee_pos_check_f(x[i : i + self.attr_dim], rel_pt)
            )
            i += self.attr_dim
            if s == 0:
                f_res.append(np.zeros((3, 1)))

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
            i += self.attr_dim
            if s == 0:
                grad[j : j + 3, i : i + self.attr_dim] = 0
                j += dim

        return grad

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
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        arm_jac[0] = obj_pos[2] - robot_pos[2]
        # Calculate jacobian for the robot base
        base_pos_jac = np.r_[np.diag(obj_pos[:2] - robot_pos[:2]), np.zeros((1, 2))]
        base_rot_jac = np.cross(np.array([0, 0, 1]), robot_pos).reshape((3, 1))
        base_jac = np.c_[base_pos_jac, base_rot_jac]
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


class HSREEReachableVer(HSREEReachable):

    # HSREEReachableVerLeftPos Robot, RobotPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step * np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step * np.array([-const.RETREAT_DIST, 0, 0])


class HSREEReachableHor(HSREEReachable):

    # HSREEReachableVerLeftPos Robot, RobotPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step * np.array([0, 0, -const.APPROACH_DIST])
        else:
            return rel_step * np.array([0, 0, const.RETREAT_DIST])


"""
    InGripper Constraint Family
"""


class HSRInGripper(robot_predicates.InGripper):

    # InGripper, Robot, Object

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type][:1])),
            ]
        )
        self.coeff = const.IN_GRIPPER_COEFF
        self.rot_coeff = const.IN_GRIPPER_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.arm = "center"
        self.eval_dim = 3
        self.rel_pt = np.array([0, 0, -const.HAND_DIST])
        super(HSRInGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def get_robot_info(self, robot_body, arm):
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(0, 5))
        return robot_trans, arm_inds

    def get_arm_jac(self, arm_jac, base_jac, obj_jac, arm):
        if not arm == "right" and not arm == "left":
            assert PredicateException("Invalid Arm Specified")

        dim = arm_jac.shape[0]
        jacobian = np.hstack((arm_jac, np.zeros((dim, 3)), base_jac, obj_jac[:, :3]))
        return jacobian

    def stacked_f(self, x):
        res = np.vstack([self.coeff * self.pos_check_f(x, rel_pt=self.rel_pt)])
        return res

    def stacked_grad(self, x):
        jac = self.pos_check_jac(x, rel_pt=self.rel_pt)
        return np.vstack([self.coeff * jac])

    def resample(self, negated, t, plan):
        JOINT_STEP = 20
        STEP_DECREASE_FACTOR = 1.5
        ATTEMPT_SIZE = 7
        LIN_SAMP_RANGE = 5

        attr_inds = OrderedDict()
        res = OrderedDict()
        obj = self.obj
        robot = self.robot
        obj.openrave_body.set_pose(obj.pose[:, t], obj.rotation[:, t])
        obj_trans = obj.openrave_body.env_body.GetTransform()
        self.set_robot_poses(
            np.r_[
                self.robot.arm[:, t], self.robot.gripper[:, t], self.robot.pose[:, t]
            ].reshape((9, 1)),
            self.robot.openrave_body,
        )
        robot_trans, arm_inds = self.get_robot_info(self.robot.openrave_body, "center")
        disp = self.rel_pos_error_f(obj_trans, robot_trans, self.rel_pt).flatten()
        robot_pos = self.robot.pose[:, t]
        pos = robot_pos.copy()
        pos[:2] -= disp[:2]
        arm_pose = self.robot.arm[:, t].copy()
        arm_pose[0] = np.maximum(arm_pose[0] - disp[2], 0)

        start, end = 0, plan.horizon - 1
        for action in plan.actions:
            if action.active_timesteps[0] <= t and action.active_timesteps[1] > t:
                start, end = action.active_timesteps
                for act_pred in plan.actions[0].preds:
                    if act_pred["pred"].spacial_anchor == True:
                        if (
                            act_pred["active_timesteps"][0]
                            + act_pred["pred"].active_range[0]
                            > t
                        ):
                            end = min(
                                end,
                                act_pred["active_timesteps"][0]
                                + act_pred["pred"].active_range[0],
                            )
                        if (
                            act_pred["active_timesteps"][1]
                            + act_pred["pred"].active_range[1]
                            < t
                        ):
                            start = max(
                                start,
                                act_pred["active_timesteps"][1]
                                + act_pred["pred"].active_range[1],
                            )

        obj_pos = self.obj.pose[:, t]
        for act in plan.actions:
            if act.active_timesteps[1] == t:
                obj_pos = self.obj.pose[:, t] + disp
        add_to_attr_inds_and_res(
            t, attr_inds, res, self.robot, [("pose", pos), ("arm", arm_pose)]
        )
        add_to_attr_inds_and_res(t, attr_inds, res, self.obj, [("pose", obj_pos)])

        start, end = max(start, t - LIN_SAMP_RANGE), min(t + LIN_SAMP_RANGE, end)
        arm_traj = np.hstack(
            [
                lin_interp_traj(robot.arm[:, start], arm_pose, t - start),
                lin_interp_traj(arm_pose, robot.arm[:, end], end - t)[:, 1:],
            ]
        ).T
        base_traj = np.hstack(
            [
                lin_interp_traj(robot.pose[:, start], pos, t - start),
                lin_interp_traj(pos, robot.pose[:, end], end - t)[:, 1:],
            ]
        ).T
        i = start + 1
        for traj in arm_traj[1:-1]:
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [("arm", traj)])
            i += 1
        i = start + 1
        for traj in base_traj[1:-1]:
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [("pose", traj)])
            i += 1

        return res, attr_inds

    def robot_obj_kinematics(self, x):
        """
        This function is used to check whether End Effective pose's position is at robot gripper's center

        Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        # Setting the poses for forward kinematics to work
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        ee_pos, ee_rot = x[-3:], np.zeros((3,))
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]  # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints


class HSRCanInGripper(HSRInGripper):
    pass


class HSRGripperAt(robot_predicates.GripperAt):

    # InGripper, Robot, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        self.coeff = const.GRIPPER_AT_COEFF
        self.rot_coeff = const.GRIPPER_AT_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.arm = "center"
        super(HSRGripperAt, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def get_robot_info(self, robot_body, arm):
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(0, 5))
        return robot_trans, arm_inds

    def robot_obj_kinematics(self, x):
        """
        This function is used to check whether End Effective pose's position is at robot gripper's center

        Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        # Setting the poses for forward kinematics to work
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        ee_pos, ee_rot = x[-6:-3], x[-3:]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]  # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints

    def stacked_f(self, x):
        return np.vstack(
            [
                self.coeff
                * self.pos_check_f(x, rel_pt=np.array([0, 0, -const.HAND_DIST]))
            ]
        )

    def stacked_grad(self, x):
        return np.vstack(
            [
                10
                * self.coeff
                * self.pos_check_jac(x, rel_pt=np.array([0, 0, -const.HAND_DIST]))
            ]
        )


class HSRAlmostInGripper(robot_predicates.AlmostInGripper):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        self.arm = "center"
        self.coeff = const.IN_GRIPPER_COEFF
        self.eval_dim = 3
        self.max_dist = np.array([0.02, 0.01, 0.02, 0.02, 0.01, 0.02])
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.rel_pt = np.array([0, 0, -const.HAND_DIST])
        super(HSRAlmostInGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_cloth_in_gripper(self, negated, t, plan)

    def stacked_f(self, x):
        pos_check = self.pos_check_f(x, rel_pt=np.array([0, 0, -const.HAND_DIST]))
        return self.coeff * np.r_[pos_check, -pos_check]

    def stacked_grad(self, x):
        pos_jac = self.pos_check_jac(x, rel_pt=np.array([0, 0, -const.HAND_DIST]))
        return self.coeff * np.r_[pos_jac, pos_jac]

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def get_robot_info(self, robot_body, arm):
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(0, 5))
        return robot_trans, arm_inds

    def get_arm_jac(self, arm_jac, base_jac, obj_jac, arm):
        if not arm == "right" and not arm == "left":
            assert PredicateException("Invalid Arm Specified")

        dim = arm_jac.shape[0]
        jacobian = np.hstack((arm_jac, np.zeros((dim, 3)), base_jac, obj_jac))
        return jacobian

    def resample(self, negated, t, plan):
        attr_inds = OrderedDict()
        res = OrderedDict()
        obj = self.obj
        obj.openrave_body.set_pose(obj.pose[:, t], obj.rotation[:, t])
        obj_trans = obj.openrave_body.env_body.GetTransform()
        self.set_robot_poses(
            np.r_[
                self.robot.arm[:, t], self.robot.gripper[:, t], self.robot.pose[:, t]
            ].reshape((9, 1)),
            self.robot.openrave_body,
        )
        robot_trans, arm_inds = self.get_robot_info(self.robot.openrave_body, "center")
        disp = self.rel_pos_error_f(obj_trans, robot_trans, self.rel_pt).flatten()
        pos = obj.pose[:, t] + disp
        add_to_attr_inds_and_res(t, attr_inds, res, self.obj, [("pose", pos)])

        return res, attr_inds

    def robot_obj_kinematics(self, x):
        """
        This function is used to check whether End Effective pose's position is at robot gripper's center

        Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        # Setting the poses for forward kinematics to work
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        ee_pos, ee_rot = x[-6:-3], x[-3:]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [
            [0, 0, 1],
            np.dot(Rz, [0, 1, 0]),
            np.dot(Rz, np.dot(Ry, [1, 0, 0])),
        ]  # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints


class HSRCanAlmostInGripper(HSRAlmostInGripper):
    pass


"""
Other Constraints
"""


class HSRBasketLevel(robot_predicates.BasketLevel):
    # HSRBasketLevel BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        self.attr_dim = 3
        self.basket = params[0]
        super(HSRBasketLevel, self).__init__(
            name, params, expected_param_types, env, debug
        )


class HSRClothTargetInWasher(ExprPredicate):
    # HSRClothTargetInWasher ClothTarget WasherTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][0]]),
                (params[1], [ATTRMAP[params[1]._type][0]]),
            ]
        )
        self.attr_dim = 6
        self.cloth_target = params[0]
        self.washer_pose = params[1]
        A = np.c_[np.eye(self.attr_dim / 2), -np.eye(self.attr_dim / 2)]
        b = np.zeros((self.attr_dim / 2, 1))
        val = np.array(
            [
                [const.WASHER_DEPTH_OFFSET / 2],
                [np.sqrt(3) * const.WASHER_DEPTH_OFFSET / 2],
                [0],
            ]
        )
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(HSRClothTargetInWasher, self).__init__(
            name, e, self.attr_inds, params, expected_param_types, priority=-2
        )


class HSRClothTargetInBasket(ExprPredicate):
    # HSRClothTargetInBasket ClothTarget BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][0]]),
                (params[1], [ATTRMAP[params[1]._type][0]]),
            ]
        )
        self.attr_dim = 6
        self.cloth_target = params[0]
        self.basket_target = params[1]

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b = np.zeros((6, 1))

        val = np.array([[0.09], [0.09], [-0.04], [0.09], [0.09], [0.04]])
        pos_expr = AffExpr(A, b)
        e = LEqExpr(pos_expr, val)
        super(HSRClothTargetInBasket, self).__init__(
            name, e, self.attr_inds, params, expected_param_types, priority=-2
        )


class HSRObjectWithinRotLimit(robot_predicates.ObjectWithinRotLimit):
    # HSRObjectWithinRotLimit Object
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        self.attr_dim = 3
        self.object = params[0]
        super(HSRObjectWithinRotLimit, self).__init__(
            name, params, expected_param_types, env, debug
        )


class HSRGripperLevel(robot_predicates.GrippersLevel):
    # HSRLeftGripperDownRot Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 0.1
        self.opt_coeff = 0.1
        self.eval_f = lambda x: self.arm_rot_check(x)
        self.eval_grad = lambda x: self.arm_rot_jac(x)
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])[:-1])])
        self.eval_dim = 1
        self.dir = [0, 0, -1]
        super(HSRGripperLevel, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        robot_body.set_pose(x[6:9].flatten())
        robot_body.set_dof({"arm": x[:5, 0], "gripper": x[5, 0]})

    def get_robot_info(self, robot_body, arm):
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(0, 5))
        return robot_trans, arm_inds

    def arm_rot_check(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_trans, arm_inds = self.get_robot_info(robot_body, "center")
        local_dir = [1, 0, 0]
        world_dir = robot_trans[:3, :3].dot(local_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot(self.dir, world_dir)) - 1]])

        return rot_val

    def arm_rot_jac(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_trans, arm_inds = self.get_robot_info(robot_body, "center")
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        tool_link = robot_body.env_body.GetLink("hand")
        manip_trans = tool_link.GetTransform()

        local_dir = [1, 0, 0]
        world_dir = robot_trans[:3, :3].dot(local_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        obj_dir = self.dir
        sign = np.sign(np.dot(obj_dir, world_dir))

        arm_jac = np.array(
            [
                np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir))
                for joint in arm_joints
            ]
        ).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))

        jac = np.zeros((1, 11))
        jac[0, :5] = arm_jac
        return jac
