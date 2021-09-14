from core.util_classes import robot_predicates
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from core.util_classes.pr2_sampling import (
    ee_reachable_resample,
    resample_bp_around_target,
)
import core.util_classes.pr2_constants as const
from collections import OrderedDict
from openravepy import DOFAffine
import numpy as np

"""
This file Defines specific PR2 related predicates
"""

# Attributes used in pr2 domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {
    "Robot": (
        ("backHeight", np.array([0], dtype=np.int)),
        ("lArmPose", np.array(list(range(7)), dtype=np.int)),
        ("lGripper", np.array([0], dtype=np.int)),
        ("rArmPose", np.array(list(range(7)), dtype=np.int)),
        ("rGripper", np.array([0], dtype=np.int)),
        ("pose", np.array([0, 1, 2], dtype=np.int)),
    ),
    "RobotPose": (
        ("backHeight", np.array([0], dtype=np.int)),
        ("lArmPose", np.array(list(range(7)), dtype=np.int)),
        ("lGripper", np.array([0], dtype=np.int)),
        ("rArmPose", np.array(list(range(7)), dtype=np.int)),
        ("rGripper", np.array([0], dtype=np.int)),
        ("value", np.array([0, 1, 2], dtype=np.int)),
    ),
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
}


class PR2At(robot_predicates.At):
    pass


class PR2RobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 20
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        super(PR2RobotAt, self).__init__(name, params, expected_param_types, env)


class PR2IsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        super(PR2IsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((const.JOINT_DIM, 1))
        active_lb = lb_limit[dof_inds].reshape((const.JOINT_DIM, 1))
        joint_move = (active_ub - active_lb) / const.JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= const.BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/const.JOINT_MOVE_FACTOR
        val = np.vstack(
            (
                joint_move,
                const.BASE_MOVE * np.ones((const.BASE_DIM, 1)),
                joint_move,
                const.BASE_MOVE * np.ones((const.BASE_DIM, 1)),
            )
        )
        A = (
            np.eye(2 * const.ROBOT_ATTR_DIM)
            - np.eye(2 * const.ROBOT_ATTR_DIM, k=const.ROBOT_ATTR_DIM)
            - np.eye(2 * const.ROBOT_ATTR_DIM, k=-const.ROBOT_ATTR_DIM)
        )
        b = np.zeros((2 * const.ROBOT_ATTR_DIM, 1))
        robot_body._set_active_dof_inds(list(range(39)))

        # Setting attributes for testing
        self.base_step = const.BASE_MOVE * np.ones((const.BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val


class PR2WithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        super(PR2WithinJointLimit, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
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
        robot_body._set_active_dof_inds(list(range(39)))

        joint_move = (active_ub - active_lb) / const.JOINT_MOVE_FACTOR
        self.base_step = const.BASE_MOVE * np.ones((3, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val


class PR2Stationary(robot_predicates.Stationary):
    pass


class PR2StationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-1]])])
        self.attr_dim = const.BASE_DIM
        super(PR2StationaryBase, self).__init__(name, params, expected_param_types, env)


class PR2StationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict(
            [(params[0], list(ATTRMAP[params[0]._type][1:-1]))]
        )
        self.attr_dim = const.TWOARMDIM
        super(PR2StationaryArms, self).__init__(name, params, expected_param_types, env)


class PR2StationaryW(robot_predicates.StationaryW):
    pass


class PR2StationaryNEq(robot_predicates.StationaryNEq):
    pass


class PR2GraspValid(robot_predicates.GraspValid):
    pass


class PR2GraspValidPos(PR2GraspValid):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][0]]),
                (params[1], [ATTRMAP[params[1]._type][0]]),
            ]
        )
        self.attr_dim = 3
        super(PR2GraspValidPos, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2GraspValidRot(PR2GraspValid):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict(
            [
                (params[0], [ATTRMAP[params[0]._type][1]]),
                (params[1], [ATTRMAP[params[1]._type][1]]),
            ]
        )
        self.attr_dim = 3
        super(PR2GraspValidRot, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2InContactRight(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][4]])])
        super(PR2InContactRight, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2InContactLeft(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][2]])])
        super(PR2InContactLeft, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2InGripper(robot_predicates.InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.eval_dim = 3
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        super(PR2InGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {
            "backHeight": back_height,
            "lArmPose": l_arm_pose,
            "lGripper": l_gripper,
            "rArmPose": r_arm_pose,
            "rGripper": r_gripper,
        }
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator("rightarm").GetArmIndices()
        return robot_trans, arm_inds

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the displacement between center of gripper and center of object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        gp = np.array([0, 0, 0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - gp - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack(
            (base_jac, torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), obj_jac)
        )
        return dist_val, dist_jac

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the rotational error between
        robot gripper's rotational axis and object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        local_dir = np.array([0.0, 0.0, 1.0])
        obj_dir = np.dot(obj_trans[:3, :3], local_dir)
        world_dir = robot_trans[:3, :3].dot(local_dir)
        obj_dir = obj_dir / np.linalg.norm(obj_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]])
        # computing robot's jacobian
        arm_jac = np.array(
            [
                np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir))
                for joint in arm_joints
            ]
        ).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = np.array(np.dot(obj_dir, np.cross([0, 0, 1], world_dir)))
        base_jac = np.array([[0, 0, base_jac]])
        # computing object's jacobian
        obj_jac = np.array(
            [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
        )
        obj_jac = np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
        # Create final 1x26 jacobian matrix
        rot_jac = np.hstack(
            (base_jac, np.zeros((1, 9)), arm_jac, np.zeros((1, 1)), obj_jac)
        )

        return (rot_val, rot_jac)


class PR2InGripperRight(PR2InGripper):
    pass


class PR2InGripperLeft(PR2InGripper):
    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("l_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator("leftarm").GetArmIndices()
        return robot_trans, arm_inds

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the displacement between center of gripper and center of object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        gp = np.array([0, 0, 0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[gp, 1])[:3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - gp - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((base_jac, torso_jac, arm_jac, np.zeros((3, 9)), obj_jac))
        return dist_val, dist_jac

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the rotational error between
        robot gripper's rotational axis and object's rotational axis

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        local_dir = np.array([0.0, 0.0, 1.0])
        obj_dir = np.dot(obj_trans[:3, :3], local_dir)
        world_dir = robot_trans[:3, :3].dot(local_dir)
        obj_dir = obj_dir / np.linalg.norm(obj_dir)
        world_dir = world_dir / np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]])
        # computing robot's jacobian
        arm_jac = np.array(
            [
                np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir))
                for joint in arm_joints
            ]
        ).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = np.array(np.dot(obj_dir, np.cross([0, 0, 1], world_dir)))
        base_jac = np.array([[0, 0, base_jac]])
        # computing object's jacobian
        obj_jac = np.array(
            [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
        )
        obj_jac = np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
        # Create final 1x26 jacobian matrix
        rot_jac = np.hstack(
            (base_jac, np.zeros((1, 1)), arm_jac, np.zeros((1, 9)), obj_jac)
        )

        return (rot_val, rot_jac)


class PR2InGripperPosRight(PR2InGripperRight):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check(x)[0]
        self.eval_grad = lambda x: self.pos_check(x)[1]
        super(PR2InGripperPosRight, self).__init__(
            name, params, expected_param_types, env, debug
        )

    # "Robot": (("backHeight", np.array([0], dtype=np.int)),
    #                          ("lArmPose", np.array(range(7), dtype=np.int)),
    #                          ("lGripper", np.array([0], dtype=np.int)),
    #                          ("rArmPose", np.array(range(7), dtype=np.int)),
    #                          ("rGripper", np.array([0], dtype=np.int)),
    #                          ("pose", np.array([0,1,2], dtype=np.int)))
    #
    #            "Can": (("pose", np.array([0,1,2], dtype=np.int)),
    #                     ("rotation", np.array([0,1,2], dtype=np.int)))

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the displacement between center of gripper and center of object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        gp = np.array([0, 0, 0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[17:20])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - gp - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack(
            (torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), base_jac, obj_jac)
        )

        return (dist_val, dist_jac)


class PR2InGripperRotRight(PR2InGripperRight):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check(x)[0]
        self.eval_grad = lambda x: self.rot_check(x)[1]
        super(PR2InGripperRotRight, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2InGripperPosLeft(PR2InGripperLeft):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check(x)[0]
        self.eval_grad = lambda x: self.pos_check(x)[1]
        super(PR2InGripperPosLeft, self).__init__(
            name, params, expected_param_types, env, debug
        )

    # "Robot": (("backHeight", np.array([0], dtype=np.int)),
    #                          ("lArmPose", np.array(range(7), dtype=np.int)),
    #                          ("lGripper", np.array([0], dtype=np.int)),
    #                          ("rArmPose", np.array(range(7), dtype=np.int)),
    #                          ("rGripper", np.array([0], dtype=np.int)),
    #                          ("pose", np.array([0,1,2], dtype=np.int)))
    #
    #            "Can": (("pose", np.array([0,1,2], dtype=np.int)),
    #                     ("rotation", np.array([0,1,2], dtype=np.int)))

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
        This function calculates the value and the jacobian of the displacement between center of gripper and center of object

        obj_trans: object's rave_body transformation
        robot_trans: robot gripper's rave_body transformation
        axises: rotational axises of the object
        arm_joints: list of robot joints
        """
        gp = np.array([0, 0, 0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[17:20])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - gp - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((torso_jac, arm_jac, np.zeros((3, 9)), base_jac, obj_jac))

        return (dist_val, dist_jac)


class PR2InGripperRotLeft(PR2InGripperLeft):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check(x)[0]
        self.eval_grad = lambda x: self.rot_check(x)[1]
        super(PR2InGripperRotLeft, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2BothEndsInGripper(PR2InGripper):
    # PR2BothEndsInGripper Robot, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.eval_dim = 6
        super(PR2BothEndsInGripper, self).__init__(
            name, params, expected_param_types, env, debug
        )

    # @profile
    def both_arm_pos_check_f(self, x):
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        self.arm = "left"
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, "left")
        l_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in l_arm_inds]
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, "right")
        r_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in r_arm_inds]
        rel_pt = np.array([0, 0, self.obj.geom.height / 2.0 + 0.1])
        l_pos_val = self.rel_pos_error_f(obj_trans, l_ee_trans, rel_pt)
        rel_pt = np.array([0, 0, -self.obj.geom.height / 2.0 - 0.1])
        r_pos_val = self.rel_pos_error_f(obj_trans, r_ee_trans, rel_pt)
        return np.vstack([l_pos_val, r_pos_val])

    # @profile
    def both_arm_pos_check_jac(self, x):
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, "left")
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, "right")

        self.arm = "right"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6:-3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([0, 0, -self.obj.geom.height / 2.0 - 0.1])
        r_obj_pos_jac = self.rel_pos_error_jac(
            obj_trans, r_ee_trans, axises, arm_joints, rel_pt
        )

        self.arm = "left"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6:-3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([0, 0, self.obj.geom.height / 2.0 + 0.1])
        l_obj_pos_jac = self.rel_pos_error_jac(
            obj_trans, l_ee_trans, axises, arm_joints, rel_pt
        )

        return np.vstack([l_obj_pos_jac, r_obj_pos_jac])

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_jac(x)])


class PR2CloseGrippers(robot_predicates.InContacts):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict(
            [(params[0], [ATTRMAP[params[0]._type][1], ATTRMAP[params[0]._type][3]])]
        )
        super(PR2CloseGrippers, self).__init__(
            name, params, expected_param_types, env, debug
        )


class PR2EEReachable(robot_predicates.EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        steps=const.EEREACHABLE_STEPS,
    ):
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[2], list(ATTRMAP[params[2]._type])),
            ]
        )
        self.attr_dim = 26
        super(PR2EEReachable, self).__init__(
            name, params, expected_param_types, env, debug, steps
        )

    def resample(self, negated, t, plan):
        return ee_reachable_resample(self, negated, t, plan)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {
            "backHeight": back_height,
            "lArmPose": l_arm_pose,
            "lGripper": l_gripper,
            "rArmPose": r_arm_pose,
            "rGripper": r_gripper,
        }
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator("rightarm").GetArmIndices()
        return robot_trans, arm_inds

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step * np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step * np.array([0, 0, const.RETREAT_DIST])

    def stacked_f(self, x):
        i = 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end + 1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(
                self.ee_pose_check_rel_obj(x[i : i + self.attr_dim], rel_pt)[0]
            )
            i += self.attr_dim
        return np.vstack(tuple(f_res))

    def stacked_grad(self, x):
        f_grad = []
        start, end = self.active_range
        t = 2 * self._steps + 1
        k = 3

        grad = np.zeros((k * t, self.attr_dim * t))
        i = 0
        j = 0
        for s in range(start, end + 1):
            rel_pt = self.get_rel_pt(s)
            grad[j : j + k, i : i + self.attr_dim] = self.ee_pose_check_rel_obj(
                x[i : i + self.attr_dim], rel_pt
            )[1]
            i += self.attr_dim
            j += k
        return grad

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
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        # obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack(
            (base_jac, torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), obj_jac)
        )

        return (dist_val, dist_jac)

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
            obj_dir = np.dot(obj_trans[:3, :3], local_dir)
            world_dir = robot_trans[:3, :3].dot(local_dir)
            rot_vals.append(np.array([[np.dot(obj_dir, world_dir) - 1]]))
            # computing robot's jacobian
            arm_jac = np.array(
                [
                    np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir))
                    for joint in arm_joints
                ]
            ).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0, 0, 1], world_dir)))
            base_jac = np.array([[0, 0, base_jac]])
            # computing object's jacobian
            obj_jac = np.array(
                [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
            )
            obj_jac = np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix
            rot_jacs.append(
                np.hstack(
                    (base_jac, np.zeros((1, 9)), arm_jac, np.zeros((1, 1)), obj_jac)
                )
            )

        rot_val = np.vstack(rot_vals)
        rot_jac = np.vstack(rot_jacs)

        return (rot_val, rot_jac)


class PR2EEReachableRight(PR2EEReachable):
    pass


class PR2EEReachableLeft(PR2EEReachable):
    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("l_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator("leftarm").GetArmIndices()
        return robot_trans, arm_inds

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
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3, 1))
        # Calculate the joint jacobian
        arm_jac = np.array(
            [
                np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor())
                for joint in arm_joints
            ]
        ).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:, 2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[:3])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0], [0], [1]])
        # Calculate object jacobian
        # obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = (
            -1
            * np.array(
                [
                    np.cross(axis, obj_pos - obj_trans[:3, 3].flatten())
                    for axis in axises
                ]
            ).T
        )
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((base_jac, torso_jac, arm_jac, np.zeros((3, 9)), obj_jac))

        return (dist_val, dist_jac)

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
            obj_dir = np.dot(obj_trans[:3, :3], local_dir)
            world_dir = robot_trans[:3, :3].dot(local_dir)
            rot_vals.append(np.array([[np.dot(obj_dir, world_dir) - 1]]))
            # computing robot's jacobian
            arm_jac = np.array(
                [
                    np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir))
                    for joint in arm_joints
                ]
            ).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0, 0, 1], world_dir)))
            base_jac = np.array([[0, 0, base_jac]])
            # computing object's jacobian
            obj_jac = np.array(
                [np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises]
            )
            obj_jac = np.r_[[0, 0, 0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix
            rot_jacs.append(
                np.hstack(
                    (base_jac, np.zeros((1, 1)), arm_jac, np.zeros((1, 9)), obj_jac)
                )
            )

        rot_val = np.vstack(rot_vals)
        rot_jac = np.vstack(rot_jacs)

        return (rot_val, rot_jac)


class PR2EEReachablePosRight(PR2EEReachableRight):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        steps=const.EEREACHABLE_STEPS,
    ):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_OPT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.attr_dim = 26
        super(PR2EEReachablePosRight, self).__init__(
            name, params, expected_param_types, env, debug, steps
        )


class PR2EEReachableRotRight(PR2EEReachableRight):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, steps=0
    ):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(PR2EEReachableRotRight, self).__init__(
            name, params, expected_param_types, env, debug, steps
        )


class PR2EEReachablePosLeft(PR2EEReachableLeft):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        steps=const.EEREACHABLE_STEPS,
    ):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_OPT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.attr_dim = 26
        super(PR2EEReachablePosLeft, self).__init__(
            name, params, expected_param_types, env, debug, steps
        )


class PR2EEReachableRotLeft(PR2EEReachableLeft):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(
        self, name, params, expected_param_types, env=None, debug=False, steps=0
    ):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(PR2EEReachableRotLeft, self).__init__(
            name, params, expected_param_types, env, debug, steps
        )


class PR2Obstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(
        self,
        name,
        params,
        expected_param_types,
        env=None,
        debug=False,
        tol=const.COLLISION_TOL,
    ):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[3], list(ATTRMAP[params[3]._type])),
            ]
        )
        super(PR2Obstructs, self).__init__(
            name, params, expected_param_types, env, debug, tol
        )

    def resample(self, negated, t, plan):
        target_pose = self.can.pose[:, t]
        return resample_bp_around_target(
            self, t, plan, target_pose, dist=const.OBJ_RING_SAMPLING_RADIUS
        )

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {
            "backHeight": back_height,
            "lArmPose": l_arm_pose,
            "lGripper": l_gripper,
            "rArmPose": r_arm_pose,
            "rGripper": r_gripper,
        }
        robot_body.set_dof(dof_value_map)

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()
            ]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()
            ]
            robot.SetActiveDOFs(
                dof_inds, DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")


class PR2ObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[3], list(ATTRMAP[params[3]._type])),
                (params[4], list(ATTRMAP[params[4]._type])),
            ]
        )
        self.OBSTRUCTS_OPT_COEFF = const.OBSTRUCTS_OPT_COEFF
        super(PR2ObstructsHolding, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        target_pose = self.obstruct.pose[:, t]
        return resample_bp_around_target(
            self, t, plan, target_pose, dist=const.OBJ_RING_SAMPLING_RADIUS
        )

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()
            ]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()
            ]
            # dof_inds = [12]+ list(range(15, 22)) + [22]+ list(range(27, 34)) + [34]
            robot.SetActiveDOFs(
                dof_inds, DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {
            "backHeight": back_height,
            "lArmPose": l_arm_pose,
            "lGripper": l_gripper,
            "rArmPose": r_arm_pose,
            "rGripper": r_gripper,
        }
        robot_body.set_dof(dof_value_map)


class PR2Collides(robot_predicates.Collides):
    pass


class PR2RCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.opt_coeff = const.RCOLLIDES_OPT_COEFF
        self.attr_inds = OrderedDict(
            [
                (params[0], list(ATTRMAP[params[0]._type])),
                (params[1], list(ATTRMAP[params[1]._type])),
            ]
        )
        super(PR2RCollides, self).__init__(
            name, params, expected_param_types, env, debug
        )
        self.dsafe = const.RCOLLIDES_DSAFE

    def resample(self, negated, t, plan):
        target_pose = self.obstacle.pose[:, t]
        return resample_bp_around_target(
            self, t, plan, target_pose, dist=const.TABLE_SAMPLING_RADIUS
        )

    def set_active_dof_inds(self, robot_body, reset=False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()
            ]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[
                dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()
            ]
            robot.SetActiveDOFs(
                dof_inds, DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis, [0, 0, 1]
            )
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {
            "backHeight": back_height,
            "lArmPose": l_arm_pose,
            "lGripper": l_gripper,
            "rArmPose": r_arm_pose,
            "rGripper": r_gripper,
        }
        robot_body.set_dof(dof_value_map)
