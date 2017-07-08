from core.util_classes import robot_predicates
from core.util_classes.common_predicates import ExprPredicate
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import baxter_sampling
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
from openravepy import DOFAffine, quatRotateDirection, matrixFromQuat
import numpy as np
import core.util_classes.baxter_constants as const
from core.util_classes.param_setup import ParamSetup
# Attribute map used in baxter domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {"Robot": (("lArmPose", np.array(range(7), dtype=np.int)),
                     ("lGripper", np.array([0], dtype=np.int)),
                     ("rArmPose", np.array(range(7), dtype=np.int)),
                     ("rGripper", np.array([0], dtype=np.int)),
                     ("pose", np.array([0], dtype=np.int)),
                     ("time", np.array([0], dtype=np.int))),
           "RobotPose": (("lArmPose", np.array(range(7), dtype=np.int)),
                         ("lGripper", np.array([0], dtype=np.int)),
                         ("rArmPose", np.array(range(7), dtype=np.int)),
                         ("rGripper", np.array([0], dtype=np.int)),
                         ("value", np.array([0], dtype=np.int))),
           "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                   ("rotation", np.array([0,1,2], dtype=np.int))),
           "EEPose": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "Target": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "Table": (("pose", np.array([0,1,2], dtype=np.int)),
                     ("rotation", np.array([0,1,2], dtype=np.int))),
           "Obstacle": (("pose", np.array([0,1,2], dtype=np.int)),
                        ("rotation", np.array([0,1,2], dtype=np.int))),
           "Basket": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "BasketTarget": (("value", np.array([0,1,2], dtype=np.int)),
                            ("rotation", np.array([0,1,2], dtype=np.int))),
           "Washer": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int)),
                      ("door", np.array([0], dtype=np.int))),
           "WasherPose": (("value", np.array([0,1,2], dtype=np.int)),
                          ("rotation", np.array([0,1,2], dtype=np.int)),
                          ("door", np.array([0], dtype=np.int))),
           "Cloth": (("pose", np.array([0,1,2], dtype=np.int)),
                     ("rotation", np.array([0,1,2], dtype=np.int))),
           "ClothTarget": (("value", np.array([0,1,2], dtype=np.int)),
                     ("rotation", np.array([0,1,2], dtype=np.int))),
           "EEVel": (("value", np.array([0], dtype=np.int)))
          }

"""
    Movement Constraints Family
"""

class BaxterAt(robot_predicates.At):
    pass

class BaxterClothAt(robot_predicates.At):
    pass

class BaxterRobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 17
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterRobotAt, self).__init__(name, params, expected_param_types, env)

class BaxterWasherAt(robot_predicates.RobotAt):

        # RobotAt, Washer, WasherPose

        def __init__(self, name, params, expected_param_types, env=None):
            self.attr_dim = 7
            self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                     (params[1], list(ATTRMAP[params[1]._type]))])
            super(BaxterWasherAt, self).__init__(name, params, expected_param_types, env)

class BaxterIsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list((ATTRMAP[params[0]._type][0], ATTRMAP[params[0]._type][2], ATTRMAP[params[0]._type][4])))])
        self.dof_cache = None
        super(BaxterIsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        dof_map = robot_body._geom.dof_map
        dof_inds = np.r_[dof_map["lArmPose"], dof_map["rArmPose"]]
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((len(dof_inds),1))
        active_lb = lb_limit[dof_inds].reshape((len(dof_inds),1))
        joint_move = (active_ub-active_lb)/const.JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= const.BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/const.JOINT_MOVE_FACTOR
        val = np.vstack((joint_move, const.BASE_MOVE*np.ones((const.BASE_DIM, 1)), joint_move, const.BASE_MOVE*np.ones((const.BASE_DIM, 1))))
        A = np.eye(len(val)) - np.eye(len(val), k=len(val)/2) - np.eye(len(val), k=-len(val)/2)
        b = np.zeros((len(val),1))
        self.base_step = const.BASE_MOVE*np.ones((const.BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class BaxterWasherWithinJointLimit(robot_predicates.WithinJointLimit):
    # BaxterWasherWithinJointLimit Washer

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], [(ATTRMAP[params[0]._type][2])])])
        super(BaxterWasherWithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        dof_map = robot_body._geom.dof_map
        dof_inds = dof_map["door"]
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((1,1))
        active_lb = lb_limit[dof_inds].reshape((1,1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(1)
        A_up_limit = np.eye(1)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2,1))
        return A, b, val

class BaxterWithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-2]))])
        super(BaxterWithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        dof_map = robot_body._geom.dof_map
        dof_inds = np.r_[dof_map["lArmPose"], dof_map["lGripper"], dof_map["rArmPose"], dof_map["rGripper"]]
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((const.JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((const.JOINT_DIM,1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(const.JOINT_DIM)
        A_up_limit = np.eye(const.JOINT_DIM)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2*const.JOINT_DIM,1))
        joint_move = (active_ub-active_lb)/const.JOINT_MOVE_FACTOR
        self.base_step = const.BASE_MOVE*np.ones((const.BASE_DIM,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class BaxterStationary(robot_predicates.Stationary):
    pass

class BaxterStationaryCloth(robot_predicates.Stationary):
    pass

class BaxterStationaryWasher(robot_predicates.StationaryBase):

    # BaxterStationaryWasher, Washer (Only pose, rotation)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][:2])])
        self.attr_dim = 6
        super(BaxterStationaryWasher, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-2]])])
        self.attr_dim = const.BASE_DIM
        super(BaxterStationaryBase, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-2]))])
        self.attr_dim = const.TWOARMDIM
        super(BaxterStationaryArms, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryW(robot_predicates.StationaryW):
    pass

class BaxterStationaryNEq(robot_predicates.StationaryNEq):
    pass

"""
    Grasping Pose Constraints Family
"""

class BaxterGraspValid(robot_predicates.GraspValid):
    pass

class BaxterClothGraspValid(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0], ('rotation', np.array([1]))]), (params[1], [ATTRMAP[params[1]._type][0], ('rotation', np.array([1]))])])

        self.attr_dim = 4
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.array([[0,0,0,-np.pi/2]]).T, np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothGraspValid, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class BaxterGraspValidPos(BaxterGraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]),(params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        super(BaxterGraspValidPos, self).__init__(name, params, expected_param_types, env, debug)

class BaxterGraspValidRot(BaxterGraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]]),(params[1], [ATTRMAP[params[1]._type][1]])])
        self.attr_dim = 3
        super(BaxterGraspValidRot, self).__init__(name, params, expected_param_types, env, debug)

class BaxterBasketGraspLeftPos(BaxterGraspValidPos):
    # BaxterBasketGraspLeftPos, EEPose, BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        # target_pos = params[1].value
        # target_pos[:2] /= np.linalg.norm(target_pos[:2])
        # target_pos *= np.array([1, 1, 0])
        # orient_mat = matrixFromQuat(quatRotateDirection(target_pos, [1, 0, 0]))[:3, :3]
        # A = np.c_[orient_mat, -orient_mat]

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [const.BASKET_OFFSET], [0]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(robot_predicates.GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class BaxterBasketGraspLeftRot(BaxterGraspValidRot):
    # BaxterBasketGraspLeftRot, EEPose, BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]]), (params[1], [ATTRMAP[params[1]._type][1]])])
        self.attr_dim = 3
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.array([[-np.pi/2], [np.pi/2], [-np.pi/2]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(robot_predicates.GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class BaxterBasketGraspRightPos(BaxterGraspValidPos):
    # BaxterBasketGraspLeftPos, EEPose, BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        # target_pos = params[1].value
        # target_pos[:2] /= np.linalg.norm(target_pos[:2])
        # target_pos *= [1, 1, 0]
        # orient_mat = matrixFromQuat(quatRotateDirection(target_pos, [1, 0, 0]))[:3, :3]

        # A = np.c_[orient_mat, -orient_mat]
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [-const.BASKET_OFFSET], [0]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(robot_predicates.GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class BaxterBasketGraspRightRot(BaxterGraspValidRot):
    # BaxterBasketGraspLeftRot, EEPose, BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]]), (params[1], [ATTRMAP[params[1]._type][1]])])
        self.attr_dim = 3
        self.ee_pose, self.target = params
        attr_inds = self.attr_inds

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.zeros((self.attr_dim,1)), np.array([[-np.pi/2], [np.pi/2], [-np.pi/2]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(robot_predicates.GraspValid, self).__init__(name, e, attr_inds, params, expected_param_types)
        self.spacial_anchor = True

class BaxterEEGraspValid(robot_predicates.EEGraspValid):

    # BaxterEEGraspValid EEPose Washer
    # TODO EEGraspValid's gradient is not working properly, go back and fix it
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.EEGRASP_VALID_COEFF
        self.rot_coeff = const.EEGRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 6
        super(BaxterEEGraspValid, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     return resample_ee_grasp_valid(self, negated, t, plan)

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
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        ee_pos, ee_rot = x[:3], x[3:6]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        obj_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return robot_trans, obj_trans, axises, obj_axises, arm_joints

    def washer_ee_check_f(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # print x[:3].flatten(), x[3:6].flatten()
        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    def washer_ee_check_jac(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]

        joint_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        washer_jac = np.array([np.cross(axis, robot_pos - x[-7:-4, 0]) for axis in axises]).T

        obj_jac = -1 * np.array([np.cross(axis, obj_pos - obj_trans[:3,3]) for axis in axises]).T
        dist_jac = np.hstack([-np.eye(3), obj_jac, np.eye(3), washer_jac, 1*joint_jac])
        return dist_jac

    def washer_ee_rot_check_f(self, x, rel_rot):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        rot_val = self.rot_lock_f(obj_trans, washer_trans, rel_rot)
        return rot_val

    def washer_ee_rot_check_jac(self, x, rel_rot):
        robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        rot_jacs = []
        for local_dir in np.eye(3):
            obj_dir = np.dot(obj_trans[:3,:3], local_dir)
            world_dir = robot_trans[:3,:3].dot(local_dir)
            # computing robot's jacobian
            door_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir)) for joint in arm_joints]).T.copy()
            door_jac = door_jac.reshape((1, len(arm_joints)))

            washer_jac = np.array([np.dot(obj_dir, np.cross(axis, world_dir)) for axis in axises])
            washer_jac = np.r_[[0,0,0], washer_jac].reshape((1, 6))

            # computing object's jacobian
            obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in obj_axises])
            obj_jac = np.r_[[0,0,0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix

            rot_jacs.append(np.hstack([obj_jac, washer_jac, 1*door_jac]))
        rot_jac = np.vstack(rot_jacs)

        return rot_jac

    def stacked_f(self, x):
        rel_pt = np.array([-0.035,0.055,-0.1])
        rel_rot = np.array([[0,0,0], [0,0,0], [0,0,1]])
        return np.vstack([self.coeff * self.washer_ee_check_f(x, rel_pt), self.rot_coeff * self.washer_ee_rot_check_f(x, rel_rot)])

    def stacked_grad(self, x):
        rel_pt = np.array([-0.035,0.055,-0.1])
        rel_rot = np.array([[0,0,0], [0,0,0], [0,0,1]])
        return np.vstack([self.coeff * self.washer_ee_check_jac(x, rel_pt), self.rot_coeff * self.washer_ee_rot_check_jac(x, rel_rot)])

"""
    Gripper Constraints Family
"""

class BaxterCloseGripperLeft(robot_predicates.InContact):

    # BaxterCloseGripperLeft Robot EEPose Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        super(BaxterCloseGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

class BaxterCloseGripperRight(robot_predicates.InContact):

    # BaxterCloseGripperRight Robot EEPose Target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][3]])])
        super(BaxterCloseGripperRight, self).__init__(name, params, expected_param_types, env, debug)

class BaxterOpenGripperLeft(BaxterCloseGripperLeft):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterOpenGripperLeft, self).__init__(name, params, expected_param_types, env, debug)
        self.expr = self.neg_expr

class BaxterOpenGripperRight(BaxterCloseGripperRight):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterOpenGripperRight, self).__init__(name, params, expected_param_types, env, debug)
        self.expr = self.neg_expr

class BaxterCloseGrippers(robot_predicates.InContacts):

    # BaxterBasketCloseGripper robot EEPose EEPose BasketTarget

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1], ATTRMAP[params[0]._type][3]])])
        super(BaxterCloseGrippers, self).__init__(name, params, expected_param_types, env, debug)

class BaxterOpenGrippers(BaxterCloseGrippers):

    # InContact robot EEPose EEPose BasketTarget

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterOpenGrippers, self).__init__(name, params, expected_param_types, env, debug)
        self.expr = self.neg_expr

"""
    Collision Constraints Family
"""

class BaxterObstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.DIST_SAFE):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[3], list(ATTRMAP[params[3]._type]))])
        super(BaxterObstructs, self).__init__(name, params, expected_param_types, env, debug, tol)
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_basket_obstructs(self, negated, t, plan)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18)), DOFAffine.RotationAxis, [0,0,1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

class BaxterObstructsCloth(BaxterObstructs):
    pass

class BaxterObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.DIST_SAFE):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[3], list(ATTRMAP[params[3]._type])),
                                 (params[4], list(ATTRMAP[params[4]._type]))])
        super(BaxterObstructsHolding, self).__init__(name, params, expected_param_types, env, debug, tol)
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_basket_obstructs_holding(self, negated, t, plan)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18)), DOFAffine.RotationAxis, [0,0,1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

class BaxterObstructsHoldingCloth(BaxterObstructsHolding):
    pass

class BaxterCollides(robot_predicates.Collides):

    # Collides Basket Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = -const.COLLIDE_COEFF
        self.neg_coeff = const.COLLIDE_COEFF
        super(BaxterCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.COLLIDES_DSAFE

class BaxterRCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        super(BaxterRCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.RCOLLIDES_DSAFE

    def resample(self, negated, t, plan):
        return baxter_sampling.resample_basket_obstructs(self, negated, t, plan)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset and self.dof_cache is not None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif not reset and self.dof_cache is None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18)), DOFAffine.RotationAxis, [0,0,1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

"""
    EEReachable Constraints Family
"""

class BaxterEEReachable(robot_predicates.EEReachable):
    def __init__(self, name, params, expected_param_types, active_range = (-const.EEREACHABLE_STEPS, const.EEREACHABLE_STEPS), env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[2], list(ATTRMAP[params[2]._type]))])
        self.attr_dim = 23
        self.coeff = const.EEREACHABLE_COEFF
        self.rot_coeff = const.EEREACHABLE_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 3+3*(1+(active_range[1] - active_range[0]))
        super(BaxterEEReachable, self).__init__(name, params, expected_param_types, active_range, env, debug)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([0, 0, const.RETREAT_DIST])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_rrt(self, negated, t, plan, inv = False)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body, arm):
        if not arm == "right" and not arm == "left":
            assert PredicateException("Invalid Arm Specified")
        # Provide functionality of Obtaining Robot information
        if arm == "right":
            tool_link = robot_body.env_body.GetLink("right_gripper")
        else:
            tool_link = robot_body.env_body.GetLink("left_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        if arm == "right":
            arm_inds = list(range(10,17))
        else:
            arm_inds = list(range(2,9))
        return robot_trans, arm_inds

class BaxterEEReachableLeft(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.arm = "left"
        super(BaxterEEReachableLeft, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class BaxterEEReachableRight(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(BaxterEEReachableRight, self).__init__(name, params, expected_param_types, (-steps, steps), env, debug)

class BaxterEEReachableLeftInv(BaxterEEReachableLeft):

    # BaxterEEReachableLeftInv Robot, StartPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_rrt(self, negated, t, plan, inv = True)

class BaxterEEReachableRightInv(BaxterEEReachableRight):

    # EEreachableInv Robot, StartPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_rrt(self, negated, t, plan, inv='True')

class BaxterEEReachableLeftVer(BaxterEEReachableLeft):

    # BaxterEEReachableVerLeftPos Robot, RobotPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_ver(self, negated, t, plan)

class BaxterEEReachableRightVer(BaxterEEReachableRight):

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_ver(self, negated, t, plan)

class BaxterEEApproachLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEEApproachLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = True)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

class BaxterEEApproachRight(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(BaxterEEApproachRight, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = True)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

class BaxterEERetreatLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEERetreatLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = False)

class BaxterEERetreatRight(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(BaxterEERetreatRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = False)

"""
    InGripper Constraint Family
"""

class BaxterInGripper(robot_predicates.InGripper):

    # InGripper, Robot, Object

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.IN_GRIPPER_COEFF
        self.rot_coeff = const.IN_GRIPPER_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        super(BaxterInGripper, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body, arm):
        if not arm == "right" and not arm == "left":
            PredicateException("Invalid Arm Specified")
        # Provide functionality of Obtaining Robot information
        if arm == "right":
            tool_link = robot_body.env_body.GetLink("right_gripper")
        else:
            tool_link = robot_body.env_body.GetLink("left_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        if arm == "right":
            arm_inds = list(range(10,17))
        else:
            arm_inds = list(range(2,9))
        return robot_trans, arm_inds

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.pos_check_f(x), self.rot_coeff * self.rot_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.pos_check_jac(x), self.rot_coeff * self.rot_check_jac(x)])

class BaxterInGripperLeft(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "left"
        self.eval_dim = 4
        super(BaxterInGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

class BaxterInGripperRight(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "right"
        self.eval_dim = 4
        super(BaxterInGripperRight, self).__init__(name, params, expected_param_types, env, debug)

class BaxterBasketInGripper(BaxterInGripper):

    # BaxterBasketInGripper Robot, Basket

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.eval_dim = 12
        super(BaxterBasketInGripper, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_basket_moveholding(self, negated, t, plan)

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_f(x), self.rot_coeff * self.both_arm_rot_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_jac(x), self.rot_coeff * self.both_arm_rot_check_jac(x)])

class BaxterWasherInGripper(BaxterInGripperLeft):

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

    def robot_robot_kinematics(self, x):
        robot_body = self.robot.openrave_body
        body = robot_body.env_body
        obj_body = self.obj.openrave_body
        obj = obj_body.env_body
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        self.set_washer_poses(x, obj_body)
        obj_trans, obj_arm_inds = self.get_washer_info(obj_body)
        obj_joints = [obj.GetJointFromDOFIndex(ind) for ind in obj_arm_inds]

        pos, rot = x[-7:-4], x[-4:-1]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(pos, rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        return robot_trans, obj_trans, arm_joints, obj_joints, axises

    def ee_contact_check_f(self, x, rel_pt):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)

        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[rel_pt, 1])[:3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    def ee_contact_check_jac(self, x, rel_pt):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)

        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[rel_pt, 1])[:3]
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()

        joint_jac = np.array([np.cross(joint.GetAxis(), obj_pos - joint.GetAnchor()) for joint in obj_joints]).T.copy()
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos).reshape((3,1))
        obj_jac = -1 * np.array([np.cross(axis, obj_pos - x[-7:-4, 0]) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac, -joint_jac]
        dist_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm)
        return dist_jac

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

        ee_pos, ee_rot = x[-7:-4], x[-4:-1]
        obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints

    def stacked_f(self, x):
        rel_pt = np.array([-0.035,0.055,-0.1])
        return np.vstack([self.coeff * self.ee_contact_check_f(x, rel_pt), self.rot_coeff * self.rot_check_f(x)])

    def stacked_grad(self, x):
        rel_pt = np.array([-0.035,0.055,-0.1])
        return np.vstack([self.coeff * self.ee_contact_check_jac(x, rel_pt), self.rot_coeff * np.c_[self.rot_check_jac(x), 0]])

class BaxterClothInGripperRight(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "right"
        self.eval_dim = 3
        super(BaxterClothInGripperRight, self).__init__(name, params, expected_param_types, env, debug)

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_cloth_in_gripper(self, negated, t, plan)

    def stacked_f(self, x):
        return self.coeff * self.pos_check_f(x)

    def stacked_grad(self, x):
        return self.coeff * self.pos_check_jac(x)

class BaxterClothInGripperLeft(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "left"
        self.eval_dim = 3
        super(BaxterClothInGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

    def resample(self, negated, t, plan):
        print "resample {}".format(self.get_type())
        return baxter_sampling.resample_cloth_in_gripper(self, negated, t, plan)

    def stacked_f(self, x):
        return self.coeff * self.pos_check_f(x)

    def stacked_grad(self, x):
        return self.coeff * self.pos_check_jac(x)

"""
    Basket Constraint Family
"""

class BaxterBasketLevel(robot_predicates.BasketLevel):
    # BaxterBasketLevel Basket
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
            self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
            self.attr_dim = 3
            self.basket = params[0]
            super(BaxterBasketLevel, self).__init__(name, params, expected_param_types, env, debug)

class BaxterObjectWithinRotLimit(robot_predicates.ObjectWithinRotLimit):
    # BaxterObjectWithinRotLimit Object
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
            self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
            self.attr_dim = 3
            self.object = params[0]
            super(BaxterObjectWithinRotLimit, self).__init__(name, params, expected_param_types, env, debug)

class BaxterGrippersLevel(robot_predicates.GrippersLevel):
    # BaxterGrippersLevel Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 1
        self.opt_coeff = 1
        self.eval_f = lambda x: self.both_arm_pos_check(x)[0]
        self.eval_grad = lambda x: self.both_arm_pos_check(x)[1]
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        self.eval_dim = 6
        super(BaxterGrippersLevel, self).__init__(name, params, expected_param_types, env, debug)

    def get_robot_info(self, robot_body, arm = "left"):
        if not arm == "right" and not arm == "left":
            PredicateException("Invalid Arm Specified")
        # Provide functionality of Obtaining Robot information
        if arm == "right":
            tool_link = robot_body.env_body.GetLink("right_gripper")
        else:
            tool_link = robot_body.env_body.GetLink("left_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        if arm == "right":
            arm_inds = list(range(10,17))
        else:
            arm_inds = list(range(2,9))
        return robot_trans, arm_inds

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def both_arm_pos_check(self, x):
        """
            This function is used to check whether:
                both grippers are at the same height

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")

        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]

        l_tool_link = robot_body.env_body.GetLink("left_gripper")
        l_manip_trans = l_tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        l_pose = OpenRAVEBody.obj_pose_from_transform(l_manip_trans)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(l_pose[:3], l_pose[3:])
        l_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        r_tool_link = robot_body.env_body.GetLink("right_gripper")
        r_manip_trans = r_tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        r_pose = OpenRAVEBody.obj_pose_from_transform(r_manip_trans)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(r_pose[:3], r_pose[3:])
        r_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        l_pos_val, l_pos_jac = self.pos_error(robot_left_trans, robot_right_trans, r_axises, left_arm_joints, right_arm_joints, [0,0,0], "left")
        r_pos_val, r_pos_jac = self.pos_error(robot_right_trans, robot_left_trans, l_axises, right_arm_joints, left_arm_joints, [0,0,0], "right")

        pos_val = np.vstack([l_pos_val, r_pos_val])
        pos_jac = np.vstack([l_pos_jac, r_pos_jac])
        return pos_val, pos_jac

    def pos_error(self, robot_arm_trans, robot_aux_arm_trans, axises, arm_joints, aux_joints, rel_pt, arm):
        """
            This function calculates the value and the jacobian of the displacement between the height of the gripper and that of the inactive gripper

            robot_trans: active robot gripper's rave_body transformation
            robot_aux_arm_trans: inactive gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        vert_axis = np.array([0, 0, 1]).reshape((3, 1))
        gp = rel_pt
        robot_pos = robot_arm_trans[:3, 3]
        robot_aux_pos = robot_aux_arm_trans[:3, 3]
        dist_val = (robot_pos.flatten() - robot_aux_pos.flatten()).reshape((3, 1)) * vert_axis
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy() * vert_axis
        # Calculate jacobian for the robot base
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos - np.zeros((3,))).reshape((3, 1)) * vert_axis
        # Calculate object jacobian
        aux_jac = -1*np.array([np.cross(joint.GetAxis(), robot_aux_pos.flatten() - joint.GetAnchor()) for joint in aux_joints]).T.copy() * vert_axis
        if arm == "left":
            dist_jac = np.hstack((arm_jac, np.zeros((3, 1)), aux_jac, np.zeros((3, 1)), base_jac))
        elif arm == "right":
            dist_jac = np.hstack((aux_jac, np.zeros((3, 1)), arm_jac, np.zeros((3, 1)), base_jac))

        return dist_val, dist_jac

class BaxterEERetiming(robot_predicates.EERetiming):
    # BaxterVelocity Robot EEVel

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list((ATTRMAP[params[0]._type]))), (params[1], [ATTRMAP[params[1]._type]])])
        self.coeff = 1
        self.eval_f = lambda x: self.vel_check(x)[0]
        self.eval_grad = lambda x: self.vel_check(x)[1]
        self.eval_dim = 1

        super(BaxterEERetiming, self).__init__(name, params, expected_param_types, env, debug)
        self.spacial_anchor = False

    # def resample(self, negated, t, plan):
    #     return resample_retiming(self, negated, t, plan)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        robot_body.set_pose([0,0,base_pose])

        dof_value_map = {"lArmPose": l_arm_pose.reshape((7,)),
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose.reshape((7,)),
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body, arm = "right"):
        if not arm == "right" and not arm == "left":
            PredicateException("Invalid Arm Specified")
        # Provide functionality of Obtaining Robot information
        if arm == "right":
            tool_link = robot_body.env_body.GetLink("right_gripper")
        else:
            tool_link = robot_body.env_body.GetLink("left_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        if arm == "right":
            arm_inds = list(range(10,17))
        else:
            arm_inds = list(range(2,9))
        return robot_trans, arm_inds

    def vel_check(self, x):
        """
            Check whether val_check(x)[0] <= 0
            x = lArmPose(t), lGripper(t), rArmPose(t), rGripper(t), pose(t), time(t), EEvel.value(t),
                lArmPose(t+1), lGripper(t+1), rArmPose(t+1), rGripper(t+1), pose(t+1), time(t), EEvel.value(t+1)
                dim (38, 1)
        """

        velocity = x[18]
        assert velocity != 0
        robot_body = self._param_to_body[self.robot]
        self.set_robot_poses(x[0:17], robot_body)
        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
        left_t0 = robot_left_trans[:3, 3]
        right_t0 = robot_right_trans[:3, 3]

        self.set_robot_poses(x[19:-2], robot_body)
        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
        left_t1 = robot_left_trans[:3, 3]
        right_t1 = robot_right_trans[:3, 3]

        left_time = np.linalg.norm(left_t1 - left_t0) / float(velocity)
        right_time = np.linalg.norm(right_t1 - right_t0) / float(velocity)
        time_spend = max(left_time, right_time)
        val = np.array([[x[36] - x[17] - time_spend]])

        jac = np.zeros((1, 38))
        jac[0, 17] = -1
        jac[0, 36] = 1

        return val, jac

class BaxterObjRelPoseConstant(robot_predicates.ObjRelPoseConstant):

    # BxterObjRelPoseConstant Basket Cloth
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][0])), (params[1], list(ATTRMAP[params[1]._type][0]))])
        self.attr_dim = 3
        super(BaxterObjRelPoseConstant).__init__(name, params, expected_param_types, env, debug)
