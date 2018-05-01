from core.util_classes import robot_predicates
from core.util_classes.common_predicates import ExprPredicate
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import baxter_sampling
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import ctrajoptpy
from collections import OrderedDict
from openravepy import DOFAffine, Environment, quatRotateDirection, matrixFromQuat
import numpy as np
import core.util_classes.baxter_constants as const
from core.util_classes.items import Box, Can, Sphere
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
           "Rotation": [("value", np.array([0], dtype=np.int))],
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
           "Fabric": (("gripleft", np.array([0,1,2], dtype=np.int)),
                      ("gripright", np.array([0,1,2], dtype=np.int))),
           "EEVel": (("value", np.array([0], dtype=np.int)))
          }

"""
    Movement Constraints Family
"""

class BaxterAt(robot_predicates.At):
    pass

class BaxterClothAt(robot_predicates.At):
    pass

class BaxterEdgeAt(robot_predicates.At):
    pass

class BaxterClothAtPose(robot_predicates.AtPose):
    pass

class BaxterRobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 17
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterRobotAt, self).__init__(name, params, expected_param_types, env)

class BaxterPoseAtRotation(robot_predicates.RobotAt):

    # RobotAt, RobotPose, Rotation

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 1
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][4:5])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterPoseAtRotation, self).__init__(name, params, expected_param_types, env)

class BaxterWasherAt(robot_predicates.RobotAt):

        # RobotAt, Washer, WasherPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 7
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterWasherAt, self).__init__(name, params, expected_param_types, env)

class BaxterClothAtHandle(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.cloth, self.target = params
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])

        self.attr_dim = 3
        target_rot = self.target.rotation[0, 0]
        handle_dist = const.BASKET_OFFSET
        offset = np.array([[handle_dist*np.cos(target_rot)], [handle_dist*np.sin(target_rot)], [const.BASKET_SHALLOW_GRIP_OFFSET]])
        target_pos = offset

        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = target_pos, np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothAtHandle, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class BaxterPosePair(robot_predicates.HLAnchor):

    # BaxterPosePair RobotPose RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 6
        self.one, self.two = params
        self.attr_inds = OrderedDict([(self.one, [("value", np.array([0,1,2], dtype=np.int))]),
                                      (self.two, [("value", np.array([0,1,2], dtype=np.int))])])
        super(BaxterPosePair, self).__init__(name, params, expected_param_types, env)

class BaxterClothInBasket(robot_predicates.HLAnchor):

    # BaxterClothInBasket Cloth ClothTarget Basket

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 6
        self.one, self.two = params
        self.attr_inds = OrderedDict([(self.one, [("pose", np.array([0,1,2], dtype=np.int))]),
                                      (self.two, [("pose", np.array([0,1,2], dtype=np.int))])])
        super(BaxterClothInBasket, self).__init__(name, params, expected_param_types, env)

class BaxterClothInWasher(robot_predicates.HLAnchor):

    # BaxterClothInWasher Cloth Washer

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 6
        self.one, self.two = params
        self.attr_inds = OrderedDict([(self.one, [("pose", np.array([0,1,2], dtype=np.int))]),
                                      (self.two, [("pose", np.array([0,1,2], dtype=np.int))])])
        super(BaxterClothInWasher, self).__init__(name, params, expected_param_types, env)

class BaxterGrippersCenteredOverBasket(robot_predicates.HLAnchor):

    # BaxterGrippersCenteredOverBasket Robot Basket

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 17
        self.one, self.two = params
        self.attr_inds = OrderedDict([(self.one, [("lArmPose", np.array([0,1,2,3,4,5,6,7], dtype=np.int)),
                                                  ("rArmPose", np.array([0,1,2,3,4,5,6,7], dtype=np.int))]),
                                      (self.two, [("pose", np.array([0,1,2], dtype=np.int))])])
        super(BaxterClothInWasher, self).__init__(name, params, expected_param_types, env)

class BaxterIsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list((ATTRMAP[params[0]._type][0], ATTRMAP[params[0]._type][2], ATTRMAP[params[0]._type][4])))])
        self.dof_cache = None
        super(BaxterIsMP, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    #@profile
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

class BaxterWithinRotLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][4:5]))])
        super(BaxterWithinRotLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        val = np.vstack((-const.ROT_LB, const.ROT_UB))
        A = np.array([[-1], [1]])
        b = np.zeros((2,1))
        return A, b, val

class BaxterStationary(robot_predicates.Stationary):
    pass

class BaxterStationaryCloth(robot_predicates.Stationary):
    pass

class BaxterStationaryEdge(robot_predicates.Stationary):
    pass

class BaxterStationaryWasher(robot_predicates.StationaryBase):

    # BaxterStationaryWasher, Washer (Only pose, rotation)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][:2])])
        self.attr_dim = 6
        super(BaxterStationaryWasher, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryWasherDoor(robot_predicates.StationaryBase):

    # BaxterStationaryWasher, Washer (Only pose, rotation)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], ATTRMAP[params[0]._type][2:])])
        self.attr_dim = 1
        super(BaxterStationaryWasherDoor, self).__init__(name, params, expected_param_types, env)

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

# class BaxterClothGraspValid(ExprPredicate):
#     def __init__(self, name, params, expected_param_types, env=None, debug=False):
#         self.ee_pose, self.target = params
#         self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0], ('rotation', np.array([1]))]), (params[1], [ATTRMAP[params[1]._type][0], ('rotation', np.array([1]))])])

#         self.attr_dim = 4
#         A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
#         b, val = np.array([[0,0,0,-np.pi/2]]).T, np.zeros((self.attr_dim,1))
#         pos_expr = AffExpr(A, b)
#         e = EqExpr(pos_expr, val)
#         super(BaxterClothGraspValid, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
#         self.spacial_anchor = True

class BaxterClothGraspValid(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target = params
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0], ('rotation', np.array([0,1]))]), (params[1], [ATTRMAP[params[1]._type][0], ('rotation', np.array([0,1]))])])

        self.attr_dim = 5
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        b, val = np.array([[0,0,0,-np.pi/2,-np.pi/2]]).T, np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothGraspValid, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class BaxterClothBothGraspValidLeft(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target, self.edge = params
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0], ('rotation', np.array([0,1]))]), (params[1], [ATTRMAP[params[1]._type][0], ('rotation', np.array([0,1]))])])

        self.attr_dim = 5
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        rot = self.target.rotation[0, 0]
        dist = self.edge.geom.height / 2 + 0.1
        b, val = np.array([[np.sin(rot)*dist, -np.cos(rot)*dist, 0, -np.pi/2,-np.pi/2]]).T, np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothBothGraspValidLeft, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
        self.spacial_anchor = True

class BaxterClothBothGraspValidRight(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.ee_pose, self.target, self.edge = params
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0], ('rotation', np.array([0,1]))]), (params[1], [ATTRMAP[params[1]._type][0], ('rotation', np.array([0,1]))])])

        self.attr_dim = 5
        A = np.c_[np.eye(self.attr_dim), -np.eye(self.attr_dim)]
        rot = self.target.rotation[0, 0]
        dist = self.edge.geom.height / 2 + 0.1
        b, val = np.array([[-np.sin(rot)*dist, np.cos(rot)*dist, 0, np.pi/2,-np.pi/2]]).T, np.zeros((self.attr_dim,1))
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothBothGraspValidRight, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)
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

class BaxterBasketGraspValidPos(robot_predicates.PosePredicate):

    # BaxterBasketGraspValid EEPose, EEPose, BasketTarget

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])), (params[1], list(ATTRMAP[params[1]._type])), (params[2], list(ATTRMAP[params[2]._type]))])
        self.eval_dim = 6
        self._env = env
        self.l_ee_pose, self.r_ee_pose, self.basket_target = params

        self.coeff = const.GRASP_VALID_COEFF
        self.rot_coeff = const.GRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.grip_offset = const.BASKET_GRIP_OFFSET

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(BaxterBasketGraspValidPos, self).__init__(name, e, self.attr_inds, params, expected_param_types, debug=debug, priority=1)

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_basket_moveholding(self, negated, t, plan)

    def pose_basket_kinematics(self, x):
        left_ee_pos, left_ee_rot = x[:3], x[3:6]
        left_trans = OpenRAVEBody.transform_from_obj_pose(left_ee_pos, left_ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(left_ee_pos, left_ee_rot)
        left_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        right_ee_pos, right_ee_rot = x[6:9], x[9:12]
        right_trans = OpenRAVEBody.transform_from_obj_pose(right_ee_pos, right_ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(right_ee_pos, right_ee_rot)
        right_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        basket_pose, basket_rot = x[12:15], x[15:]
        basket_trans = OpenRAVEBody.transform_from_obj_pose(basket_pose, basket_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(basket_pose, basket_rot)
        basket_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        return left_trans, right_trans, basket_trans, left_axises, right_axises, basket_axises

    #@profile
    def both_arm_pos_check_f(self, x):
        """
            This function is used to check whether:
                basket is at both robot gripper's center

            x -> left_ee_pos, left_ee_rot, right_ee_pos, right_ee_rot, basket_pos, basket_rot
        """
        left_rel_pt = [const.BASKET_OFFSET, self.grip_offset, 0]
        right_rel_pt = [-const.BASKET_OFFSET, self.grip_offset, 0]
        # left_rel_pt = [0, 0, -const.BASKET_NARROW_OFFSET]
        # right_rel_pt = [0, 0, const.BASKET_NARROW_OFFSET]
        left_trans, right_trans, basket_trans, left_axises, right_axises, basket_axises = self.pose_basket_kinematics(x)

        left_target_pos = basket_trans.dot(np.r_[left_rel_pt, 1])[:3]
        left_pos = left_trans[:3, 3]
        left_dist_val = (left_target_pos - left_pos).reshape((3,1))

        right_target_pos = basket_trans.dot(np.r_[right_rel_pt, 1])[:3]
        right_pos = right_trans[:3, 3]
        right_dist_val = (right_target_pos - right_pos).reshape((3,1))

        return np.vstack([left_dist_val, right_dist_val])

    def both_arm_ee_check_jac(self, x):
        left_rel_pt = [const.BASKET_OFFSET, self.grip_offset, 0]
        right_rel_pt = [-const.BASKET_OFFSET, self.grip_offset, 0]
        # left_rel_pt = [0, 0, -const.BASKET_NARROW_OFFSET]
        # right_rel_pt = [0, 0, const.BASKET_NARROW_OFFSET]
        left_trans, right_trans, basket_trans, left_axises, right_axises, basket_axises = self.pose_basket_kinematics(x)

        left_target_pos = basket_trans.dot(np.r_[left_rel_pt, 1])[:3]
        left_pos = left_trans[:3, 3]
        left_dist_val = (left_target_pos - left_pos).flatten()

        right_target_pos = basket_trans.dot(np.r_[right_rel_pt, 1])[:3]
        right_pos = right_trans[:3, 3]
        right_dist_val = (right_target_pos - right_pos).flatten()

        basket_pos_jac = np.vstack([np.array([np.cross(axis, left_target_pos - x[12:15, 0]) for axis in basket_axises]).T, \
                                                             np.array([np.cross(axis, right_target_pos - x[12:15, 0]) for axis in basket_axises]).T])

        left_jac = -1 * np.array([np.cross(axis, left_target_pos - left_pos) for axis in left_axises]).T
        right_jac = -1 * np.array([np.cross(axis, right_target_pos - right_pos) for axis in right_axises]).T

        return np.hstack([-np.eye(6,3), np.vstack([left_jac, np.zeros((3,3))]), -np.eye(6,3,-3), np.vstack([np.zeros((3,3)), right_jac]), np.eye(6,3)+np.eye(6,3,-3), np.zeros((6,3))])

    def stacked_f(self, x):
        return self.coeff * self.both_arm_pos_check_f(x)

    def stacked_grad(self, x):
        return self.coeff*self.both_arm_ee_check_jac(x)

class BaxterBasketGraspValidShallowPos(BaxterBasketGraspValidPos):

    # BaxterBasketGraspValid EEPose, EEPose, BasketTarget

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterBasketGraspValidShallowPos, self).__init__(name, params, expected_param_types, env, debug)
        self.grip_offset = const.BASKET_SHALLOW_GRIP_OFFSET

class BaxterBasketGraspValidRot(robot_predicates.PosePredicate):

    # BaxterBasketGraspValid EEPose, EEPose, BasketTarget

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])), (params[1], list(ATTRMAP[params[1]._type])), (params[2], list(ATTRMAP[params[2]._type]))])
        self.eval_dim = 6
        self._env = env
        self.l_ee_pose, self.r_ee_pose, self.basket_target = params

        self.coeff = const.GRASP_VALID_COEFF
        self.rot_coeff = const.GRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad

        e = EqExpr(Expr(self.eval_f, self.eval_grad), np.zeros((self.eval_dim, 1)))
        super(BaxterBasketGraspValidRot, self).__init__(name, e, self.attr_inds, params, expected_param_types, debug=debug, priority=1)

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_basket_moveholding(self, negated, t, plan)

    def pose_basket_kinematics(self, x):
        left_ee_pos, left_ee_rot = x[:3], x[3:6]
        left_trans = OpenRAVEBody.transform_from_obj_pose(left_ee_pos, left_ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(left_ee_pos, left_ee_rot)
        left_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        right_ee_pos, right_ee_rot = x[6:9], x[9:12]
        right_trans = OpenRAVEBody.transform_from_obj_pose(right_ee_pos, right_ee_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(right_ee_pos, right_ee_rot)
        right_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        basket_pose, basket_rot = x[12:15], x[15:]
        basket_trans = OpenRAVEBody.transform_from_obj_pose(basket_pose, basket_rot)
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(basket_pose, basket_rot)
        basket_axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        return left_trans, right_trans, basket_trans, left_axises, right_axises, basket_axises

    #@profile
    def both_arm_rot_check_f(self, x):
        """
            This function is used to check whether:
                basket is at robot gripper's center
        """
        left_rot, right_rot, basket_rot = x[3:6], x[9:12], x[15:18]
        left_rot_error = np.array([basket_rot[0]-np.pi/2, [np.pi/2], [0]]) - left_rot
        right_rot_error = np.array([basket_rot[0]-np.pi/2, [np.pi/2], [0]]) - right_rot
        # left_rot_error = np.array([basket_rot[0], [np.pi/2], [0]]) - left_rot
        # right_rot_error = np.array([basket_rot[0], [np.pi/2], [0]]) - right_rot
        return np.vstack([left_rot_error, right_rot_error])

    def both_arm_ee_rot_check_jac(self, x):
        return np.hstack([np.zeros((6, 3)), -np.eye(6,3), np.zeros((6,3)), -np.eye(6,3,-3), np.zeros((6,3)), np.eye(6,1)+np.eye(6,1,-3), np.zeros((6, 2))])

    def stacked_f(self, x):
        return self.rot_coeff * self.both_arm_rot_check_f(x)

    def stacked_grad(self, x):
        return self.rot_coeff*self.both_arm_ee_rot_check_jac(x)

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
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [const.BASKET_OFFSET], [0.03]])
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
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [-const.BASKET_OFFSET], [0.03]])
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
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.EEGRASP_VALID_COEFF
        self.rot_coeff = const.EEGRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 6
        # rel_pt = np.array([-0.04, 0.07, -0.115]) # np.array([-0.035,0.055,-0.1])
        # self.rel_pt = np.array([-0.04,0.07,-0.1])
        self.rel_pt = np.array([0, 0.06, 0]) # np.zeros((3,))
        self.rot_dir = np.array([0,0,1])
        super(BaxterEEGraspValid, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     return baxter_sampling.resample_ee_grasp_valid(self, negated, t, plan)

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

    #@profile
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

    #@profile
    def washer_ee_check_f(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    #@profile
    def washer_ee_check_jac(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]

        joint_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        washer_jac = np.array([np.cross(axis, robot_pos - x[-7:-4, 0]) for axis in axises]).T

        obj_jac = -1 * np.array([np.cross(axis, obj_pos - obj_trans[:3,3]) for axis in axises]).T
        dist_jac = np.hstack([-np.eye(3), obj_jac, np.eye(3), washer_jac, 1*joint_jac])
        return dist_jac

    def washer_ee_rot_check_f(self, x, rot_dir):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_f(obj_trans, robot_trans, self.rot_dir)
        return x[3:6] - np.array(const.RESAMPLE_ROT).reshape((3,1))


    #@profile
    def washer_ee_rot_check_jac(self, x, rel_rot):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints, self.rot_dir)
        return np.hstack([np.zeros((3,3)), np.eye(3), np.zeros((3, 7))])

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_f(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_f(x, self.rot_dir)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_jac(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_jac(x, self.rot_dir)])


class BaxterEEOpenedDoorGraspValid(robot_predicates.EEGraspValid):

    # BaxterEEGraspValid EEPose Washer
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.EEGRASP_VALID_COEFF
        self.rot_coeff = const.EEGRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 6
        # rel_pt = np.array([-0.04, 0.07, -0.115]) # np.array([-0.035,0.055,-0.1])
        # self.rel_pt = np.array([-0.04,0.07,-0.1])
        self.rel_pt = np.array([0.0, 0.0, 0.0]) # np.array([0, 0.06, 0]) # np.zeros((3,))
        self.rot_dir = np.array([0,0,1])
        super(BaxterEEOpenedDoorGraspValid, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     return baxter_sampling.resample_ee_grasp_valid(self, negated, t, plan)

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

    #@profile
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

    #@profile
    def washer_ee_check_f(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    #@profile
    def washer_ee_check_jac(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]

        joint_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        washer_jac = np.array([np.cross(axis, robot_pos - x[-7:-4, 0]) for axis in axises]).T

        obj_jac = -1 * np.array([np.cross(axis, obj_pos - obj_trans[:3,3]) for axis in axises]).T
        dist_jac = np.hstack([-np.eye(3), obj_jac, np.eye(3), washer_jac, 1*joint_jac])
        return dist_jac

    def washer_ee_rot_check_f(self, x, rot_dir):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_f(obj_trans, robot_trans, self.rot_dir)
        return x[3:6] - np.array(const.RESAMPLE_OPENED_DOOR_ROT).reshape((3,1))


    #@profile
    def washer_ee_rot_check_jac(self, x, rel_rot):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints, self.rot_dir)
        return np.hstack([np.zeros((3,3)), np.eye(3), np.zeros((3, 7))])

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_f(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_f(x, self.rot_dir)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_jac(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_jac(x, self.rot_dir)])


class BaxterEEClosedDoorGraspValid(robot_predicates.EEGraspValid):

    # BaxterEEGraspValid EEPose Washer
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.EEGRASP_VALID_COEFF
        self.rot_coeff = const.EEGRASP_VALID_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.eval_dim = 6
        # rel_pt = np.array([-0.04, 0.07, -0.115]) # np.array([-0.035,0.055,-0.1])
        # self.rel_pt = np.array([-0.04,0.07,-0.1])
        self.rel_pt = np.array([0.0, 0.0, 0.0]) # np.array([0, 0.06, 0]) # np.zeros((3,))
        self.rot_dir = np.array([0,0,1])
        super(BaxterEEClosedDoorGraspValid, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     return baxter_sampling.resample_ee_grasp_valid(self, negated, t, plan)

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

    #@profile
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

    #@profile
    def washer_ee_check_f(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))
        return dist_val

    #@profile
    def washer_ee_check_jac(self, x, rel_pt):
        washer_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)

        robot_pos = washer_trans.dot(np.r_[rel_pt, 1])[:3]
        obj_pos = obj_trans[:3, 3]

        joint_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        washer_jac = np.array([np.cross(axis, robot_pos - x[-7:-4, 0]) for axis in axises]).T

        obj_jac = -1 * np.array([np.cross(axis, obj_pos - obj_trans[:3,3]) for axis in axises]).T
        dist_jac = np.hstack([-np.eye(3), obj_jac, np.eye(3), washer_jac, 1*joint_jac])
        return dist_jac

    def washer_ee_rot_check_f(self, x, rot_dir):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_f(obj_trans, robot_trans, self.rot_dir)
        return x[3:6] - np.array(const.RESAMPLE_CLOSED_DOOR_ROT).reshape((3,1))


    #@profile
    def washer_ee_rot_check_jac(self, x, rel_rot):
        # robot_trans, obj_trans, axises, obj_axises, arm_joints = self.washer_obj_kinematics(x)
        # return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints, self.rot_dir)
        return np.hstack([np.zeros((3,3)), np.eye(3), np.zeros((3, 7))])

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_f(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_f(x, self.rot_dir)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.washer_ee_check_jac(x, self.rel_pt), self.rot_coeff * self.washer_ee_rot_check_jac(x, self.rot_dir)])


class BaxterEEGraspValidSide(BaxterEEGraspValid):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterEEGraspValidSide, self).__init__(name, params, expected_param_types, env, debug)
        self.rot_dir = np.array([np.pi/2,0,0])
"""
    Gripper Constraints Family
"""

class BaxterCloseGripperLeft(robot_predicates.InContact):

    # BaxterCloseGripperLeft Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        super(BaxterCloseGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

class BaxterCloseGripperRight(robot_predicates.InContact):

    # BaxterCloseGripperRight Robot

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

    # BaxterBasketCloseGripper robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1], ATTRMAP[params[0]._type][3]])])
        super(BaxterCloseGrippers, self).__init__(name, params, expected_param_types, env, debug)

class BaxterOpenGrippers(BaxterCloseGrippers):

    # InContact robot

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

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "I need to think about how not to hit anything."
        else:
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

class BaxterObstructsWasher(BaxterObstructs):
    """
    This collision checks the washer as a solid cube
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.DIST_SAFE):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEFF
        self.neg_coeff = const.OBSTRUCTS_COEFF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[3], list(ATTRMAP[params[3]._type]))])
        super(BaxterObstructs, self).__init__(name, params, expected_param_types, env, debug, tol)
        self.dsafe = 1e-2 # const.DIST_SAFE
        # self.test_env = Environment()
        # self._cc = ctrajoptpy.GetCollisionChecker(self.test_env)
        # self._param_to_body = {}
        # self._param_to_body[params[0]] = OpenRAVEBody(self._env, 'baxter_obstruct', params[0].geom)
        self.true_washer_body = self._param_to_body[params[3]]
        self._param_to_body[params[3]] = [OpenRAVEBody(self._env, 'washer_obstruct', Box([.375, .375, .375])),
                                          OpenRAVEBody(self._env, 'obstruct_door', Can(.35, .05)),
                                          OpenRAVEBody(self._env, 'obstruct_handle', Sphere(.05,))]
        self._param_to_body[params[3]][0].set_pose([0,0,0])
        self._param_to_body[params[3]][1].set_pose([0,0,0])
        self._param_to_body[params[3]][2].set_pose([0,0,0])

        f = lambda x: self.coeff*self.robot_obj_collision(x)[0]
        grad = lambda x: self.coeff*self.robot_obj_collision(x)[1]

        ## so we have an expr for the negated predicate
        f_neg = lambda x: self.neg_coeff*self.robot_obj_collision(x)[0]
        grad_neg = lambda x: self.neg_coeff*self.robot_obj_collision(x)[1]

        col_expr = Expr(f, grad)
        links = len(self.robot.geom.col_links)

        val = np.zeros((len(self.col_link_pairs)*3,1))
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
        self.true_washer_body.set_dof({'door': x[-1]})
        rot = washer_rot[0]
        x_offset = np.sin(rot)*0.1
        y_offset = -np.cos(rot)*0.1
        washer_body.set_pose(washer_pos-[[x_offset],[y_offset],[0]], washer_rot)
        door_trans = self.true_washer_body.env_body.GetLink('washer_door').GetTransform()
        washer_door_pos = door_trans.dot([0, 0.025, 0, 1])[:3]
        washer_door.set_pose(washer_door_pos, [washer_rot[0]+[np.pi/2+x[-1]], np.pi/2, 0])
        handle_trans = self.true_washer_body.env_body.GetLink('washer_handle').GetTransform()
        washer_handle.set_pose(handle_trans[:3,3]+[0,0,0.04])

        # Make sure two body is in the same environment
        assert robot_body.env_body.GetEnv() == washer_body.env_body.GetEnv()
        self.set_active_dof_inds(robot_body, reset=False)
        # Setup collision checkers
        self._cc.SetContactDistance(const.MAX_CONTACT_DISTANCE)
        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_body.env_body)
        # Calculate value and jacobian
        col_val, col_jac = self._calc_grad_and_val(robot_body, washer_body, collisions)

        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_door.env_body)
        door_col_val, door_col_jac = self._calc_grad_and_val(robot_body, washer_door, collisions)

        collisions = self._cc.BodyVsBody(robot_body.env_body, washer_handle.env_body)
        handle_col_val, handle_col_jac = self._calc_grad_and_val(robot_body, washer_handle, collisions)
        # set active dof value back to its original state (For successive function call)
        self.set_active_dof_inds(robot_body, reset=True)
        # self._cache[flattened] = (col_val.copy(), col_jac.copy())
        # print "col_val", np.max(col_val)
        washer_body.set_pose([0,0,0])
        washer_door.set_pose([0,0,0])
        washer_handle.set_pose([0,0,0])
        # return col_val+door_col_val+handle_col_val, col_jac+door_col_jac+handle_col_jac
        return np.vstack([col_val, door_col_val, handle_col_val]), np.vstack([col_jac, door_col_jac, handle_col_jac])

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
        rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
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

            grad = np.zeros((1, self.attr_dim+7))
            grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)
            col_vec =  -sign*normal
            # Calculate object pose jacobian
            grad[:, self.attr_dim:self.attr_dim+3] = col_vec
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, self.attr_dim+3:self.attr_dim+6] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot, linkObj)] = [self.dsafe - distance, grad, robot.GetLink(linkRobot), robot.GetLink(linkObj)]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim+7)), None, None])
            vals.append(col_infos[0])
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape((len(greds), self.attr_dim+7))


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

    # def resample(self, negated, t, plan):
    #     if const.PRODUCTION:
    #         print "I need to think about how I'm not going to hit the washer."
    #     else:
    #         print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_washer_obstructs(self, negated, t, plan)

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

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "I need to think about how not to hit anything."
        else:
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

        rel_pt = np.zeros((3,))
        manip = robot_body.env_body.GetManipulator("right_arm")
        if self.obj.name == "basket":
            rel_pt = np.array([0,-const.BASKET_OFFSET,-0.03])
            manip = robot_body.env_body.GetManipulator("left_arm")

        robot_trans = manip.GetTransform()
        basket_pos = np.dot(robot_trans, np.r_[rel_pt, 1])[:3]
        x[-6:-3] = basket_pos.reshape(x[-6:-3].shape)

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

        l_manip = robot_body.env_body.GetManipulator("left_arm")
        l_pos = l_manip.GetTransform()[:3,3]
        x[-6:-3] = l_pos.reshape(x[-6:-3].shape)

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

    #@profile
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

class BaxterRSelfCollides(robot_predicates.RSelfCollides):

    # RCollides Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        super(BaxterRSelfCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.RCOLLIDES_DSAFE

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

class BaxterCollidesWasher(BaxterRCollides):
    """
    This collision checks the full mock-up as a set of its individual parts
    """
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -const.RCOLLIDE_COEFF
        self.neg_coeff = const.RCOLLIDE_COEFF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterRCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = 1e-2 # const.RCOLLIDES_DSAFE

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
        obj_body.set_dof({'door': x[-1]})

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
        rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
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

            grad = np.zeros((1, self.attr_dim+7))
            grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)
            col_vec =  -sign*normal
            # Calculate object pose jacobian
            grad[:, self.attr_dim:self.attr_dim+3] = col_vec
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, self.attr_dim+3:self.attr_dim+6] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot, linkObj)] = [self.dsafe - distance, grad, robot.GetLink(linkRobot), robot.GetLink(linkObj)]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim+7)), None, None])
            vals.append(col_infos[0])
            if len(col_infos[1][0]) == 6:
                col_infos[1] = np.c_[col_infos[1], [[0]]]
                import ipdb; ipdb.set_trace()
            greds.append(col_infos[1])

        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape((len(greds), self.attr_dim+7))

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

    def resample(self, negated, t, plan):
        # return None, None
        if const.PRODUCTION:
            print "I need to make sure I don't hit the washer."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_rcollides(self, negated, t, plan)

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

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach.\n"
        else:
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

    #@profile
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
            arm_inds = self.robot.geom.dof_map['rArmPose']
        else:
            arm_inds = self.robot.geom.dof_map['lArmPose']
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
        if const.PRODUCTION:
            print "Let me try a new approach.\n"
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_rrt(self, negated, t, plan, inv = True)

class BaxterEEReachableRightInv(BaxterEEReachableRight):

    # EEreachableInv Robot, StartPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_rrt(self, negated, t, plan, inv='True')

class BaxterEEReachableLeftVer(BaxterEEReachableLeft):

    # BaxterEEReachableVerLeftPos Robot, RobotPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_ver(self, negated, t, plan)

class BaxterEEReachableRightVer(BaxterEEReachableRight):

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_eereachable_ver(self, negated, t, plan)

class BaxterEEApproachLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEEApproachLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = True)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0,  0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

class BaxterEEApproachOpenDoorLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEEApproachOpenDoorLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        rel_pt = np.array([-const.APPROACH_DIST/2, -const.APPROACH_DIST*np.sqrt(3)/2, 0])
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach=True, rel_pt=rel_pt)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST*np.sqrt(3)/2, const.APPROACH_DIST/2, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST*np.sqrt(3)/2, -const.RETREAT_DIST/2, 0])

class BaxterEEApproachCloseDoorLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEEApproachCloseDoorLeft, self).__init__(name, params, expected_param_types, (-steps, 0), env, debug)

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
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

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
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

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = False)

class BaxterEERetreatOpenDoorLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEERetreatOpenDoorLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0,  0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach = False)

class BaxterEERetreatCloseDoorLeft(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "left"
        super(BaxterEERetreatCloseDoorLeft, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

    # def get_rel_pt(self, rel_step):
    #     if rel_step <= 0:
    #         return rel_step*np.array([const.APPROACH_DIST/2, const.APPROACH_DIST*np.sqrt(3)/2, 0])
    #     else:
    #         return rel_step*np.array([-const.RETREAT_DIST/2, -const.RETREAT_DIST*np.sqrt(3)/2, 0])

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST/2, const.APPROACH_DIST*np.sqrt(3)/2, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST/2, -const.RETREAT_DIST*np.sqrt(3)/2, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
            print "resample {}".format(self.get_type())
        rel_pt = np.array([-const.APPROACH_DIST/2, -const.APPROACH_DIST*np.sqrt(3)/2, 0])
        return baxter_sampling.resample_washer_ee_approach(self, negated, t, plan, approach=False, rel_pt=rel_pt)

class BaxterEERetreatRight(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, steps=const.EEREACHABLE_STEPS, env=None, debug=False):
        self.arm = "right"
        super(BaxterEERetreatRight, self).__init__(name, params, expected_param_types, (0, steps), env, debug)

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

    #@profile
    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "Let me try a new approach."
        else:
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

class BaxterBasketInGripper(BaxterInGripper):

    # BaxterBasketInGripper Robot, Basket

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.eval_dim = 15
        self.grip_offset = const.BASKET_GRIP_OFFSET
        super(BaxterBasketInGripper, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_basket_moveholding(self, negated, t, plan)

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_f(x), self.rot_coeff * self.both_arm_rot_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_jac(x), self.rot_coeff * self.both_arm_rot_check_jac(x)])

class BaxterBasketInGripperShallow(BaxterBasketInGripper):

    # BaxterBasketInGripper Robot, Basket

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterBasketInGripperShallow, self).__init__(name, params, expected_param_types, env, debug)
        self.grip_offset = const.BASKET_SHALLOW_GRIP_OFFSET

class BaxterBothEndsInGripper(BaxterInGripper):

    # BaxterBothEndsInGripper Robot, Cylinder

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.eval_dim = 6
        super(BaxterBothEndsInGripper, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, 'left')
        l_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in l_arm_inds]
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, 'right')
        r_arm_joints = [body.GetJointFromDOFIndex(ind) for ind in r_arm_inds]
        rel_pt = np.array([0,0,self.obj.geom.height/2.0+0.1])
        # rel_pt = np.array([0, 2*const.BASKET_NARROW_OFFSET,0])
        l_pos_val = self.rel_pos_error_f(obj_trans, l_ee_trans, rel_pt)
        rel_pt = np.array([0,0,-self.obj.geom.height/2.0-0.1])
        # rel_pt = np.array([0, -2*const.BASKET_NARROW_OFFSET,0])
        r_pos_val = self.rel_pos_error_f(obj_trans, r_ee_trans, rel_pt)
        return np.vstack([l_pos_val, r_pos_val])

    #@profile
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

        l_ee_trans, l_arm_inds = self.get_robot_info(robot_body, 'left')
        r_ee_trans, r_arm_inds = self.get_robot_info(robot_body, 'right')

        self.arm = "right"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6: -3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([0,0,-self.obj.geom.height/2.0-0.1])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        r_obj_pos_jac = self.rel_pos_error_jac(obj_trans, r_ee_trans, axises, arm_joints, rel_pt)

        self.arm = "left"
        obj_body = self.obj.openrave_body
        obj_body.set_pose(x[-6: -3], x[-3:])
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        rel_pt = np.array([0,0,self.obj.geom.height/2.0+0.1])
        # rel_pt = np.array([0, 0, -const.BASKET_NARROW_OFFSET])
        l_obj_pos_jac = self.rel_pos_error_jac(obj_trans, l_ee_trans, axises, arm_joints, rel_pt)

        return np.vstack([l_obj_pos_jac, r_obj_pos_jac])


    def stacked_f(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.both_arm_pos_check_jac(x)])

class BaxterWasherInGripper(BaxterInGripper):

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.eval_dim = 4
        self.arm = 'left'
        # self.rel_pt = np.array([-0.04,0.07,-0.1])
        self.rel_pt = np.array([0.0, 0.0, 0.0]) # np.array([0, 0.06, 0]) # np.zeros((3,))
        super(BaxterWasherInGripper, self).__init__(name, params, expected_param_types, env, debug)
        self.rot_coeff = 1e-2 #const.WASHER_IN_GRIPPER_ROT_COEFF

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

    #@profile
    # def resample(self, negated, t, plan):
        # print "resample {}".format(self.get_type())
        # return baxter_sampling.resample_washer_in_gripper(self, negated, t, plan)

    #@profile
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

    #@profile
    def ee_contact_check_f(self, x, rel_pt):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)

        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[rel_pt, 1])[:3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))

        return dist_val

    #@profile
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

    #@profile
    def rot_check_f(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        robot = np.array([0,0,1])
        washer_rot = [0,1,0]
        return self.rot_error_f(obj_trans, robot_trans, washer_rot, robot)

    #@profile
    def rot_check_jac(self, x):
        obj_trans, robot_trans, axises, arm_joints = self.robot_obj_kinematics(x)
        robot = np.array([0,0,1])
        washer_rot = [0,1,0]
        return self.rot_error_jac(obj_trans, robot_trans, axises, arm_joints,  washer_rot, robot)

    #@profile
    def robot_obj_kinematics(self, x):
        """
            This function is used to check whether End Effective pose's position is at robot gripper's center

            Note: Child classes need to provide set_robot_poses and get_robot_info functions.
        """
        # Getting the variables
        # robot_body = self.robot.openrave_body
        # body = robot_body.env_body
        # # Setting the poses for forward kinematics to work
        # self.set_robot_poses(x, robot_body)
        # robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        # arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        # ee_pos, ee_rot = x[-7:-4], x[-4:-1]
        # obj_trans = OpenRAVEBody.transform_from_obj_pose(ee_pos, ee_rot)
        # Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(ee_pos, ee_rot)
        # axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # # Obtain the pos and rot val and jac from 2 function calls
        # return obj_trans, robot_trans, axises, arm_joints

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

        return obj_trans, robot_trans, axises, arm_joints


    def stacked_f(self, x):
        return np.vstack([self.coeff * self.ee_contact_check_f(x, self.rel_pt), self.rot_coeff * self.rot_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([self.coeff * self.ee_contact_check_jac(x, self.rel_pt), self.rot_coeff * np.c_[self.rot_check_jac(x), np.zeros((1,))]])

class BaxterClothInGripperRight(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "right"
        self.eval_dim = 3
        super(BaxterClothInGripperRight, self).__init__(name, params, expected_param_types, env, debug)

    def resample(self, negated, t, plan):
        if const.PRODUCTION:
            print "I need to think about where I'm going to put my gripper."
        else:
            print "resample {}".format(self.get_type())
        return baxter_sampling.resample_cloth_in_gripper(self, negated, t, plan)

    def stacked_f(self, x):
        return self.coeff * self.pos_check_f(x)

    def stacked_grad(self, x):
        return self.coeff * self.pos_check_jac(x)

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
        self.obj.openrave_body.set_pose(x[-6:-3], x[-3:])

class BaxterClothInGripperLeft(BaxterInGripper):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "left"
        self.eval_dim = 3
        super(BaxterClothInGripperLeft, self).__init__(name, params, expected_param_types, env, debug)

    # def resample(self, negated, t, plan):
    #     print "resample {}".format(self.get_type())
    #     return baxter_sampling.resample_cloth_in_gripper(self, negated, t, plan)

    def stacked_f(self, x):
        return self.coeff * self.pos_check_f(x)

    def stacked_grad(self, x):
        return self.coeff * self.pos_check_jac(x)

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
        self.obj.openrave_body.set_pose(x[-6:-3], x[-3:])

class BaxterGripperAt(robot_predicates.GripperAt):

    # InGripper, Robot, EEPose

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = const.GRIPPER_AT_COEFF
        self.rot_coeff = const.GRIPPER_AT_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        super(BaxterGripperAt, self).__init__(name, params, expected_param_types, env, debug)

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
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))] # axises = [axis_z, axis_y, axis_x]
        # Obtain the pos and rot val and jac from 2 function calls
        return obj_trans, robot_trans, axises, arm_joints

    def stacked_f(self, x):
        return np.vstack([self.coeff * self.pos_check_f(x)])

    def stacked_grad(self, x):
        return np.vstack([10*self.coeff * self.pos_check_jac(x)])

    def resample(self, negated, t, plan):
        return baxter_sampling.resample_gripper_at(self, negated, t, plan)

class BaxterGripperAtLeft(BaxterGripperAt):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "left"
        self.eval_dim = 3
        super(BaxterGripperAtLeft, self).__init__(name, params, expected_param_types, env, debug)

class BaxterGripperAtRight(BaxterGripperAt):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.arm = "right"
        self.eval_dim = 3
        super(BaxterGripperAtRight, self).__init__(name, params, expected_param_types, env, debug)

class BaxterPushWasher(robot_predicates.IsPushing):

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.eval_dim = 4
        self.arm = 'left'
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = 1#const.IN_GRIPPER_COEFF
        self.rot_coeff = 1#const.IN_GRIPPER_ROT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.rel_pt = np.array([-.2,-0.07,0])
        super(BaxterPushWasher, self).__init__(name, params, expected_param_types, env, debug)

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

    def set_washer_poses(self, x, washer_body):
        pose, rotation = x[-7:-4], x[-4:-1]
        door = x[-1]
        washer_body.set_pose(pose, rotation)
        washer_body.set_dof({"door": door})

    def get_washer_info(self, washer_body):
        tool_link = washer_body.env_body.GetLink("washer_door")
        washer_trans = tool_link.GetTransform()
        washer_inds = [0]

        return washer_trans, washer_inds

    #@profile
    def robot_robot_kinematics(self, x):
        robot_body = self.robot1.openrave_body
        body = robot_body.env_body
        washer_body = self.robot2.openrave_body
        washer = washer_body.env_body
        self.set_robot_poses(x, robot_body)
        robot_trans, arm_inds = self.get_robot_info(robot_body, self.arm)
        arm_joints = [body.GetJointFromDOFIndex(ind) for ind in arm_inds]

        self.set_washer_poses(x, washer_body)
        washer_trans, washer_arm_inds = self.get_washer_info(washer_body)
        washer_joints = [washer.GetJointFromDOFIndex(ind) for ind in washer_arm_inds]

        pos, rot = x[-7:-4], x[-4:-1]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(pos, rot)
        axises = [[0,0,1], np.dot(Rz, [0,1,0]), np.dot(Rz, np.dot(Ry, [1,0,0]))]

        return robot_trans, washer_trans, arm_joints, washer_joints, axises

    #@profile
    def ee_contact_check_f(self, x, rel_pt):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)
        robot_pos = robot_trans[:3, 3]
        obj_pos = np.dot(obj_trans, np.r_[rel_pt, 1])[:3]
        dist_val = (robot_pos - obj_pos).reshape((3,1))

        return dist_val

    #@profile
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

    #@profile
    def ee_rot_check_f(self, x, local_dir=[1,0,0]):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])
        return rot_val

    #@profile
    def ee_rot_check_jac(self, x, local_dir=[1,0,0]):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))
        # computing robot's jacobian
        arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in arm_joints]).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = sign*np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir))).reshape((1,1))
        # computing object's jacobian
        obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
        obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))
        # Create final 1x23 jacobian matrix
        rot_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm)
        return rot_jac

    def stacked_f(self, x):
        rel_pt = self.rel_pt
        rot_dir = np.array([0,np.pi/2,0])
        return np.vstack([self.coeff * self.ee_contact_check_f(x, rel_pt), self.rot_coeff * self.ee_rot_check_f(x)])

    def stacked_grad(self, x):
        rel_pt = self.rel_pt
        return np.vstack([self.coeff * self.ee_contact_check_jac(x, rel_pt), self.rot_coeff * np.c_[self.ee_rot_check_jac(x), np.zeros((1,))]])

class BaxterPushInsideWasher(BaxterPushWasher):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterPushInsideWasher, self).__init__(name, params, expected_param_types, env, debug)
        self.rel_pt = np.array([-.1,-0.075,0])

class BaxterPushOutsideWasher(BaxterPushWasher):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterPushOutsideWasher, self).__init__(name, params, expected_param_types, env, debug)
        self.rel_pt = np.array([-0.2,0.12,0.2])

class BaxterPushOutsideCloseWasher(BaxterPushWasher):
    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterPushOutsideCloseWasher, self).__init__(name, params, expected_param_types, env, debug)
        self.rel_pt = np.array([-0.2,0.145,0.0])

class BaxterPushHandle(BaxterPushWasher):

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        super(BaxterPushHandle, self).__init__(name, params, expected_param_types, env, debug)
        self.rel_pt = np.array([-0.06, 0.0, 0.04])

    def get_washer_info(self, washer_body):
        tool_link = washer_body.env_body.GetLink("washer_handle")
        washer_trans = tool_link.GetTransform()
        washer_inds = [0]

        return washer_trans, washer_inds

    def ee_rot_check_f(self, x, local_dir=[1,0,0]):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])
        return rot_val

    def ee_rot_check_jac(self, x, local_dir=[1,0,0]):
        robot_trans, obj_trans, arm_joints, obj_joints, axises = self.robot_robot_kinematics(x)
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))
        # computing robot's jacobian
        arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in arm_joints]).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = sign*np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir))).reshape((1,1))
        # computing object's jacobian
        obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
        obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))
        # Create final 1x23 jacobian matrix
        rot_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac, self.arm)
        return rot_jac


"""
    Basket Constraint Family
"""

class BaxterBasketLevel(robot_predicates.BasketLevel):
    # BaxterBasketLevel BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]])])
        self.attr_dim = 3
        self.basket = params[0]
        super(BaxterBasketLevel, self).__init__(name, params, expected_param_types, env, debug)

class BaxterClothTargetInWasher(ExprPredicate):
    # BaxterClothTargetInWasher ClothTarget WasherTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 6
        self.cloth_target = params[0]
        self.washer_pose = params[1]
        A = np.c_[np.eye(self.attr_dim/2), -np.eye(self.attr_dim/2)]
        b = np.zeros((self.attr_dim/2,1))
        val = np.array([[const.WASHER_DEPTH_OFFSET/2], [np.sqrt(3)*const.WASHER_DEPTH_OFFSET/2], [0]])
        pos_expr = AffExpr(A, b)
        e = EqExpr(pos_expr, val)
        super(BaxterClothTargetInWasher, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)

class BaxterClothTargetInBasket(ExprPredicate):
    # BaxterClothTargetInBasket ClothTarget BasketTarget
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 6
        self.cloth_target = params[0]
        self.basket_target = params[1]

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b = np.zeros((6,1))

        val = np.array([[.09], [.09], [-.04], [.09], [.09], [.04]])
        pos_expr = AffExpr(A, b)
        e = LEqExpr(pos_expr, val)
        super(BaxterClothTargetInBasket, self).__init__(name, e, self.attr_inds, params, expected_param_types, priority = -2)

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
        self.grip_offset = const.BASKET_GRIP_OFFSET
        super(BaxterGrippersLevel, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    #@profile
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

    #@profile
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

class BaxterGrippersWithinYDist(robot_predicates.GrippersLevel):
    # BaxterGrippersLevel Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 1
        self.opt_coeff = 1
        self.eval_f = lambda x: self.both_arm_pos_check(x)[0]
        self.eval_grad = lambda x: self.both_arm_pos_check(x)[1]
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        self.eval_dim = 6
        self.grip_offset = const.BASKET_GRIP_OFFSET
        super(BaxterGrippersLevel, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    #@profile
    def both_arm_pos_check(self, x):
        """
            This function is used to check whether:
                both grippers are at a certain xy distance

            Note: Child class that uses this function needs to provide set_robot_poses and get_robot_info functions
        """
        # Obtain openrave body
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x, robot_body)

        dist = self.params[-1].value[0,0]

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

    #@profile
    def pos_error(self, robot_arm_trans, robot_aux_arm_trans, axises, arm_joints, aux_joints, rel_pt, arm):
        """
            This function calculates the value and the jacobian of the displacement between the the gripper and the inactive gripper on the XY plane

            robot_trans: active robot gripper's rave_body transformation
            robot_aux_arm_trans: inactive gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        dist = self.params[-1].value[0,0]

        axis = np.array([0, 1, 1]).reshape((3, 1))
        gp = rel_pt
        robot_pos = robot_arm_trans[:3, 3]
        robot_aux_pos = robot_aux_arm_trans[:3, 3]
        dist_val = (robot_pos.flatten() - robot_aux_pos.flatten()).reshape((3, 1)) * axis
        dist_val[1] -= dist
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy() * axis
        # Calculate jacobian for the robot base
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos - np.zeros((3,))).reshape((3, 1)) * axis
        # Calculate object jacobian
        aux_jac = -1*np.array([np.cross(joint.GetAxis(), robot_aux_pos.flatten() - joint.GetAnchor()) for joint in aux_joints]).T.copy() * axis
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

    #@profile
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

    #@profile
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
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]), (params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        super(BaxterObjRelPoseConstant, self).__init__(name, params, expected_param_types, env, debug)

class BaxterGrippersDownRot(robot_predicates.GrippersLevel):
    # BaxterGrippersDownRot Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 0.1
        self.opt_coeff = 0.1
        self.eval_f = lambda x: self.both_arm_rot_check(x)
        self.eval_grad = lambda x: self.both_arm_rot_jac(x)
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])[:-1])])
        self.eval_dim = 2
        super(BaxterGrippersDownRot, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    def both_arm_rot_check(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")

        local_dir = [1, 0, 0]
        world_dir = robot_left_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        left_rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])

        world_dir = robot_right_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        right_rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])
        return np.r_[left_rot_val, right_rot_val]

    def both_arm_rot_jac(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        right_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in right_arm_inds]
        l_tool_link = robot_body.env_body.GetLink("left_gripper")
        l_manip_trans = l_tool_link.GetTransform()

        r_tool_link = robot_body.env_body.GetLink("right_gripper")
        r_manip_trans = r_tool_link.GetTransform()

        local_dir = [1, 0, 0]
        world_dir = robot_left_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))

        left_arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in left_arm_joints]).T.copy()
        left_arm_jac = left_arm_jac.reshape((1, len(left_arm_joints)))

        local_dir = [1, 0, 0]
        world_dir = robot_right_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))

        right_arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in right_arm_joints]).T.copy()
        right_arm_jac = right_arm_jac.reshape((1, len(right_arm_joints)))

        jac = np.zeros((2, 17))
        jac[0,:7] = left_arm_jac
        jac[1,8:15] = right_arm_jac
        return jac

class BaxterLeftGripperDownRot(robot_predicates.GrippersLevel):
    # BaxterLeftGripperDownRot Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 0.1
        self.opt_coeff = 0.1
        self.eval_f = lambda x: self.left_arm_rot_check(x)
        self.eval_grad = lambda x: self.left_arm_rot_jac(x)
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])[:-1])])
        self.eval_dim = 2
        super(BaxterGrippersDownRot, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    def left_arm_rot_check(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        local_dir = [1, 0, 0]
        world_dir = robot_left_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        left_rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])

        return left_rot_val

    def left_arm_rot_jac(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        left_arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in left_arm_inds]
        l_tool_link = robot_body.env_body.GetLink("left_gripper")
        l_manip_trans = l_tool_link.GetTransform()

        local_dir = [1, 0, 0]
        world_dir = robot_left_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))

        left_arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in left_arm_joints]).T.copy()
        left_arm_jac = left_arm_jac.reshape((1, len(left_arm_joints)))

        jac = np.zeros((1, 17))
        jac[0,:7] = left_arm_jac
        return jac

class BaxterRightGripperDownRot(robot_predicates.GrippersLevel):
    # BaxterRightGripperDownRot Robot
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.coeff = 0.1
        self.opt_coeff = 0.1
        self.eval_f = lambda x: self.left_arm_rot_check(x)
        self.eval_grad = lambda x: self.left_arm_rot_jac(x)
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])[:-1])])
        self.eval_dim = 2
        super(BaxterGrippersDownRot, self).__init__(name, params, expected_param_types, env, debug)

    #@profile
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

    def right_arm_rot_check(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_trans, arm_inds = self.get_robot_info(robot_body, "right")
        local_dir = [1, 0, 0]
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        rot_val = np.array([[np.abs(np.dot([0,0,-1], world_dir)) - 1]])

        return rot_val

    def left_arm_rot_jac(self, x):
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        self.set_robot_poses(x, robot_body)

        robot_trans, arm_inds = self.get_robot_info(robot_body, "right")
        arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]
        tool_link = robot_body.env_body.GetLink("right_gripper")
        manip_trans = tool_link.GetTransform()

        local_dir = [1, 0, 0]
        world_dir = robot_trans[:3,:3].dot(local_dir)
        world_dir = world_dir/np.linalg.norm(world_dir)
        obj_dir = [0,0,-1]
        sign = np.sign(np.dot(obj_dir, world_dir))

        arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign * world_dir)) for joint in arm_joints]).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))

        jac = np.zeros((1, 17))
        jac[8:15] = arm_jac
        return jac
