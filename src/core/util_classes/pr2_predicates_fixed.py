from core.util_classes.common_predicates import ExprPredicate
from core.util_classes import robot_predicates
from core.util_classes.viewer import OpenRAVEViewer
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.robots import PR2
from core.util_classes.sampling import get_col_free_base_pose_around_target, \
    get_col_free_torso_arm_pose, get_random_theta
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
import numpy as np
import ctrajoptpy
import time
import openravepy

"""
This file Defines specific PR2 related predicates
"""


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


# Dimensional Constants
BASE_DIM = 3e0
JOINT_DIM = 1.7e1
ROBOT_ATTR_DIM = 2e1
# Movement Constraints Constants
BASE_MOVE = 1e0
JOINT_MOVE_FACTOR = 1e1

TWOARMDIM = 1.6e1
# Attributes used in pr2 domain. (Tuple to avoid changes to the attr_inds)
PR2_ATTRMAP = {"Robot": (("backHeight", np.array([0], dtype=np.int)),
                         ("lArmPose", np.array(range(7), dtype=np.int)),
                         ("lGripper", np.array([0], dtype=np.int)),
                         ("rArmPose", np.array(range(7), dtype=np.int)),
                         ("rGripper", np.array([0], dtype=np.int)),
                         ("pose", np.array([0,1,2], dtype=np.int))),
                "RobotPose": (("backHeight", np.array([0], dtype=np.int)),
                              ("lArmPose", np.array(range(7), dtype=np.int)),
                              ("lGripper", np.array([0], dtype=np.int)),
                              ("rArmPose", np.array(range(7), dtype=np.int)),
                              ("rGripper", np.array([0], dtype=np.int)),
                              ("value", np.array([0,1,2], dtype=np.int))),
                "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                        ("rotation", np.array([0,1,2], dtype=np.int))),
                "EEPose": (("pose", np.array([0,1,2], dtype=np.int)),
                        ("rotation", np.array([0,1,2], dtype=np.int)))
              }

class PR2RobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 20
        self.attr_inds = OrderedDict([(params[0], list(PR2_ATTRMAP[params[0]._type])),
                                 (params[1], list(PR2_ATTRMAP[params[1]._type]))])
        super(PR2RobotAt, self).__init__(name, params, expected_param_types, env)

class PR2IsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(PR2_ATTRMAP[params[0]._type]))])
        super(PR2IsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((JOINT_DIM,1))
        joint_move = (active_ub-active_lb)/JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/JOINT_MOVE_FACTOR
        val = np.vstack((joint_move, BASE_MOVE*np.ones((BASE_DIM, 1)), joint_move, BASE_MOVE*np.ones((BASE_DIM, 1))))
        A = np.eye(2*ROBOT_ATTR_DIM) - np.eye(2*ROBOT_ATTR_DIM, k=ROBOT_ATTR_DIM) - np.eye(2*ROBOT_ATTR_DIM, k=-ROBOT_ATTR_DIM)
        b = np.zeros((2*ROBOT_ATTR_DIM,1))
        robot_body._set_active_dof_inds(range(39))

        # Setting attributes for testing
        self.base_step = BASE_MOVE*np.ones((BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class PR2WithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(PR2_ATTRMAP[params[0]._type][:-1]))])
        super(PR2WithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((JOINT_DIM,1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(JOINT_DIM)
        A_up_limit = np.eye(JOINT_DIM)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2*JOINT_DIM,1))
        robot_body._set_active_dof_inds(range(39))

        self.base_step = BASE_MOVE*np.ones((3,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class PR2StationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(self.robot, [PR2_ATTRMAP[self.robot._type][-1]])])
        super(PR2StationaryBase, self).__init__(self, name, params, expected_param_types, env, BASEDIM)

class PR2StationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type][1:-1]))])
        super(PR2StationaryArms, self).__init__(self, name, params, expected_param_types, env, TWOARMDIM)

class PR2InContact(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(self.robot, [PR2_ATTRMAP[self.robot._type][4]])])
        super(PR2InContact, self).__init__(name, e, attr_inds, params, expected_param_types)




class PR2InGripper(robot_predicates.InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type])),
                                 (self.can, list(PR2_ATTRMAP[self.can._type]))])
        super(PR2InGripper, self).__init__(self, name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot.GetManipulator('rightarm').GetArmIndices()
        return robot_trans, arm_inds


class PR2InGripperPos(PR2InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.IN_GRIPPER_COEFF = IN_GRIPPER_COEFF
        self.INGRIPPER_OPT_COEFF = INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check[0]
        self.eval_grad = lambda x: self.pos_check[1]
        super(PR2InGripperPos, self).__init__(self, name, params, expected_param_types, env, debug)

class PR2InGripperRot(PR2InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.IN_GRIPPER_COEFF = IN_GRIPPER_COEFF
        self.INGRIPPER_OPT_COEFF = INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check[0]
        self.eval_grad = lambda x: self.rot_check[1]
        super(PR2InGripperRot, self).__init__(self, name, params, expected_param_types, env, debug)

class PR2EEReachable(robot_predicates.EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type])),
                                 (self.ee_pose, list(PR2_ATTRMAP[self.ee_pose._type]))])
        super(PR2EEReachable, self).__init__(name, params, expected_param_types, env, debug, steps)

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

class PR2EEReachablePos(PR2EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.EEREACHABLE_COEFF = 1
        self.EEREACHABLE_OPT_COEFF = 1
        self.eval_f = stacked_f
        self.eval_grad = stacked_grad
        self._dim = 26
        self.EEREACHABLE_OPT = EEREACHABLE_OPT_COEFF
        super(PR2EEReachablePos, self).__init__(name, params, expected_param_types, env, debug, steps)


class PR2EEReachableRot(PR2EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.EEREACHABLE_COEFF = EEREACHABLE_COEFF
        self.EEREACHABLE_OPT_COEFF = EEREACHABLE_ROT_OPT_COEFF
        self.check_f = lambda x: ee_rot_check[0]
        self.check_grad = lambda x: ee_rot_check[1]
        super(PR2EEReachableRot, self).__init__(name, params, expected_param_types, env, debug, steps)

class PR2Obstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=COLLISION_TOL):
        self.attr_dim = 20
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type])),
                                 (self.can, list(PR2_ATTRMAP[self.can._type]))])
        super(PR2Obstructs, self).__init__(name, params, expected_param_types, env, debug, tol)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)


class PR2ObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type])),
                                 (self.obstruct, list(PR2_ATTRMAP[self.obstruct._type])),
                                 (self.held, list(PR2_ATTRMAP[self.held._type]))])
        super(PR2ObstructsHolding, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        robot_body.set_dof(back_height, l_arm_pose, l_gripper, r_arm_pose, r_gripper)

class PR2Collides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.attr_inds = OrderedDict([(self.robot, list(PR2_ATTRMAP[self.robot._type])),
                                 (self.obstacle, list(PR2_ATTRMAP[self.obstacle._type]))])
        super(PR2Collides, self).__init__(name, params, expected_param_types, env, debug)
