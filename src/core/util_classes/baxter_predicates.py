from core.util_classes import robot_predicates
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
import numpy as np

BASE_DIM = 1
JOINT_DIM = 16
ROBOT_ATTR_DIM = 17

BASE_MOVE = 1
JOINT_MOVE_FACTOR = 10
TWOARMDIM = 16

# Attribute map used in baxter domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {"Robot": (("lArmPose", np.array(range(7), dtype=np.int)),
                     ("lGripper", np.array([0], dtype=np.int)),
                     ("rArmPose", np.array(range(7), dtype=np.int)),
                     ("rGripper", np.array([0], dtype=np.int)),
                     ("pose", np.array([0], dtype=np.int))),
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
                        ("rotation", np.array([0,1,2], dtype=np.int)))
          }

class BaxterAt(robot_predicates.At):
    pass

class BaxterRobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 17
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterRobotAt, self).__init__(name, params, expected_param_types, env)

class BaxterIsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        super(BaxterIsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds(list(range(2,2+JOINT_DIM)))
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
        robot_body._set_active_dof_inds(range(18))

        # Setting attributes for testing
        self.base_step = BASE_MOVE*np.ones((BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class BaxterWithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        super(BaxterWithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds(list(range(2,2+JOINT_DIM)))
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
        robot_body._set_active_dof_inds(range(18))

        joint_move = (active_ub-active_lb)/JOINT_MOVE_FACTOR
        self.base_step = BASE_MOVE*np.ones((BASE_DIM,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val


class BaxterStationary(robot_predicates.Stationary):
    pass

class BaxterStationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-1]])])
        self.attr_dim = BASEDIM
        super(PR2StationaryBase, self).__init__(self, name, params, expected_param_types, env)

class BaxterStationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        self.attr_dim = TWOARMDIM
        super(PR2StationaryArms, self).__init__(self, name, params, expected_param_types, env)

class BaxterStationaryW(robot_predicates.StationaryW):
    pass

class BaxterStationaryNEq(robot_predicates.StationaryNEq):
    pass


