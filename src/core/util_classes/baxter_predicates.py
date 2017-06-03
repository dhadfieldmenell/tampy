from core.util_classes import robot_predicates
from core.util_classes.common_predicates import ExprPredicate
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.baxter_sampling import resample_obstructs, resample_eereachable_rrt, resample_basket_eereachable_rrt, resample_rcollides, resample_pred, resample_basket_obstructs
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
from openravepy import DOFAffine, quatRotateDirection, matrixFromQuat
import numpy as np
import core.util_classes.baxter_constants as const
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
                        ("rotation", np.array([0,1,2], dtype=np.int))),
           "Basket": (("pose", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "BasketTarget": (("value", np.array([0,1,2], dtype=np.int)),
                            ("rotation", np.array([0,1,2], dtype=np.int))),
           "EEVel": (("value", np.array([0, 1, 2], dtype=np.int)),
                     ("rotation", np.array([0, 1, 2], dtype=np.int)))
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

class BaxterWithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
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

class BaxterStationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-1]])])
        self.attr_dim = const.BASE_DIM
        super(BaxterStationaryBase, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        self.attr_dim = const.TWOARMDIM
        super(BaxterStationaryArms, self).__init__(name, params, expected_param_types, env)

class BaxterStationaryW(robot_predicates.StationaryW):
    pass

class BaxterStationaryNEq(robot_predicates.StationaryNEq):
    pass

class BaxterGraspValid(robot_predicates.GraspValid):
    pass

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

class BaxterInContact(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][3]])])
        super(BaxterInContact, self).__init__(name, params, expected_param_types, env, debug)

class BaxterBasketInContact(robot_predicates.InContacts):

    # InContact robot EEPose EEPose BasketTarget

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1], ATTRMAP[params[0]._type][3]])])
        super(BaxterBasketInContact, self).__init__(name, params, expected_param_types, env, debug)

class BaxterBasketOpenGripper(BaxterBasketInContact):

    # InContact robot EEPose EEPose BasketTarget

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterBasketOpenGripper, self).__init__(name, params, expected_param_types, env, debug)
        self.expr = self.neg_expr

class BaxterObstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.DIST_SAFE):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -const.OBSTRUCTS_COEEF
        self.neg_coeff = const.OBSTRUCTS_COEEF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type]))])
        super(BaxterObstructs, self).__init__(name, params, expected_param_types, env, debug, tol)
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        # return resample_pred(self, negated, t, plan)
        return resample_basket_obstructs(self, negated, t, plan)
        # return None, None

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
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

class BaxterObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type])),
                                 (params[4], list(ATTRMAP[params[4]._type]))])
        self.OBSTRUCTS_OPT_COEFF = const.OBSTRUCTS_OPT_COEFF
        super(BaxterObstructsHolding, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        return resample_basket_obstructs(self, negated, t, plan)
        # return resample_pred(self, negated, t, plan)
        # return None, None

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
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18)), DOFAffine.RotationAxis, [0,0,1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

class BaxterCollides(robot_predicates.Collides):

    # Collides Basket Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.COLLIDES_DSAFE

class BaxterRCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.coeff = -1
        self.neg_coeff = 1
        self.opt_coeff = const.RCOLLIDES_OPT_COEFF
        super(BaxterRCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.RCOLLIDES_DSAFE

    def resample(self, negated, t, plan):
        return resample_pred(self, negated, t, plan)
        # return resample_obstructs(self, negated, t, plan)

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
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18)), DOFAffine.RotationAxis, [0,0,1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")


class BaxterEEReachable(robot_predicates.EEReachable):

    # EEreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[2], list(ATTRMAP[params[2]._type]))])
        self.attr_dim = 23
        self.ee_resample = False
        super(BaxterEEReachable, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return resample_eereachable_rrt(self, negated, t, plan)

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

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([0, 0, const.RETREAT_DIST])

    def stacked_f(self, x):
        i = 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(self.ee_pose_check_rel_obj(x[i:i+self.attr_dim], rel_pt)[0])
            i += self.attr_dim
        return np.vstack(tuple(f_res))

    def stacked_grad(self, x):
        f_grad = []
        start, end = self.active_range
        t = (2*self._steps+1)
        k = 3

        grad = np.zeros((k*t, self.attr_dim*t))
        i = 0
        j = 0
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            grad[j:j+k, i:i+self.attr_dim] = self.ee_pose_check_rel_obj(x[i:i+self.attr_dim], rel_pt)[1]
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
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3,1))
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos - np.zeros((3,))).reshape((3,1))
        # Calculate object jacobian
        # obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = -1*np.array([np.cross(axis, obj_pos - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = self.get_arm_jac(arm_jac, base_jac, obj_jac)

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
            obj_dir = np.dot(obj_trans[:3,:3], local_dir)
            world_dir = robot_trans[:3,:3].dot(local_dir)
            rot_vals.append(np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]]))
            sign = np.sign(np.dot(obj_dir, world_dir))
            # computing robot's jacobian
            arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign*world_dir)) for joint in arm_joints]).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir)))
            base_jac = base_jac.reshape((1,1))
            # computing object's jacobian
            obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
            obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))
            # Create final 1x26 jacobian matrix
            rot_jacs.append(self.get_arm_jac(arm_jac, base_jac, obj_jac))
            # rot_jacs.append(np.hstack((np.zeros((1, 8)), arm_jac, np.zeros((1,1)), base_jac, obj_jac)))

        rot_val = np.vstack(rot_vals)
        rot_jac = np.vstack(rot_jacs)

        return (rot_val, rot_jac)

    def arm_jac_cancatenation(self, arm_jac, base_jac, obj_jac, arm = "right"):
        dim = arm_jac.shape[0]
        if arm == "left":
            jacobian = np.hstack((arm_jac, np.zeros((dim, 1)), np.zeros((dim, 8)), base_jac, obj_jac))
        elif arm == "right":
            jacobian = np.hstack((np.zeros((dim, 8)), arm_jac, np.zeros((dim, 1)), base_jac, obj_jac))
        else:
            PredicateException("Invalid Arm Specified")

        return jacobian

class BaxterEEReachablePos(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.coeff = 1
        self.opt_coeff = 1
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        super(BaxterEEReachablePos, self).__init__(name, params, expected_param_types, env, debug, steps)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

class BaxterEEReachableRot(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=0):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(BaxterEEReachableRot, self).__init__(name, params, expected_param_types, env, debug, steps)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

class BaxterEEReachableInvPos(BaxterEEReachablePos):

    # EEreachableInv Robot, StartPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return resample_eereachable_rrt(self, negated, t, plan, inv= True)

class BaxterEEReachableInvRot(BaxterEEReachableRot):

    # EEreachableInv Robot, StartPose, EEPose

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([0, 0, -const.RETREAT_DIST])
        else:
            return rel_step*np.array([-const.APPROACH_DIST, 0, 0])

    def resample(self, negated, t, plan):
        return resample_eereachable_rrt(self, negated, t, plan, inv='True')

class BaxterEEReachableVerLeftPos(BaxterEEReachable):

    # BaxterEEReachableVerLeftPos Robot, RobotPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.coeff = 1
        self.opt_coeff = 1
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        super(BaxterEEReachableVerLeftPos, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return resample_basket_eereachable_rrt(self, negated, t, plan)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "left")

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

class BaxterEEReachableVerRightPos(BaxterEEReachable):

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.coeff = 1
        self.opt_coeff = 1
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        super(BaxterEEReachableVerRightPos, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return resample_basket_eereachable_rrt(self, negated, t, plan)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([-const.RETREAT_DIST, 0, 0])

class BaxterEEReachableVerLeftRot(BaxterEEReachable):
    # axterEEReachableVerLeftRot, Robot, RobotPose, EEPose
    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=0):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(BaxterEEReachableVerLeftRot, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return resample_basket_eereachable_rrt(self, negated, t, plan)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "left")

class BaxterEEReachableVerRightRot(BaxterEEReachable):
    # BaxterEEReachableVerRightRot, Robot, RobotPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=0):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(BaxterEEReachableVerRightRot, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return resample_basket_eereachable_rrt(self, negated, t, plan)

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

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

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
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [0.317], [0]])
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
        b, val = np.zeros((self.attr_dim,1)), np.array([[0], [-0.317], [0]])
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

class BaxterInGripper(robot_predicates.InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
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

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints, rel_pt = [0,0,0], arm="right"):
        """
            This function calculates the value and the jacobian of the displacement between center of gripper and center of object

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
        base_jac = np.cross(np.array([0, 0, 1]), robot_pos - np.zeros((3,))).reshape((3,1))
        # Calculate object jacobian
        obj_jac = -1*np.array([np.cross(axis, obj_pos - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, arm)
        return dist_val, dist_jac

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints, arm="right"):
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
        rot_val = np.array([[np.abs(np.dot(obj_dir, world_dir)) - 1]])
        sign = np.sign(np.dot(obj_dir, world_dir))
        # computing robot's jacobian
        arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign*world_dir)) for joint in arm_joints]).T.copy()
        arm_jac = arm_jac.reshape((1, len(arm_joints)))
        base_jac = sign*np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir))).reshape((1,1))
        # computing object's jacobian
        obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
        obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))
        # Create final 1x23 jacobian matrix
        rot_jac = self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, arm)
        return (rot_val, rot_jac)

    def arm_jac_cancatenation(self, arm_jac, base_jac, obj_jac, arm = "right"):
        dim = arm_jac.shape[0]
        if arm == "left":
            jacobian = np.hstack((arm_jac, np.zeros((dim, 1)), np.zeros((dim, 8)), base_jac, obj_jac))
        elif arm == "right":
            jacobian = np.hstack((np.zeros((dim, 8)), arm_jac, np.zeros((dim, 1)), base_jac, obj_jac))
        else:
            PredicateException("Invalid Arm Specified")

        return jacobian

class BaxterInGripperPos(BaxterInGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.eval_dim = 3
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check(x)[0]
        self.eval_grad = lambda x: self.pos_check(x)[1]
        super(BaxterInGripperPos, self).__init__(name, params, expected_param_types, env, debug)

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

class BaxterInGripperRot(BaxterInGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.eval_dim = 3
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check(x)[0]
        self.eval_grad = lambda x: self.rot_check(x)[1]
        super(BaxterInGripperRot, self).__init__(name, params, expected_param_types, env, debug)

    def get_arm_jac(self, arm_jac, base_jac, obj_jac):
        return self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, "right")

class BaxterBasketInGripperPos(BaxterInGripper):
    # BaxterBasketInGripperPos Robot, Basket

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.eval_dim = 6
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.both_arm_pos_check(x)[0]
        self.eval_grad = lambda x: self.both_arm_pos_check(x)[1]
        super(BaxterBasketInGripperPos, self).__init__(name, params, expected_param_types, env, debug)

class BaxterBasketInGripperRot(BaxterInGripper):
    # BaxterBasketInGripperRot Robot, Basket

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.eval_dim = 6
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.both_arm_rot_check(x)[0]
        self.eval_grad = lambda x: self.both_arm_rot_check(x)[1]
        super(BaxterBasketInGripperRot, self).__init__(name, params, expected_param_types, env, debug)

    def rot_error(self, obj_trans, robot_trans, axises, arm_joints, arm="right"):
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
        expected = np.array([0, 0, 1])
        for local_dir in np.eye(3):
            obj_dir = np.dot(obj_trans[:3,:3], local_dir)
            world_dir = robot_trans[:3,:3].dot(local_dir)
            obj_dir = obj_dir/np.linalg.norm(obj_dir)
            world_dir = world_dir/np.linalg.norm(world_dir)
            rot_vals.append(np.array([[np.abs(np.dot(obj_dir, world_dir)) - expected.dot(local_dir)]]))
            sign = np.sign(np.dot(obj_dir, world_dir))
            # computing robot's jacobian
            arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), sign*world_dir)) for joint in arm_joints]).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            base_jac = sign*np.array(np.dot(obj_dir, np.cross([0,0,1], world_dir))).reshape((1,1))
            # computing object's jacobian
            obj_jac = np.array([np.dot(world_dir, np.cross(axis, obj_dir)) for axis in axises])
            obj_jac = sign*np.r_[[0,0,0], obj_jac].reshape((1, 6))
            # Create final 1x23 jacobian matrix
            rot_jacs.append(self.arm_jac_cancatenation(arm_jac, base_jac, obj_jac, arm))

        rot_val = np.vstack(rot_vals)
        rot_jac = np.vstack(rot_jacs)
        return (rot_val, rot_jac)


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


class BaxterUpperVelocity(robot_predicates.Velocity):
    # BaxterVelocity Robot EEVel

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list((ATTRMAP[params[0]._type]))), (params[1], [(ATTRMAP[params[1]._type][0])])])
        self.dof_cache = None
        self.coeff = 1
        self.eval_f = lambda x: self.vel_check(x)[0]
        self.eval_grad = lambda x: self.vel_check(x)[1]
        self.eval_dim = 12

        super(BaxterUpperVelocity, self).__init__(name, params, expected_param_types, env, debug)
        self.spacial_anchor = False

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
            x = lArmPose(t), lGripper(t), rArmPose(t), rGripper(t), pose(t), EEvel.value(t),
                lArmPose(t+1), lGripper(t+1), rArmPose(t+1), rGripper(t+1), pose(t+1), EEvel.value(t+1)
                dim (40, 1)
        """

        jac = np.zeros((12, 40))
        robot_body = self._param_to_body[self.params[self.ind0]]
        robot = robot_body.env_body
        # Set poses and Get transforms
        self.set_robot_poses(x[0:17], robot_body)
        robot_left_trans, left_arm_inds = self.get_robot_info(robot_body, "left")
        robot_right_trans, right_arm_inds = self.get_robot_info(robot_body, "right")
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

class BaxterLowerVelocity(BaxterUpperVelocity):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        super(BaxterLowerVelocity, self).__init__(name, params, expected_param_types, env, debug)
        self.coeff = -1
