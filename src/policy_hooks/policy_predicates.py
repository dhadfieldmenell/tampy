from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.common_constants as const
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

import numpy as np


class BaxterPolicyPredicate(ExprPredicate):
    def __init__(self, name, params, state_inds, action_inds, policy_func, dX, dU, coeff):
        self.handle = []
        self.policy_func = policy_func
        self.state_inds = state_inds
        self.action_inds = action_inds
        self.dX = dX
        self.dU
        self.x = np.zeros((self.dX))
        self.coeff = coeff

        self.robot = None
        for param in params:
            if param._type == 'Robot':
                self.robot = param
                break
        self.attr_inds = OrderedDict([])
        expected_param_types = [param.get_type() for param in params]

        f = lambda x: self.coeff*self.error_f(x)
        grad = lambda x: self.coeff*self.error_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.dU,1))
        e = EqExpr(pos_expr, val)

        super(PolicyPredicate, self).__init__(name, e, self.attr_inds, params, expected_param_types, tol=const.POSE_TOL, active_range=(0,0), priority=4)


    def replace_policy_func(self, new_func):
        self.policy_func = new_func


    def get_param_vector(self, t):
        for p in self.params:
            for attr in const.ATTR_MAP[p._type]:
                if (p.name, attr) in self.state_inds:
                    self.x[self.state_inds[p.name, atrr]] = getattr(p, attr)[:, t]
        return self.x.reshape((self.dX, 1))


    def error_f(self, x):
        self.robot.openrave_body.set_dof({'lArmPose': x[self.state_inds[(self.robot.name, 'lArmPose')]],
                                          'lGripper': x[self.state_inds[(self.robot.name, 'lGripper')]],
                                          'rArmPose': x[self.state_inds[(self.robot.name, 'rArmPose')]],
                                          'rGripper': x[self.state_inds[(self.robot.name, 'rGripper')]]})

        robot_trans = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()
        pos = robot_trans[:3, 3]
        policy_ee = self.policy_func(x)

        dist_val = (pos - policy_ee[self.action_inds[('baxter', 'ee_left_pos')]]).reshape((3,1))
        
        policy_ee_rot = OpenRAVEBody.transform_from_obj_pose([0, 0, 0], policy_ee[self.action_inds[('baxter', 'left_ee_rot')]])
        rot_val = []
        local_dir = np.eye(3)
        offset = np.eye(3)
        for i in range(3):
            obj_dir = np.dot(policy_ee_rot, local_dir[i])
            world_dir = robot_trans[:3,:3].dot(local_dir[i])
            rot_val.append([np.dot(obj_dir, world_dir) - offset[i].dot(local_dir[i])])
        rot_val = np.vstack(rot_vals)

        gripper_val = [x[self.state_inds[(self.robot.name, 'lGripper')]] - policy_ee[self.action_inds[(self.robot.name, 'lGripper')]]]
        return np.r_[dist_val, rot_val, gripper_val]


    def error_grad(self, x):
        jac = np.zeros((self.dU, self.dX))
        arm_inds = range(2,9)
        arm_joints = [self.robot.openrave_body.env_body.GetJointFromDOFIndex(ind) for ind in arm_inds]
        self.robot.openrave_body.set_dof({'lArmPose': x[self.state_inds[(self.robot.name, 'lArmPose')]],
                                          'lGripper': x[self.state_inds[(self.robot.name, 'lGripper')]],
                                          'rArmPose': x[self.state_inds[(self.robot.name, 'rArmPose')]],
                                          'rGripper': x[self.state_inds[(self.robot.name, 'rGripper')]]})

        robot_trans = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()
        policy_ee = self.policy_func(x)

        robot_pos = robot_trans[:3, 3]
        # policy_pos = policy_ee[self.action_inds[('baxter', 'ee_left_pos')]]
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos - joint.GetAnchor()) for joint in arm_joints]).T.copy()

        pos_jac = np.zeros((3, self.dX))
        pos_jac[:, self.state_inds[(self.robot.name, 'lArmPose')]] = arm_jac

        policy_ee_rot = OpenRAVEBody.transform_from_obj_pose([0, 0, 0], policy_ee[self.action_inds[(self.robot.name, 'left_ee_rot')]])
        rot_jac = []
        for local_dir in np.eye(3):
            obj_dir = np.dot(policy_ee_rot, local_dir)
            world_dir = robot_trans[:3,:3].dot(local_dir)
            arm_jac = np.array([np.dot(obj_dir, np.cross(joint.GetAxis(), world_dir)) for joint in arm_joints]).T.copy()
            arm_jac = arm_jac.reshape((1, len(arm_joints)))
            jac_vec = np.zeros((self.dX))
            jac_vec[self.state_inds[(self.robot.name, 'lArmPose')]] = arm_jac
            rot_jac.append(jac_vec)
        rot_jac = np.vstack(rot_jac)

        gripper_jac = np.zeros((self.dX))
        gripper_jac[self.state_inds[(self.robot.name, 'lGripper')]] = 1

        jac[self.action_inds[(self.robot.name, 'ee_left_pos')]] = pos_jac
        jac[self.action_inds[(self.robot.name, 'ee_left_rot')]] = rot_jac
        jac[self.action_inds[(self.robot.name, 'lGripper')]] = gripper_jac

        return jac
