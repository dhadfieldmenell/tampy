from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.baxter_constants as const
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

import numpy as np

from collections import OrderedDict


class BaxterPolicyEEPredicate(ExprPredicate):
    def __init__(self, name, params, state_inds, action_inds, policy_func, dX, dU, coeff, grad_coeff=0.1):
        self.handle = []
        self.policy_func = policy_func
        self.state_inds = state_inds
        self.action_inds = action_inds
        self.dX = dX
        self.dU = dU
        self.x_vec = np.zeros((dX,))
        self.coeff = coeff
        self.act_offset = 16

        self.robot = None
        for param in params:
            if param._type == 'Robot':
                self.robot = param
                break
        self.attr_inds = OrderedDict([])
        expected_param_types = [param.get_type() for param in params]

        f = lambda x: self.coeff*self.error_f(x)
        grad = lambda x: grad_coeff*self.coeff*self.error_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.dU,1))
        e = EqExpr(pos_expr, val)

        super(BaxterPolicyEEPredicate, self).__init__(name, e, self.attr_inds, params, expected_param_types, tol=1e-2, active_range=(0,0), priority=4)


    def replace_policy_func(self, new_func):
        self.policy_func = new_func


    def get_param_vector(self, t):
        for p in self.params:
            if p == self.robot: continue
            for attr in const.ATTR_MAP[p._type]:
                if (p.name, attr[0]) in self.state_inds and (p.name != 'baxter' or attr[0] =='pose'):
                    inds = self.state_inds[p.name, attr[0]] - self.act_offset
                    self.x_vec[inds] = getattr(p, attr[0])[:, t]
        self.x_vec[-16:-9] = self.robot.lArmPose[:, t]
        self.x_vec[-9:-8] = self.robot.lGripper[:, t]
        self.x_vec[-8:-1] = self.robot.rArmPose[:, t]
        self.x_vec[-1:] = self.robot.rGripper[:, t]
        return self.x_vec.reshape((self.dX, 1)).copy()


    def error_f(self, x):
        self.robot.openrave_body.set_dof({'lArmPose': x[-16:-9, 0],
                                          'lGripper': x[-9:-8, 0],
                                          'rArmPose': x[-8:-1, 0],
                                          'rGripper': x[-1:, 0]})

        l_pos = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransformPose()
        r_pos = self.robot.openrave_body.env_body.GetLink('right_gripper').GetTransformPose()
        policy_ee = self.policy_func(x.flatten().copy())

        l_dist_val = (policy_ee[self.action_inds[('baxter', 'ee_left_pos')]] - l_pos[-3:]).reshape((3,1))
        r_dist_val = (policy_ee[self.action_inds[('baxter', 'ee_right_pos')]] - r_pos[-3:]).reshape((3,1))
        l_rot_val = (policy_ee[self.action_inds[('baxter', 'ee_left_rot')]] - l_pos[:4]).reshape((4,1))
        r_rot_val = (policy_ee[self.action_inds[('baxter', 'ee_right_rot')]] - r_pos[:4]).reshape((4,1))

        l_policy_grip = const.GRIPPER_CLOSE_VALUE if policy_ee[self.action_inds[(self.robot.name, 'lGripper')]] < 0.5 else const.GRIPPER_OPEN_VALUE
        r_policy_grip = const.GRIPPER_CLOSE_VALUE if policy_ee[self.action_inds[(self.robot.name, 'rGripper')]] < 0.5 else const.GRIPPER_OPEN_VALUE
        l_gripper_val = l_policy_grip - x[self.state_inds[(self.robot.name, 'lGripper')]]
        r_gripper_val = r_policy_grip - x[self.state_inds[(self.robot.name, 'rGripper')]]
        error = np.zeros((self.dU,1))
        error[self.action_inds['baxter', 'ee_left_pos']] = l_dist_val
        error[self.action_inds['baxter', 'ee_left_rot']] = l_rot_val
        error[self.action_inds['baxter', 'ee_right_pos']] = r_dist_val
        error[self.action_inds['baxter', 'ee_right_rot']] = r_rot_val
        error[self.action_inds['baxter', 'lGripper'], 0] = l_gripper_val
        error[self.action_inds['baxter', 'rGripper'], 0] = r_gripper_val
        return error


    def error_grad(self, x):
        jac = np.zeros((self.dU, self.dX))
        l_arm_inds = range(2,9)
        l_arm_joints = [self.robot.openrave_body.env_body.GetJointFromDOFIndex(ind) for ind in l_arm_inds]
        r_arm_inds = range(10,17)
        r_arm_joints = [self.robot.openrave_body.env_body.GetJointFromDOFIndex(ind) for ind in r_arm_inds]
        self.robot.openrave_body.set_dof({'lArmPose': x[-16:-9, 0],
                                          'lGripper': x[-9:-8, 0],
                                          'rArmPose': x[-8:-1, 0],
                                          'rGripper': x[-1:, 0]})

        l_pos = self.robot.openrave_body.env_body.GetLink('left_gripper').GetTransformPose()
        r_pos = self.robot.openrave_body.env_body.GetLink('right_gripper').GetTransformPose()

        l_dist_jac = np.array([np.cross(joint.GetAxis(), l_pos[-3:] - joint.GetAnchor()) for joint in l_arm_joints]).T.copy()
        r_dist_jac = np.array([np.cross(joint.GetAxis(), r_pos[-3:] - joint.GetAnchor()) for joint in r_arm_joints]).T.copy()

        l_rot_jac = self.robot.openrave_body.env_body.GetManipulator("left_arm").CalculateRotationJacobian()
        r_rot_jac = self.robot.openrave_body.env_body.GetManipulator("right_arm").CalculateRotationJacobian()
        jac[self.action_inds[(self.robot.name, 'ee_left_pos')], -16:-9] = l_dist_jac
        jac[self.action_inds[(self.robot.name, 'ee_left_rot')], -16:-9] = l_rot_jac
        jac[self.action_inds[(self.robot.name, 'ee_right_pos')], -8:-1] = r_dist_jac
        jac[self.action_inds[(self.robot.name, 'ee_right_rot')], -8:-1] = r_rot_jac
        jac[self.action_inds[(self.robot.name, 'lGripper')], -9:-8] = 1
        jac[self.action_inds[self.robot.name, 'rGripper'], -1:] = 1

        return jac

class BaxterPolicyPredicate(ExprPredicate):
    def __init__(self, name, params, state_inds, action_inds, policy_func, dX, dU, coeff, grad_coeff=0.1):
        self.handle = []
        self.policy_func = policy_func
        self.state_inds = state_inds
        self.action_inds = action_inds
        self.dX = dX # - dU # Remove velocity state
        self.dU = dU
        self._x = np.zeros((dX,))
        self.coeff = coeff
        self.grad_coeff = grad_coeff
        self.act_offset = 0

        self.robot = None
        for param in params:
            if param._type == 'Robot':
                self.robot = param
                break
        self.attr_inds = OrderedDict([])
        self.params = params
        expected_param_types = [param.get_type() for param in params]

        f = lambda x: self.coeff*self.error_f(x)
        grad = lambda x: self.grad_coeff*self.coeff*self.error_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.dU,1))
        e = EqExpr(pos_expr, val)

        super(BaxterPolicyPredicate, self).__init__(name, e, self.attr_inds, params, expected_param_types, tol=1e-2, active_range=(0,0), priority=4)


    def replace_policy_func(self, new_func):
        self.policy_func = new_func


    def get_param_vector(self, t):
        for p in self.params:
            for attr in const.ATTR_MAP[p._type]:
                if (p.name, attr[0]) in self.state_inds:
                    if (p.name, attr[0]) in self.action_inds:
                        self._x[self.state_inds[p.name, attr[0]]] = getattr(p, attr[0])[:, t]
                    else:
                        inds = self.state_inds[p.name, attr[0]].flatten() - self.act_offset
                        self._x[inds] = getattr(p, attr[0])[:, t]
        return self._x.reshape((-1,1))

    # def get_param_vector(self, t):
    #     for p in self.params:
    #         for attr in const.ATTR_MAP[p._type]:
    #             if (p.name, attr[0]) in self.state_inds:
    #                 if (p.name, attr[0]) in self.action_inds:
    #                     self._x[self.state_inds[p.name, attr[0]]] = getattr(p, attr[0])[:, t]
    #                 else:
    #                     inds = self.state_inds[p.name, attr[0]].flatten() - self.act_offset
    #                     self._x[inds] = getattr(p, attr[0])[:, t]
    #     self._x[-self.dU:] = np.r_[self.robot.rArmPose[:, t], self.robot.rGripper[:, t],
    #                                self.robot.lArmPose[:, t], self.robot.lGripper[:, t]]
    #     return self._x.reshape((-1, 1))


    def error_f(self, x):
        X = np.zeros((self.dX))
        param_names = [param.name for param in self.params]
        for (name, attr) in self.state_inds:
            if name not in param_names or attr.endswith('__vel'): continue
            if (name, attr) in self.action_inds:
                X[self.state_inds[(name, attr)]] = x[self.state_inds[(name, attr)]].flatten()
            else:
                inds = self.state_inds[(name, attr)].flatten() - self.act_offset
                X[self.state_inds[(name, attr)]] = x[inds].flatten()
        policy_joints = self.policy_func(X.copy())

        dist_val_l = (x[self.state_inds['baxter', 'lArmPose']].flatten() - policy_joints[self.action_inds[('baxter', 'lArmPose')]])
        gripper_val_l = x[self.state_inds['baxter', 'lGripper']].flatten() - policy_joints[self.action_inds[(self.robot.name, 'lGripper')]]
        dist_val_r = (x[self.state_inds['baxter', 'rArmPose']].flatten() - policy_joints[self.action_inds[('baxter', 'rArmPose')]])
        gripper_val_r = x[self.state_inds['baxter', 'rGripper']].flatten() - policy_joints[self.action_inds[(self.robot.name, 'rGripper')]]
        return np.r_[dist_val_l, gripper_val_l, dist_val_r, gripper_val_r].reshape((-1, 1))
        # error = np.zeros((self.dX))
        # error[self.state_inds['baxter', 'lArmPose']] = dist_val_l
        # error[self.state_inds['baxter', 'rArmPose']] = dist_val_r
        # error[self.state_inds['baxter', 'lGripper']] = gripper_val_l
        # error[self.state_inds['baxter', 'rGripper']] = gripper_val_r
        # return error

    def error_grad(self, x):
        jac = np.zeros((self.dU, self.dX))
        jac[self.action_inds[('baxter', 'lArmPose')], self.state_inds[('baxter', 'lArmPose')]] = -1.0
        jac[self.action_inds[('baxter', 'rArmPose')], self.state_inds[('baxter', 'rArmPose')]] = -1.0
        jac[self.action_inds[(self.robot.name, 'lGripper')], self.state_inds[(self.robot.name, 'lGripper')]] = -1.0
        jac[self.action_inds[(self.robot.name, 'rGripper')], self.state_inds[(self.robot.name, 'rGripper')]] = -1.0

        return jac
