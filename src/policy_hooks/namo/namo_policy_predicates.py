from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.baxter_constants as const
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

import numpy as np

from collections import OrderedDict

class NAMOPolicyPredicate(ExprPredicate):   
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

        dist_val = (x[self.state_inds['pr2', 'pose']].flatten() - policy_joints[self.action_inds[('baxter', 'lArmPose')]])
        gripper_val = x[self.state_inds['pr2', 'gripper']].flatten() - policy_joints[self.action_inds[(self.robot.name, 'lGripper')]]
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