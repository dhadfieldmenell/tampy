from core.util_classes.common_predicates import ExprPredicate
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.baxter_constants as const
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

import numpy as np

from collections import OrderedDict

class NAMOPolicyPredicate(ExprPredicate):
    def __init__(self, name, plan, policy_func, coeff, agent, task_ind, obj_ind, targ_ind, grad_coeff=0.1):
        self.handle = []
        self.policy_func = policy_func
        self.state_inds = plan.state_inds
        self.action_inds = plan.action_inds
        self.dX = plan.dX
        self.dU = plan.dU
        self.coeff = coeff
        self.grad_coeff = grad_coeff
        self.agent = agent
        self.task_ind = task_ind
        self.obj_ind = obj_ind
        self.targ_ind = targ_ind

        self._u = np.zeros(self.dU)

        self.attr_inds = OrderedDict([])
        self.plan = plan
        expected_param_types = []

        f = lambda x: self.coeff*self.error_f(x)
        grad = lambda x: self.grad_coeff*self.coeff*self.error_grad(x)

        pos_expr, val = Expr(f, grad), np.zeros((self.dU,1))
        e = EqExpr(pos_expr, val)

        super(NAMOPolicyPredicate, self).__init__(name, e, self.attr_inds, [], expected_param_types, tol=1e-2, active_range=(-agent.hist_len, 1), priority=4)


    def replace_policy_func(self, new_func):
        self.policy_func = new_func


    def get_param_vector(self, t):
        for p_name in self.plan.params:
            p = self.plan.params[p_name]
            for attr in const.ATTR_MAP[p._type]:
                if (p.name, attr[0]) in self.action_inds:
                    self._u[self.action_inds[p.name, attr[0]]] = getattr(p, attr[0])[:, t]
        return self._u.reshape((-1,1))


    def error_f(self, x):
        sample = Sample(self.agent)
        task_vec = np.zeros((len(self.agent.task_list)), dtype=np.float32)
        task_vec[self.task_ind] = 1.
        obj_vec = np.zeros((len(self.agent.obj_list)), dtype='float32')
        targ_vec = np.zeros((len(self.agent.targ_list)), dtype='float32')
        obj_vec[self.obj_ind] = 1.
        targ_vec[self.targ_ind] = 1.
        target_vec = np.zeros((self.agent.target_dim,))
        for target_name in self.agent.targ_list:
            target = self.plan.params[target_name]
            target_vec[self.agent.target_inds[target.name, 'value']] = target.value[:,0]

        sample.set(STATE_ENUM, X[self.dU*hist_len:self.dU*(hist_len+1)].copy(), 0)
        sample.set(TASK_ENUM, task_vec, 0)
        sample.set(OBJ_ENUM, obj_vec, 0)
        sample.set(TARG_ENUM, targ_vec, 0)
        sample.set(TARGETS_ENUM, target_vec, 0)
        hist_len = self.agent.hist_len
        hist = np.array(x[:self.dU*hist_len]).reshape((hist_len, self.dU))
        sample.set(TRAJ_HIST, hist, 0)
        policy_out = self.policy_func(sample)
        return policy_out.flatten() - X[self.dU*(hist_len+1):]

    def error_grad(self, x):
        active_len = self.active_range[1] - self.active_range[0] + 1
        jac = np.zeros((self.dU, active_len*self.dU))
        jac[self.action_inds[('pr2', 'pose')], self.action_inds[('pr2', 'pose')] + (active_len - 1)*self.dU] = -1.0
        jac[self.action_inds[('pr2', 'gripper')], self.action_inds[('pr2', 'gripper')] + (active_len - 1)*self.dU] = -1.0

        return jac
