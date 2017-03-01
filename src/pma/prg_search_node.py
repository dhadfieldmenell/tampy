from IPython import embed as shell
from core.internal_repr.state import State
from core.internal_repr.problem import Problem
import random

class SearchNode(object):
    """
    There are two types of nodes in the plan refinement graph (PRGraph). High-level search
    nodes store abstract and concrete representations of the problem (concrete is an instance
    of the Problem class), and they interface to planning with the chosen HLSolver. Low-level search
    nodes store the Plan object for refinement, and they interface to planning with the chosen LLSolver.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either HL or LL search node.")

    def heuristic(self):
        """
        The node with the highest heuristic value is selected at each iteration of p_mod_abs.
        """
        return 0

    def is_hl_node(self):
        return False

    def is_ll_node(self):
        return False

    def plan(self):
        raise NotImplementedError("Override this.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, domain, concr_prob, prefix=None):
        self.abs_prob = abs_prob
        self.domain = domain
        self.concr_prob = concr_prob
        self.prefix = prefix if prefix else []

    def is_hl_node(self):
        return True

    def plan(self, solver):
        plan_obj = solver.solve(self.abs_prob, self.domain, self.concr_prob)
        if self.prefix:
            return self.prefix + plan_obj
        else:
            return plan_obj

    def heuristic(self):
        return -1

class LLSearchNode(SearchNode):
    def __init__(self, plan, prob):
        self.curr_plan = plan
        self.concr_prob = prob

    def get_problem(self, i, failed_pred):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        params = self.concr_prob.init_state.params.copy()
        last_action = [a for a in self.curr_plan.actions
                    if a.active_timesteps[0] <= i and a.active_timesteps[1] >= i][0]
        state_time = last_action.active_timesteps[0]
        preds = [p['pred'] for p in last_action.preds if p['hl_info'] != 'eff' and p['negated'] == False]
        new_state = State("state_{}".format(i), params, preds, state_time)
        # import ipdb; ipdb.set_trace()
        # TODO figure out new state
        new_problem = Problem(new_state, self.concr_prob.goal_preds.copy(),
                              self.concr_prob.env)
        return new_problem

    def solved(self):
        return len(self.curr_plan.get_failed_preds()) == 0

    def is_ll_node(self):
        return True

    def plan(self, solver):
        solver.solve(self.curr_plan)

    def get_failed_pred(self):
        failed_pred = self.curr_plan.get_failed_pred()
        return failed_pred[2], failed_pred[1]

    def gen_child(self):
        return random.randint(0,1)
