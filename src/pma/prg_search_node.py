from IPython import embed as shell
from core.internal_repr.state import State
from core.internal_repr.problem import Problem
from core.util_classes.learning import PostLearner
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
        return -self.priority

    def is_hl_node(self):
        return False

    def is_ll_node(self):
        return False

    def plan(self):
        raise NotImplementedError("Override this.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, domain, concr_prob, priority = 0, prefix = None):
        self.abs_prob = abs_prob
        self.domain = domain
        self.concr_prob = concr_prob
        self.prefix = prefix if prefix else []
        self.priority = priority

    def is_hl_node(self):
        return True

    def plan(self, solver):
        plan_obj = solver.solve(self.abs_prob, self.domain, self.concr_prob, self.prefix)
        return plan_obj

class LLSearchNode(SearchNode):
    def __init__(self, plan, prob, priority = 1):
        self.curr_plan = plan
        self.concr_prob = prob
        self.child_record = {}
        self.priority = priority

    def get_problem(self, i, failed_pred, suggester):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        state_name = "state_{}".format(self.priority)
        state_params = self.curr_plan.params.copy()
        last_action = [a for a in self.curr_plan.actions if a.active_timesteps[0] <= i and a.active_timesteps[1] >= i][0]
        state_timestep = last_action.active_timesteps[0]
        # state_timestep = 0
        state_preds = [p['pred'] for p in last_action.preds if p['hl_info'] != 'eff' and  p['negated'] == False]
        for p in state_preds:
            arange = p.active_range
            p.active_range = (arange[0], max(arange[1], state_timestep))
        state_preds.append(failed_pred)
        new_state = State(state_name, state_params, state_preds, state_timestep)
        """
        Suggester shuold sampling based on biased Distribution according to learned
        theta for each parameter.
        """
        feature_fun = None
        resampled_action = suggester.sample(state, feature_fun)
        """
        End of Suggester
        """
        goal_preds = self.concr_prob.goal_preds.copy()
        new_problem = Problem(new_state, goal_preds, self.concr_prob.env)

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
        """
            Make sure plan refinement graph doesn't generate doplicate child.
            self.child_record is a dict that maps planing prefix to failed predicates it
            encountered so far.
        """

        fail_step, fail_pred = self.get_failed_pred()
        plan_prefix = tuple(self.curr_plan.prefix(fail_step))
        fail_pred_type = fail_pred.get_type()

        if plan_prefix not in self.child_record.keys() or fail_pred_type not in self.child_record.get(plan_prefix, []):
            self.child_record[plan_prefix] = self.child_record.get(plan_prefix, []) + [fail_pred_type]
            return True
        else:
            return False
