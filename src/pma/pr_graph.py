from IPython import embed as shell
from core.parse_config_to_problem import ParseConfigToProblem
from parse_config_to_solvers import ParseConfigToSolvers
import argparse
from Queue import PriorityQueue

"""
Many methods called in this class have detailed documentation.
"""
def p_mod_abs(config, max_iter=100):
    problem = ParseConfigToProblem(config).parse()
    if problem.goal_test():
        print "Goal is already satisfied. No planning done."
        return False
    hl_solver, ll_solver = ParseConfigToSolvers(config).parse()
    n0 = HLSearchNode(hl_solver.translate(problem, config), problem)

    Q = PriorityQueue()
    Q.put((0, n0))
    for _ in range(max_iter):
        n = Q.get()[1]
        if n.is_hl_node():
            p_c = n.plan(hl_solver)
            c = LLSearchNode(p_c, n.concr_prob)
            Q.put((-n.heuristic(), n))
            Q.put((-c.heuristic(), c))
        elif n.is_ll_node():
            n.plan(ll_solver)
            if n.solved():
                return n.extract_plan()
            Q.put((-n.heuristic(), n))
            if n.gen_child():
                fail_step, fail_pred = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred)
                c = HLSearchNode(hl_solver.translate(n_problem, config), n_problem, prefix=n.plan.prefix(fail_step))
                Q.put((-c.heuristic(), c))

    return False

class SearchNode(object):
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either HL or LL search node.")

    def heuristic(self):
        """
        The node with the highest heuristic value is selected at each iteration.
        """
        return 0

    def is_hl_node(self):
        return False
    
    def is_ll_node(self):
        return False

    def plan(self, solver):
        raise NotImplementedError("Call plan() for HL or LL search node.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, concr_prob, prefix=[]):
        self.prefix = prefix
        self.abs_prob = abs_prob
        self.concr_prob = concr_prob

    def is_hl_node(self):
        return True

    def plan(self, solver):
        return self.prefix + solver.solve(self.abs_prob, self.concr_prob)

class LLSearchNode(SearchNode):
    def __init__(self, plan, concr_prob):
        """
        Instantiates skeleton Plan object by retaining relevant data from the Problem instance stored at the HL search node.
        """
        self.plan = plan
        self.problem = concr_prob

    def get_problem(self, i, failed_pred):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        raise NotImplementedError

    def solved(self):
        raise NotImplementedError

    def is_ll_node(self):
        return True

    def plan(self, solver):
        """
        Uses solver to spend computation optimizing the plan.
        """
        solver.solve(self.plan)

    def get_failed_pred(self):
        return self.plan.get_failed_pred()
