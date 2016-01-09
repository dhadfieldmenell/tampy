from core import parse_config_to_problem
import parse_config_to_solvers

"""
Many methods called in this class have documentation.
"""
def p_mod_abs(config_file, max_iter=100):
    problem = parse_config_to_problem.parse(config_file)
    hl_solver, ll_solver = parse_config_to_solvers.parse(config_file)
    n0 = HLSearchNode(hl_solver.translate(problem, config_file), problem)

    Q = PriorityQueue()
    Q.push(n0, 0)
    for _ in range(max_iter):
        n = Q.pop()
        if n.is_hl_node():
            p_c = n.plan(hl_solver)
            c = LLSearchNode(p_c, n.concr_prob)
            Q.push(n, n.heuristic())
            Q.push(c, c.heuristic())
        elif n.is_ll_node():
            n.plan(ll_solver)
            if n.solved():
                return n.extract_plan()
            Q.push(n, n.heuristic())
            if n.gen_child():
                fail_step, fail_pred = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred)
                c = HLSearchNode(hl_solver.translate(n_problem, config_file), n_problem, prefix=n.plan.prefix(fail_step))
                Q.push(c, c.heuristic())
        else:
            raise NotImplementedError

    return False

class SearchNode(object):
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either HL or LL search node.")

    def heuristic(self):
        return 0

    def is_hl_node(self):
        return False
    
    def is_ll_node(self):
        return False

    def plan(self, solver):
        raise NotImplementedError("Call plan() for HL or LL search node.")
    
class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, concr_prob, prefix=None):
        self.prefix = None
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
