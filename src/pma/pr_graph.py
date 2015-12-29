from core import parse_config_to_py
import parse_config_to_solvers

def p_mod_abs(domain_file, problem_file, max_iter=100):
    # returns Problem instance
    problem = parse_config_to_py.parse(domain_file, problem_file)
    # returns objects that set up abstract and concrete
    # problems to solve
    hl_solver, ll_solver = parse_config_to_solvers.parse(domain_file, problem_file)
    # search nodes are keyed by high-level rep
    n0 = HLSearchNode(hl_solver.translate(problem), problem)

    Q = PriorityQueue()
    Q.push(n0, 0)
    for _ in range(max_iter):
        n = Q.pop()
        # is this an HL node
        if n.is_abs():
            p_c = n.plan(hl_solver)
            c = LLSearchNode(p_c, n.concr_prob)
            Q.push(n, n.heuristic())
            Q.push(c, c.heuristic())
        # is this an LL node
        elif n.is_concrete():
            # updates n.plan
            n.plan(ll_solver)
            if n.solved():
                return n.extract_plan()
            # push back onto queue
            Q.push(n, n.heuristic())
            if n.gen_child():
                # returns timestep and a predicate that isn't
                # satisfied in the current plan
                i, fail = n.get_failed_pred()
                n_problem = n.get_problem(i, fail)
                c = HLSearchNode(hl_solve.translate(n_problem), n_problem, prefix=n.plan.prefix(i))
                Q.push(c, c.heuristic())
        else:
            raise NotImplemented

    return False

class SearchNode(object):
    def __init__(self):
        raise NotImplemented

    def heuristic(self):
        return 0

    def is_concrete(self):
        return False
    
    def is_abs(self):
        return False

    def plan(self, solver):
        raise NotImplemented
    
class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, concr_prob, prefix=None):
        self.prefix = None
        self.abs_prob = abs_prob
        self.concr_prob = concr_prob

    def is_abs(self):
        return True

    def plan(self, solver):
        """
        generate a plan to solve this problem
        """
        return self.prefix + solver.solve(self.abs_prob, self.concr_prob)

class LLSearchNode(SearchNode):    
    def __init__(self, plan, concr_prob):
        self.plan = plan
        self.problem = concr_prob

    def get_problem(self, i, failed_pred):
        """
        return a representation of the search problem which
        starts from the end state of step i and goes to the same goal        
        """
        raise NotImplemented

    def solved(self):
        raise NotImplemented

    def is_concrete(self):
        return True

    def plan(self, solver):
        """
        use solver to spend computation optimizing the plan
        should also increment any state tracking the optimization
        history
        """
        solver.solve(self.plan)

    def get_failed_pred(self):
        return self.plan.get_failed_pred()
