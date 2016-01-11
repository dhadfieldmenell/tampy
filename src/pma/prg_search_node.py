from IPython import embed as shell

class SearchNode:
    """
    There are two types of nodes in the plan refinement graph (PR Graph). High-level search
    nodes store abstract and concrete representations of the problem (concrete is an instance
    of the Problem class), and they interface to running the chosen HLSolver. Low-level search
    nodes store the Plan object for refinement, and they interface to running the chosen LLSolver.
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

    def plan(self, solver):
        raise NotImplementedError("Override this.")

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
    def __init__(self, abs_plan, concr_prob):
        """
        Instantiates a skeleton Plan object based on the abstract plan abs_plan, which is returned by
        HLSearchNode.plan(). Relevant data is retained from concr_prob, the Problem instance stored in this HL search node.
        """
        raise NotImplementedError

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
        solver.solve(self.curr_plan)

    def get_failed_pred(self):
        return self.curr_plan.get_failed_pred()
