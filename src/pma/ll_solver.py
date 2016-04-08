from IPython import embed as shell

class LLSolver(object):
    """
    LLSolver solves the underlying optimization problem using motion planning. This is where different refinement strategies
    (e.g. backtracking, randomized), different motion planners, and different optimization strategies (global, sequential)
    are implemented.
    """
    def solve(self, plan):
        raise NotImplementedError("Override this.")

class NAMOSolver(LLSolver):
    def solve(self, plan):
        raise NotImplementedError

class CanSolver(LLSolver):
    pass

class DummyLLSolver(LLSolver):
    def solve(self, plan):
        return "solve"
