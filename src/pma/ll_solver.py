from IPython import embed as shell

class LLSolver:
    """
    LLSolver solves the underlying motion planning problem. This is where different refinement strategies
    (e.g. backtracking, randomized), different motion planners, and different optimization strategies (global, sequential)
    are handled.
    """
    def solve(self, plan):
        raise NotImplementedError("Override this.")

class NAMOSolver(LLSolver):
    pass

class CanSolver(LLSolver):
    pass

class DummyLLSolver(LLSolver):
    def solve(self, plan):
        return "solve"
