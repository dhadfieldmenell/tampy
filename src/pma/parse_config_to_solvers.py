from IPython import embed as shell
import hl_solver
import ll_solver

class ParseConfigToSolvers:
    """
    Read the configuration data and spawn HLSolver and LLSolver objects.
    """
    def __init__(self, config):
        self.config = config

    def parse(self):
        if "HLSolver" not in self.config or "LLSolver" not in self.config:
            raise Exception("Must define both HL solver and LL solver in config file.")
        s = self.config["HLSolver"]
        if not hasattr(hl_solver, s):
            raise Exception("HLSolver '%s' not defined!"%s)
        hls = getattr(hl_solver, s)()
        s = self.config["LLSolver"]
        if not hasattr(ll_solver, s):
            raise Exception("LLSolver '%s' not defined!"%s)
        lls = getattr(ll_solver, s)()
        return hls, lls
