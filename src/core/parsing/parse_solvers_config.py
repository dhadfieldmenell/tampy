from errors_exceptions import HLException, LLException, SolversConfigException
from pma import hl_solver, ll_solver_gurobi


class ParseSolversConfig(object):
    """
    Read the solver configuration data and spawn the corresponding HLSolver and LLSolver objects.
    """
    @staticmethod
    def parse(solvers_config, domain_config):
        # parse out the HLSolver and LLSolver
        if "HLSolver" not in solvers_config or "LLSolver" not in solvers_config:
            raise SolversConfigException("Must define both HL solver and LL solver in solvers config file.")
        s = solvers_config["HLSolver"]
        if not hasattr(hl_solver, s):
            raise HLException("HLSolver '%s' not defined!"%s)
        hls = getattr(hl_solver, s)(domain_config)
        s = solvers_config["LLSolver"]
        if not hasattr(ll_solver, s):
            raise LLException("LLSolver '%s' not defined!"%s)
        lls = getattr(ll_solver, s)()

        return hls, lls
