from IPython import embed as shell

class Domain(object):
    """
    A single Domain object gets created every time p_mod_abs is run. It stores the parameter, predicate, and action
    schemas. It also stores the hl_solver and ll_solver objects, along with an abstract representation
    of the domain, which gets passed into the task planner in HLSolver.solve().
    """
    def __init__(self, hl_solver, ll_solver, abs_domain, param_schema, pred_schema, action_schema):
        self.hl_solver = hl_solver
        self.ll_solver = ll_solver
        self.abs_domain = abs_domain
        self.param_schema = param_schema
        self.pred_schema = pred_schema
        self.action_schema = action_schema
