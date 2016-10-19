from IPython import embed as shell

class State(object):
    """
    A state is parametrized by parameters, a 0-indexed timestep, and predicates (see Predicate class). It maintains the
    predicates that hold true at that timestep. A concrete state is one in which all the predicates are concrete.

    NOTE: Currently, we only use this class in conjunction with Problem objects' concrete initial states, for
    HL search nodes. At the low level, the state at each timestep is implicit in the parameter trajectory tables.
    """
    def __init__(self, name, params, preds=None, timestep=0):
        self.name = name
        self.params = params
        self.preds = set(preds) if preds else set()
        self.timestep = timestep

    def is_concrete(self):
        for p in self.params.itervalues():
            if not p.is_symbol() and not p.is_defined():
                return False
        return True

    def is_consistent(self):
        for p in self.preds:
            if p.active_range != (0,0): continue
            if p.is_concrete() and not p.test(time=self.timestep):
                print "Initial State Not Consistent with predicates {}".format(p)
                return False
        return True
