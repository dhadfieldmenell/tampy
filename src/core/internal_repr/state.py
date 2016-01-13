from IPython import embed as shell

class State(object):
    """
    A state is parametrized by a 0-indexed timestep and predicates (see Predicate class). It maintains the
    predicates that hold true at that timestep. A concrete state is one in which all the predicates are concrete.

    NOTE: Currently, we only use this class in conjunction with Problem objects' concrete initial states, for
    HL search nodes. At the low level, states are implicit in the parameter trajectory tables.
    """
    def __init__(self, name, params, preds=None, timestep=0):
        self.name = name
        self.params = set(params)
        self.preds = preds if preds else []
        self.timestep = timestep

    def is_concrete(self):
        return all(pred.is_concrete() for pred in self.preds)

    def is_consistent(self):
        return all(pred.test(time=self.timestep) for pred in self.preds)
