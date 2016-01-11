from IPython import embed as shell

class State:
    """
    A state is parametrized by a 0-indexed timestep and predicates (see Predicate class). It maintains the
    predicates that hold true at that timestep. A concrete state is one in which all the predicates are concrete.

    NOTE: Currently, we only use this class in conjunction with Problem objects' concrete initial states, for
    HL search nodes. At the low level, states are implicit in the parameter trajectory tables.
    """
    def __init__(self, name, preds, timestep):
        self.name = name
        self.preds = preds
        # add all parameters used into a set, for convenience in case it's ever needed
        self.params = set()
        for pred in self.preds:
            for param in pred.params:
                self.params.add(param)
        self.timestep = timestep

    def is_concrete(self):
        return all(pred.is_concrete() for pred in self.preds)

    def is_consistent(self):
        return all(pred.test(start_time=self.timestep, end_time=self.timestep) for pred in self.preds)
