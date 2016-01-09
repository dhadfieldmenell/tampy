class State:
    """
    A state is parametrized by a timestep and predicates (see Predicate class). It maintains the
    predicates that hold true at that timestep. A concrete state is one in which all state variables have values.

    NOTE: Currently, we only use this class in conjunction with Problem objects' concrete initial states, for
    HL search nodes. At the low level, states are implicit in the parameter trajectory tables.
    """
    def __init__(self, name, preds):
        self.name = name
        self.preds = preds
        self.consistent = True
        for p in self.preds:
            if not p.test(objs):
                self.consistent = False
        self._is_concr = self.check_concrete()

    def is_concrete(self):
        return self._is_concr

    def check_concrete(self):
        raise NotImplementedError
