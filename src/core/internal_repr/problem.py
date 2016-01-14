from IPython import embed as shell

class Problem(object):
    """
    Problem objects are associated with HL search nodes only. Problem objects contain a concrete initial state (see State class)
    and a set of goal predicates (which must hold true in the goal state). Each time an LL search node
    is spawned from the associated HL search node, this initial state is used to initialize parameter trajectories.
    """
    def __init__(self, init_state, goal_preds):
        if not init_state.is_concrete():
            raise Exception("Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?")
        if not init_state.is_consistent():
            raise Exception("Initial state is not consistent (predicates are violated).")
        self.init_state = init_state
        self.goal_preds = goal_preds

    def goal_test(self):
        # because problems are associated with HL search nodes,
        # only need to check timestep 0 here
        return all(pred.test(time=0) for pred in self.goal_preds)
