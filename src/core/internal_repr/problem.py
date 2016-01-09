class Problem:
    """
    Problem objects are associated with HL search nodes only, so they don't store information about the
    parameter trajectories across the entire time horizon. Problem objects contain a concrete initial state (see State class)
    and a set of goal predicates (which must hold true in the goal state). Each time an LL search node
    is spawned from the associated HL search node, this initial state is used to initialize parameter trajectories.
    """
    def __init__(self, key, init_state, goal_preds):
        self.key = key
        assert init_state.is_concrete()
        self.init_state = init_state
        self.goal_preds = goal_preds
