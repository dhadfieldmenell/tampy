from IPython import embed as shell

class Problem(object):
    """
    Problem objects are associated with HL search nodes only. Problem objects contain a concrete initial state (see State class)
    and a set of goal predicates (which must hold true in the goal state). Each time an LL search node
    is spawned from the associated HL search node, this initial state is used to initialize parameter trajectories.
    env_data stores any relevant environment geometry information that needs to be maintained and later passed into the LLSolver. It
    gets created in the init_env classes.
    """
    def __init__(self, init_state, goal_preds, env_data, time_horizon):
        if not init_state.is_concrete():
            raise Exception("Initial state is not concrete. Have all non-symbol parameters been instantiated with a value?")
        if not init_state.is_consistent():
            raise Exception("Initial state is not consistent (predicates are violated).")
        self.init_state = init_state
        self.goal_preds = goal_preds
        self.env_data = env_data
        self.horizon = time_horizon

    def goal_test(self):
        return all(pred.test(start_time=self.horizon, end_time=self.horizon) for pred in self.goal_preds)
