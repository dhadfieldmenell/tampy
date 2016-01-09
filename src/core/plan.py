class Plan:
    """
    A plan is a sequence of actions. This class also has methods for executing
    actions in simulation.
    """
    def __init__(self, actions):
        self.actions = actions
