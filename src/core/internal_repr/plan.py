from IPython import embed as shell

class Plan(object):
    """
    A plan is a sequence of actions. This class also defines methods for executing
    actions in simulation using the chosen viewer.
    """
    def __init__(self, actions):
        self.actions = actions

    def execute(self):
        raise NotImplementedError
