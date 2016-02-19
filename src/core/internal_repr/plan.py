from IPython import embed as shell

class Plan(object):
    """
    A plan has the following.

    params: dictionary of plan parameters, mapping name to object
    actions: list of Actions
    horizon: total number of timesteps for plan

    This class also defines methods for executing actions in simulation using the chosen viewer.
    """
    IMPOSSIBLE = "Impossible"

    def __init__(self, params, actions, horizon):
        self.params = params
        self.actions = actions
        self.horizon = horizon

    def execute(self):
        raise NotImplementedError
