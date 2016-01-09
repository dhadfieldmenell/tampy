class Action:
    """
    An action is comprised of parameters, preconditions, and effects.
    """
    def __init__(self, params, pre, eff):
        self.params = params
        self.pre = pre
        self.eff = eff
