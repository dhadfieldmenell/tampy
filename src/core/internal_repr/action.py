"""
An action is comprised of parameters, preconditions, and effects. Actions alter the state.
"""

class Action:
    def __init__(self, params, pre, eff):
        self.params = params
        self.pre = pre
        self.eff = eff
