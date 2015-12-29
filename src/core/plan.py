"""
A plan is a sequence of actions. This class also stores trajectories and methods for executing
actions in simulation.
"""

class Plan:
    def __init__(self, actions):
        self.actions = actions
