"""
A problem is comprised of a concrete initial state and an abstract goal state.
"""

class Problem:
    def __init__(self, init, goal):
        assert init.is_concrete()
        assert goal.is_abstract()
        self.init = init
        self.goal = goal
