"""
HLSolver deals with interfacing to the chosen task planner, e.g. methods for
translating to PDDL if using FF/FD.
"""

class HLSolver:
    def translate(self, abs_prob, concr_prob):
        raise NotImplemented
