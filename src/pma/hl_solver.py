"""
HLSolver deals with interfacing to the chosen task planner, e.g. methods for
translating to PDDL if using FF/FD.
"""

class HLSolver:
    def translate(self, concr_prob, config_file):
        """
        Translates concrete (instantiated) problem to representation required for task planner.
        Also has access to the configuration file.
        E.g. for an FFsolver this would return a PDDL domain (only generated once) and problem file.
        """
        raise NotImplemented

    def solve(self, abs_prob, concr_prob):
        """
        abs_prob is what was returned by self.translate.
        An FFSolver would only need to use abs_prob here, but in
        general a task planner may make use of the geometry so we
        pass in the concrete problem as well.
        """
        raise NotImplemented
