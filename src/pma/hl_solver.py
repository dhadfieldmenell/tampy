class HLSolver:
    """
    HLSolver deals with interfacing to the chosen task planner, e.g. methods for
    translating to PDDL if using FF/FD.
    """

    def translate(self, concr_prob, config_file):
        """
        Translates concrete (instantiated) problem to representation required for task planner.
        Also has access to the configuration file in case it's necessary. Initial state should be based
        on concr_prob initial state, NOT initial state from config_file (which may be outdated).
        E.g. for an FFsolver this would return a PDDL domain (only generated once) and problem file.
        """
        raise NotImplementedError("Override this.")

    def solve(self, abs_prob, concr_prob):
        """
        Solves the problem and returns a skeleton Plan object, which is instantiated in LLSearchNode's init.
        abs_prob is what was returned by self.translate().
        An FFSolver would only need to use abs_prob here, but in
        general a task planner may make use of the geometry, so we
        pass in the concrete problem as well.
        """
        raise NotImplementedError("Override this.")
