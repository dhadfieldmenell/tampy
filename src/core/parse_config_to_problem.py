"""
Read a configuration file and spawn Problem object storing initial state and goal.
"""

import internal_rep

class ParseConfigToProblem:
    def __init__(self, config_file):
        self.config_file = config_file

    def _parse_domain(self):
        raise NotImplemented

    def _parse_problem(self):
        raise NotImplemented

    def parse(self):
        domain = self._parse_domain()
        return self._parse_problem()
