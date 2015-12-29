"""
Read a configuration file and spawn Python objects representing
the domain and problem.
"""

import internal_rep

class ParseConfigToPy:
    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file

    def _parse_domain(self):
        raise NotImplemented

    def _parse_problem(self):
        raise NotImplemented

    def parse(self):
        domain = self._parse_domain()
        return self._parse_problem()
