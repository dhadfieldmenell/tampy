"""
classes and methods to read a domain description
"""

import internal_rep

def parse_domain(domain_file):
    raise NotImplemented

def parse_problem(domain, problem_file):
    raise NotImplemented

def parse(domain_file, problem_file):
    domain = parse_domain(domain_file)
    return parse_problem(domain, problem_file)
