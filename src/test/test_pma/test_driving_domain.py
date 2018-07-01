import numpy as np
import unittest, time, main
from pma import hl_solver, driving_solver
from core.driving_utils import *
from core.parsing import parse_domain_config, parse_driving_problem_config

class TestDrivingDomain(unittest.Testcase):
    def basic_drive(self):
        domain_fname = '../domains/driving_domain/driving.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        print "loading driving problem..."
        p_c = main.parse_file_to_dict('../domains/driving_domain/driving_probs/basic.prob')
        problem = parse_driving_problem_config.ParseDrivingProblemConfig.parse(p_c, domain)

        plan_str = [
            '1: DRIVE_DOWN_ROAD USER0 ROAD0 START0 END0 VU_LIMIT VL_LIMIT AU_LIMIT AL_LIMIT'
        ]
        plan = hls.get_plan(plan_str, domain, problem)
