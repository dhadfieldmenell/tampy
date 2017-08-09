#!/bin/bash
cd ../domains/laundry_domain
python generate_hl_domain.py
python generate_hl_problem.py
cd ../../src

nose2 test.test_pma.test_hl_solver.TestHLSolver.test_hl_plan --debug
