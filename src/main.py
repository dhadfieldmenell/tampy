from IPython import embed as shell
import argparse, traceback
from errors_exceptions import TampyException
from pma import pr_graph

"""
Entry-level script. Calls pr_graph.p_mod_abs() to plan, then runs the plans in
simulation using the chosen viewer.
"""

def parse_file_to_dict(f_name):
    d = {}
    with open(f_name, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d

def main(domain_file, problem_file, solvers_file):
    try:
        domain_config = parse_file_to_dict(domain_file)
        problem_config = parse_file_to_dict(problem_file)
        solvers_config = parse_file_to_dict(solvers_file)
        plan, msg = pr_graph.p_mod_abs(domain_config, problem_config, solvers_config)
        if plan:
            print "Executing plan!"
            plan.execute()
        else:
            print msg
    except TampyException as e:
        print "Caught an exception in Tampy:"
        traceback.print_exc()
        print "Terminating..."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tampy.")
    parser.add_argument("domain_file",
                        help="Path to the domain file to use. All domain settings should be specified in this file.")
    parser.add_argument("problem_file",
                        help="Path to the problem file to use. All problem settings should be specified in this file. Spawned by a generate_*_prob.py script.")
    parser.add_argument("solvers_file",
                        help="Path to the file naming the solvers to use. The HLSolver and LLSolver to use should be specified here.")
    args = parser.parse_args()
    main(args.domain_file, args.problem_file, args.solvers_file)
