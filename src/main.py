from IPython import embed as shell
import argparse
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

def main(domain_file, problem_file):
    domain_config = parse_file_to_dict(domain_file)
    problem_config = parse_file_to_dict(problem_file)
    plan = pr_graph.p_mod_abs(domain_config, problem_config)
    if plan:
        print "Executing plan!"
        plan.execute()
    else:
        print "Unable to find valid plan."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tampy.")
    parser.add_argument("domain_file",
                        help="Path to the domain file to use. All domain settings should be specified in this file.")
    parser.add_argument("problem_file",
                        help="Path to the problem file to use. All problem settings should be specified in this file. Spawned by a generate_*_prob.py script.")
    args = parser.parse_args()
    main(args.domain_file, args.problem_file)
