from IPython import embed as shell
import argparse
from pma import pr_graph

"""
Entry-level script. Calls pr_graph.p_mod_abs() to plan, then runs the plans in
simulation using the chosen viewer.
"""

def parse_config_file_to_dict(config_file):
    config = {}
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split(":", 1)
                config[k.strip()] = v.strip()
    return config

def main(config_file):
    config = parse_config_file_to_dict(config_file)
    plan = pr_graph.p_mod_abs(config)
    if plan:
        print "Executing plan!"
        plan.execute()
    else:
        print "Unable to find valid plan."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tampy.")
    parser.add_argument("config_file",
                        help="Path to the config.txt file to use. All settings should be specified in this file.")
    args = parser.parse_args()
    main(args.config_file)
