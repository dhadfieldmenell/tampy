from IPython import embed as shell
import os
from glob import glob
import sys

"""
Every *.py file in this codebase must have a corresponding test_*.py file in the test/ folder,
at the same location in the mimicked directory structure. This script checks for that.
"""

EXCLUDE = ["__init__", "check_test_dir_structure"]

files = [y for x in os.walk(".") for y in glob(os.path.join(x[0], '*.py'))]
for f in files:
    if not f.startswith("./test") and not any(x in f for x in EXCLUDE):
        f2 = f.replace("/", "/test_").replace("./", "./test/")
        if f2 not in files:
            print("%s doesn't have matching test script, aborting."%f)
            sys.exit(-1)
