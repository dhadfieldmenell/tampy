from policy_hooks.gen_log_tables import *

include = []
gen_data_plots(xvar='time', yvar=['success at end', 'any target', 'subgoals closest distance'], keywords=['test_two_objects'], lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='test_run', exclude=[], split_runs=False, include=include, inter=60, window=300, ylim=[(0.,1.), (0.,1.), (0, 6)])
gen_data_plots(xvar='number of plans', yvar=['success at end', 'any target', 'subgoals closest distance'], keywords=['test_two_objects'], lab='test', label_vars=['descr'], separate=True, keyind=5, ylabel='test_run_per_plan', exclude=[], split_runs=False, include=include, inter=5, window=200, ylim=[(0.,1.), (0.,1.), (0, 6)])

