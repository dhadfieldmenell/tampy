import os


LOG_DIR = 'tf_saved/'
X_VARS = ['time', 'n_opt_calls', 'n_runs', 'n_learning_iters']
Y_VARS = ['n_success', 'opt_cost', 'tree_life']


def gen_table():
    exp_probs = os.listdir(LOG_DIR)
    for exp_name in exp_probs:
        dir_prefix = LOF_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f in file_names if f.beginswith('rollout')]
            rollout_data = {}
            for r in rollout_logs:
                with open(full_dir+'/'+r, 'r') as f:
                    rollout_data[r] = f.read()


gen_table()

