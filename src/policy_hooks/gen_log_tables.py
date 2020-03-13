import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = 'tf_saved/'
X_VARS = ['time', 'n_opt_calls', 'n_runs', 'n_learning_iters']
Y_VARS = ['n_success', 'opt_cost', 'tree_life']


def gen_table():
    exp_probs = os.listdir(LOG_DIR)
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('rollout')]
            rollout_data = {}
            for r in rollout_logs:
                with open(full_dir+'/'+r, 'r') as f:
                    rollout_data[r] = f.read()


def gen_learning_plots(x_var='N'):
    exp_probs = os.listdir(LOG_DIR)
    print(exp_probs)
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            print(dir_name)
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('policy') and f.endswith('.txt')]
            rollout_data = {}
            for r in rollout_logs:
                print(r)
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    train, val = eval(next_data)
                    rollout_data[r] = (train, val)

            x_pts = {r: [rollout_data[r][0][i][x_var] for i in range(len(rollout_data[r][0]))] for r in rollout_data}
            y_pts = {r: [rollout_data[r][0][i]['loss'] for i in range(len(rollout_data[r][0]))] for r in rollout_data}
            #x_pts = {r: range(len(rollout_data[r][0])) for r in rollout_data}
            #y_pts = {r: rollout_data[r][0] for r in rollout_data}
            for r in rollout_data:
                plt.title(r+' train losses')
                x, y = zip(*sorted(zip(x_pts[r], y_pts[r])))
                plt.plot(x, y)
                plt.xlabel('iterations')
                plt.ylabel('loss')
                plt.savefig(full_dir+'/'+r.split('.')[0]+'_train_losses.png', pad_inches=0.01)
                plt.clf()

            x_pts = {r: [rollout_data[r][1][i][x_var] for i in range(len(rollout_data[r][1]))] for r in rollout_data}
            y_pts = {r: [rollout_data[r][1][i]['loss'] for i in range(len(rollout_data[r][1]))] for r in rollout_data}
            #x_pts = {r: range(len(rollout_data[r][1])) for r in rollout_data}
            #y_pts = {r: rollout_data[r][1] for r in rollout_data}
            for r in rollout_data:
                plt.title(r+' val losses')
                if not len(x_pts[r]): continue
                x, y = zip(*sorted(zip(x_pts[r], y_pts[r])))
                plt.plot(x, y)
                plt.xlabel('iterations')
                plt.ylabel('loss')
                plt.savefig(full_dir+'/'+r.split('.')[0]+'_val_losses.png', pad_inches=0.01)
                plt.clf()


def gen_traj_cost_plots(x_var='time'):
    exp_probs = os.listdir(LOG_DIR)
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('rollout') and f.endswith('True.txt')]
            rollout_data = {}
            for r in rollout_logs:
                print(r)
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    costs = eval(next_data)
                    rollout_data[r] = costs
            task_data = {}
            for r in rollout_data:
                data = rollout_data[r]
                costs = [(d[x_var], d['traj_cost']) for d in data if len(d['traj_cost'].keys())]
                for t, c in costs:
                    for task in c:
                        if task not in task_data:
                            task_data[task] = []
                        task_data[task].append((t,np.mean(c[task])))
            for task in task_data:
                data = np.array(task_data[task])
                x_pts = data[:,0]
                y_pts = data[:,1]
                plt.title('Task {0} traj cost'.format(task))
                plt.xlabel(x_var)
                plt.ylabel('Avg. Traj cost')
                x, y = zip(*sorted(zip(x_pts, y_pts)))
                plt.plot(x, y)
                plt.savefig(full_dir+'/'+r.split('.')[0]+'_task_{0}_traj_costs_vs_{1}.png'.format(task, x_var), pad_inches=0.01)
                plt.clf()


def gen_first_success_plots(x_var='time'):
    exp_probs = os.listdir(LOG_DIR)
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('rollout') and f.endswith('False.txt')]
            rollout_data = {}
            for r in rollout_logs:
                print(r)
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    costs = eval(next_data)
                    for c in costs:
                        c['ind'] = np.mean(c['first_success'])
                    rollout_data[r] = costs
            for r in rollout_data:
                data = rollout_data[r]
                costs = [(d[x_var], d['ind']) for d in data]
                data = np.array(costs)
                x_pts = data[:,0]
                y_pts = data[:,1]
                plt.title('Steps to first success')
                plt.xlabel(x_var)
                plt.ylabel('Step of first success')
                x, y = zip(*sorted(zip(x_pts, y_pts)))
                plt.plot(x, y)
                plt.savefig(full_dir+'/'+r.split('.')[0]+'_goal_vs_{0}.png'.format(x_var), pad_inches=0.01)
                plt.clf()


def gen_plots(x_var, y_var, overlay=True, mode='scalar', val=0.):
    exp_probs = os.listdir(LOG_DIR)
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            full_dir = dir_prefix + dir_name
            with open(full_dir+'/exp_info.txt', 'r') as f:
                exp_info = f.read()

            file_names = os.listdir(full_dir)
            rollout_logs = [f for f in file_names if f.startswith('rollout') and f.endswith('.txt')]
            rollout_data = {}
            for r in rollout_logs:
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if x_var in next_data[0] and y_var in next_data[0]:
                    rollout_data[r] = next_data

            x_pts = {r: [rollout_data[r][i][x_var] for i in range(len(rollout_data[r]))] for r in rollout_data}

            if mode=='scalar':
                y_pts = {r: [rollout_data[r][i][y_var] for i in range(len(rollout_data[r]))] for r in rollout_data}
            elif mode=='first':
                y_pts = {}
                for r in rollout_data:
                    for i in range(len(rollout_data[r])):
                        rollout_data[r][i][y_var].append(val)
                    y_pts[r] = [rollout_data[r][y_var].index(val) for i in range(len(rollout_data[r]))]
            elif mode=='avg':
                y_pts = {r: np.mean([rollout_data[r][i][y_var] for i in range(len(rollout_data[r]))]) for r in rollout_data}

            if overlay:
                plt.title('{0} vs. {1}'.format(x_var, y_var))
                c = plt.cm.rainbow(np.linspace(0, 1, len(rollout_logs)))
                for r in rollout_data:
                    plt.plot(x_pts[r], y_pts[r], c=c.pop())
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.savefig(full_dir+'_{0}_{1}.png'.format(x_var, y_var), pad_inches=0.01)
                plt.clf()

            else:
                for r in rollout_data:
                    plt.title(r+' {0} {1}'.format(x_var, y_var))
                    plt.plot(x_pts[r], y_pts[r])
                    plt.xlabel(x_var)
                    plt.ylabel(y_var)
                    plt.savefig(full_dir+'/'+r.split('.')[0]+'_{0}_{1}.png'.format(x_var, y_var), pad_inches=0.01)
                    plt.clf()

# gen_table()
# gen_plots()
gen_learning_plots()
gen_traj_cost_plots()
gen_traj_cost_plots('n_opt_calls')
# gen_first_success_plots()

