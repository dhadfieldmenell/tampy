import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

FRAME = 10
TWINDOW = 900
TDELTA = 150
MIN_FRAME = 30
nan = np.nan
LOG_DIR = 'tf_saved/'
SAVE_DIR = '/home/michaelmcdonald/Dropbox/'
X_VARS = ['time', 'n_opt_calls', 'n_runs', 'n_learning_iters']
Y_VARS = ['n_success', 'opt_cost', 'tree_life']


def get_colors(n_colors):
    return cm.rainbow(np.linspace(0, 1, n_colors))


def get_policy_data(policy, keywords=[]):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                full_dir = dir_prefix + dir_name
                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                r = 'policy_{0}_log.txt'.format(policy)
                rollout_data = {}
                if not os.path.isfile(full_dir+'/'+r): continue
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    r_data = eval(next_data)
                    for pt in r_data:
                        if type(pt['var']) is dict:
                            pt['var'] = pt['var'][policy]
                    rollout_data[r] = r_data
                    data[k][full_exp][full_dir] = rollout_data

    return data


def get_rollout_data(keywords=[], nfiles=20):
    exp_probs = os.listdir(LOG_DIR)
    data = {}
    for k in keywords:
        data[k] = {}
        for exp_name in exp_probs:
            dir_prefix = LOG_DIR + exp_name + '/'
            if not os.path.isdir(dir_prefix): continue
            exp_dirs = os.listdir(dir_prefix)
            for dir_name in exp_dirs:
                d = dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                full_dir = dir_prefix + dir_name
                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                rollout_logs = ['rollout_log_{0}_True.txt'.format(i) for i in range(nfiles)]# [f for f in file_names if f.startswith('rollout')]
                rollout_logs += ['rollout_log_{0}_False.txt'.format(i) for i in range(nfiles)]
                rollout_data = {}
                for r in rollout_logs:
                    if not os.path.isfile(full_dir+'/'+r): continue
                    with open(full_dir+'/'+r, 'r') as f:
                        next_data = f.read()
                    if len(next_data):
                        r_data = eval(next_data)
                        rollout_data[r] = r_data
                    else:
                        print('no data for', r)
                data[k][full_exp][full_dir] = rollout_data

    return data


def gen_first_success_plots(x_var='time'):
    exp_probs = os.listdir(LOG_DIR)
    master_plot = []
    all_data = []
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
            ts_to_data = {}
            for i, r in enumerate(rollout_logs):
                print(exp_name, dir_name, r)
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    costs = eval(next_data)
                    if 'avg_first_success' not in costs[0]: continue
                    for j, c in enumerate(costs):
                        c['ind'] = np.mean(c['avg_first_success'])
                        if j not in ts_to_data:
                            ts_to_data[j] = []
                        ts_to_data[j].append(c)
                    rollout_data[r] = costs
            xs = [np.mean([c[x_var] for c in ts_to_data[n]]) for n in ts_to_data]
            ys = [np.mean([c['ind'] for c in ts_to_data[n]]) for n in ts_to_data]
            all_data.append((exp_name+dir_name, xs, ys))
                
    plt.title('Steps to first success')
    plt.xlabel(x_var)
    plt.ylabel('Avg. step of first success')
    colors = get_colors(len(all_data))
    for i, (label, xs, ys) in enumerate(all_data):
        plt.plot(xs, ys, label=label, color=colors[i])
    plt.savefig(SAVE_DIR+'/goal_vs_{0}.png'.format(x_var), pad_inches=0.01)
    plt.clf()


def get_td_loss(keywords=[], exclude=[], pre=False):
    tdelta = 5
    exp_probs = os.listdir(LOG_DIR)
    exp_data = {}
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        # exp_data = []
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = True
                for k in keywords:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue

            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:-1]
            i = 0
            data = []
            while i < 20: 
                if not os.path.isdir(full_exp+str(i)):
                    i += 1
                    continue
                fnames = os.listdir(full_exp+str(i))
                info = [f for f in fnames if f.find('td_error') >= 0 and f.endswith('npy') and f.find('test') < 0]
                if len(info):
                    cur_data = []
                    for step in info:
                        cur_data.append(np.load(full_exp+str(i)+'/'+step))
                    dlen = max([len(dt) for dt in cur_data])
                    data.append([])
                    for n in range(dlen):
                        data[-1].append(np.mean([cur_data[ind][n] for ind in range(len(cur_data)) if n < len(cur_data[ind])], axis=0))
                    data[-1] = np.array(data[-1])
                i += 1

            if not len(data): 
                print('skipping', full_exp)
                continue
            dlen = min([len(d) for d in data])
            dmax = max([len(d) for d in data])
            print('Gathering data for', full_exp, 'length:', dlen, 'all len:', [len(d) for d in data])
            end = False
            cur_t = 0
            while not end:
                end = True
                for d in data:
                    next_frame = d[cur_t:cur_t+tdelta] # [pt[0] for pt in d if pt[0,3] >= cur_t and pt[0,3] <= cur_t + TWINDOW]
                    if len(next_frame):
                        end = False
                        if len(next_frame) >= MIN_FRAME:
                            next_pt = np.mean(next_frame, axis=0)
                            no, nt = 0, 0 # int(next_pt[4]), int(next_pt[2])
                            if (no, nt) not in exp_data:
                                exp_data[no, nt] = []
                            exp_data[no, nt].append((full_exp, cur_t, next_pt[0]))
                cur_t += tdelta


            '''
            for i in range(dmax - FRAME):
                cur_t = np.mean([d[i:i+FRAME,:,3] for d in data if i+FRAME < len(d)])

                for d in data:
                    if len(d) < i+FRAME: continue
                    cur_fr = np.mean(d[i:i+FRAME], axis=0)
                    for pt in cur_fr:
                        val = pt[0]
                        # cur_t = pt[3]
                        nt = int(pt[2])
                        no = int(pt[4])
                        if (no, nt) not in exp_data:
                            exp_data[no, nt] = []
                        exp_data[no, nt].append((full_exp, cur_t, val))
            '''
        for no, nt in exp_data:
            print('Plotting', no, nt, exp_name)
            pd_frame = pd.DataFrame(exp_data[no, nt], columns=['exp_name', 'time', 'value'])
            sns.set()
            sns_plot = sns.relplot(x='time', y='value', hue='exp_name', kind='line', data=pd_frame)
            keyid = ''
            for key in keywords:
                keyid += '_{0}'.format(key)
            pre_lab = '_pre' if pre else ''
            sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_td_error{2}{3}.png'.format(no, nt, keyid, pre_lab))




def get_hl_tests(keywords=[], exclude=[], pre=False, rerun=False, xvar='time', avg_time=True, tdelta=TDELTA, wind=TWINDOW, lab=''):
    exp_probs = os.listdir(LOG_DIR)
    exp_data = {}
    exp_len_data = {}
    used = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        # exp_data = []
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = True
                for k in keywords:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue

            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:full_dir.rfind('_')]
            # full_exp = full_dir[:-1]
            if full_exp in used: continue
            used.append(full_exp)
            i = 0
            data = []
            while i < 20: 
                '''
                if not os.path.isdir(full_exp+str(i)):
                    i += 1
                    continue
                fnames = os.listdir(full_exp+str(i))
                '''
                cur_dir = '{0}_{1}'.format(full_exp, i)
                if not os.path.isdir(cur_dir):
                    i += 1
                    continue
                fnames = os.listdir(cur_dir)

                if pre:
                    info = [f for f in fnames if f.find('hl_test_pre') >= 0 and f.endswith('pre_log.npy')]
                elif rerun:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_rerun_log.npy')]
                else:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                if len(info):
                    # data.append(np.load(full_exp+str(i)+'/'+info[0]))
                    data.append(np.load(cur_dir+'/'+info[0]))
                    if len(data[-1].shape) < 3: data = data[:-1]
                i += 1

            if not len(data): 
                print('skipping', full_exp)
                continue
            dlen = min([len(d) for d in data])
            dmax = max([len(d) for d in data])
            print('Gathering data for', full_exp, 'length:', dlen, 'all len:', [len(d) for d in data])
            end = False
            cur_t = 0
            while not end:
                end = True
                for d in data:
                    if not avg_time or rerun:
                        next_frame = [pt[0] for pt in d]
                    elif xvar == 'N':
                        next_frame = [pt[0] for pt in d if pt[0,6] >= cur_t and pt[0,6] <= cur_t + wind]
                    else:
                        next_frame = [pt[0] for pt in d if pt[0,3] >= cur_t and pt[0,3] <= cur_t + wind]
                    if len(next_frame):
                        end = not avg_time
                        if rerun or xvar != 'time' or len(next_frame) >= MIN_FRAME:
                            if avg_time:
                                next_pts = [np.mean(next_frame, axis=0)]
                            else:
                                next_pts = next_frame

                            for next_pt in next_pts:
                                no, nt = int(next_pt[4]), int(next_pt[5])
                                if (no, nt) not in exp_data:
                                    exp_data[no, nt] = []
                                if (no, nt) not in exp_len_data:
                                    exp_len_data[no, nt] = []
                                
                                if xvar == 'time':
                                    exp_data[no, nt].append((full_exp, cur_t, next_pt[0]))
                                    exp_len_data[no, nt].append((full_exp, cur_t, next_pt[1]))
                                elif xvar == 'N':
                                    exp_data[no, nt].append((full_exp, next_pt[6], next_pt[0]))
                                    exp_len_data[no, nt].append((full_exp, next_pt[6], next_pt[1]))
                                elif rerun:
                                    exp_data[no, nt].append((full_exp, next_pt[-1], next_pt[0]))
                                    exp_len_data[no, nt].append((full_exp, next_pt[-1], next_pt[1]))
                cur_t += tdelta


            '''
            for i in range(dmax - FRAME):
                cur_t = np.mean([d[i:i+FRAME,:,3] for d in data if i+FRAME < len(d)])

                for d in data:
                    if len(d) < i+FRAME: continue
                    cur_fr = np.mean(d[i:i+FRAME], axis=0)
                    for pt in cur_fr:
                        val = pt[0]
                        # cur_t = pt[3]
                        nt = int(pt[2])
                        no = int(pt[4])
                        if (no, nt) not in exp_data:
                            exp_data[no, nt] = []
                        exp_data[no, nt].append((full_exp, cur_t, val))
            '''
        for no, nt in exp_data:
            print('Plotting', no, nt, exp_name)
            pd_frame = pd.DataFrame(exp_data[no, nt], columns=['exp_name', xvar, 'value'])
            sns.set()
            sns_plot = sns.relplot(x=xvar, y='value', hue='exp_name', kind='line', data=pd_frame)
            keyid = ''
            for key in keywords:
                keyid += '_{0}'.format(key)
            pre_lab = '_pre' if pre else ''
            if rerun: pre_lab += '_rerun'
            sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_val{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))

        for no, nt in exp_len_data:
            print('Plotting', no, nt, exp_name)
            pd_frame = pd.DataFrame(exp_len_data[no, nt], columns=['exp_name', xvar, 'length'])
            sns.set()
            sns_plot = sns.relplot(x=xvar, y='length', hue='exp_name', kind='line', data=pd_frame)
            keyid = ''
            for key in keywords:
                keyid += '_{0}'.format(key)
            pre_lab = '_pre' if pre else ''
            if rerun: pre_lab += '_rerun'
            sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_len{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))


def plot(data, columns, descr):
    for k in data:
        pd_frame = pd.DataFrame(data[k], columns=columns)
        sns.set()
        sns_plot = sns.relplot(x=columns[1], y=columns[2], hue=columns[0], kind='line', data=pd_frame)
        sns_plot.savefig(SAVE_DIR+'/{0}_{1}.png'.format(k, descr))



def gen_data_plots(xvar, yvar, keywords=[], lab='rollout'):
    if lab == 'rollout':
        rd = get_rollout_data(keywords)
    else:
        rd = get_policy_data(lab, keywords)
    data = {}
    print('Collected data...')
    for keyword in rd:
        key_data = []
        new_data = rd[keyword]
        for fullexp in rd[keyword]:
            e = new_data[fullexp]
            exp_data = {xvar:[], yvar:[]}
            for ename in e:
                curexp = e[ename]
                cur_xdata = []
                cur_ydata = []

                for rname in curexp:
                    r = curexp[rname]
                    if not len(r): continue
                    if xvar not in r[0] or yvar not in r[0]: continue
                    cur_xdata.append([np.mean([r[t][xvar] for t in range(i, i+FRAME)]) for i in range(len(r)-FRAME)])
                    cur_ydata.append([np.mean([r[t][yvar] for t in range(i, i+FRAME)]) for i in range(len(r)-FRAME)])
                if not len(cur_xdata): 
                    print('Skipping', xvar, yvar, 'for', ename)
                    continue
                dlen = min([len(d) for d in cur_xdata])
                for i in range(len(cur_xdata)):
                    cur_xdata[i] = cur_xdata[i][:dlen]
                    cur_ydata[i] = cur_ydata[i][:dlen]
                cur_xdata = np.mean(np.array(cur_xdata), axis=0)
                cur_ydata = np.mean(np.array(cur_ydata), axis=0)

                exp_data[xvar].append(cur_xdata)
                exp_data[yvar].append(cur_ydata)
            if not len(exp_data[xvar]):
                print('no data from', xvar, 'for', fullexp)
                continue
            dlen = min([len(d) for d in exp_data[xvar]])
            print('DLEN', dlen, fullexp)
            xvals = np.mean([dn[:dlen] for dn in exp_data[xvar]], axis=0)
            for i in range(dlen):
                for yvals in exp_data[yvar]:
                    key_data.append((fullexp, xvals[i], yvals[i]))
            print('Set data for', keyword, fullexp)
        data[keyword] = key_data

    plot(data, ['exp_name', xvar, yvar], '{0}_vs_{1}'.format(xvar, yvar))


#get_hl_tests(['retrain_2by'], xvar='N', avg_time=False, tdelta=5000, wind=5000, pre=False, exclude=['0001', '10000'])
get_hl_tests(['compact_base'], xvar='time', pre=False)
get_hl_tests(['compact_base'], xvar='time', pre=True)
gen_data_plots(xvar='time', yvar='val_loss', keywords=['compact_base'], lab='policy')

