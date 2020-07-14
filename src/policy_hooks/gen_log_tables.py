import cPickle as pickle
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import array, float32
import seaborn as sns

FRAME = 10
TWINDOW = 300
TDELTA = 300
MIN_FRAME = 30
nan = np.nan
LOG_DIR = 'tf_saved/'
prefix = os.path.expanduser('~')
SAVE_DIR = prefix+'/Dropbox/'
X_VARS = ['time', 'n_opt_calls', 'n_runs', 'n_learning_iters']
Y_VARS = ['n_success', 'opt_cost', 'tree_life']


def get_colors(n_colors):
    return cm.rainbow(np.linspace(0, 1, n_colors))


def get_policy_data(policy, keywords=[], exclude=[]):
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
                full_dir = dir_prefix + dir_name
                if d.find('.') >= 0 or d.find('trained') >= 0: continue
                skip = False
                for ekey in exclude:
                    if full_dir.find(ekey) >= 0:
                        skip = True
                if skip: continue
                if not os.path.isdir(full_dir) or full_dir.find(k) < 0: continue
                full_exp = full_dir[:-1]
                if full_exp not in data[k]:
                    data[k][full_exp] = {}

                file_names = os.listdir(full_dir)
                r = 'policy_{0}_log.txt'.format(policy)
                rollout_data = {}
                if not os.path.isfile(full_dir+'/'+r):
                    r = 'policy_{0}_log.pkl'.format(policy)
                if not os.path.isfile(full_dir+'/'+r): continue
                with open(full_dir+'/'+r, 'r') as f:
                    next_data = f.read()
                if len(next_data):
                    r_data = eval(next_data)
                    for pt in r_data:
                        if type(pt['train_loss']) is dict:
                            pt['train_loss'] = pt['train_loss']['loss']
                        if type(pt['val_loss']) is dict:
                            pt['val_loss'] = pt['val_loss']['loss']
                        if 'var' in pt and type(pt['var']) is dict:
                            pt['var'] = pt['var'][policy]
                    rollout_data[r] = r_data
                    data[k][full_exp][full_dir] = rollout_data

    return data


def get_rollout_data(keywords=[], nfiles=20, exclude=[]):
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


def get_fail_info(keywords=[], exclude=[], pre=False, rerun=False, xvar='time', avg_time=True, tdelta=TDELTA, wind=TWINDOW, lab='', lenthresh=0.99, label_vars=[], include=[], max_t=14400):
    exp_probs = os.listdir(LOG_DIR)
    exp_data = {}
    exp_len_data = {}
    exp_dist_data = {}
    exp_true_data = {}
    targets = {}
    used = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = False
                for k in keywords:
                    if dir_name.find(k) < 0 and dir_prefix.find(k) < 0:
                        skip = True
                if skip: continue

            if len(include):
                skip = True
                for k in include:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue


            print(dir_name)
            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                        print('skipping', dir_name)
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:full_dir.rfind('_')]
            if full_exp in used: continue
            used.append(full_exp)
            i = 0
            data = []
            while i < 20: 
                cur_dir = '{0}_{1}'.format(full_exp, i)
                if not os.path.isdir(cur_dir):
                    i += 1
                    continue
                fnames = os.listdir(cur_dir)

                info = [f for f in fnames if f.find('failure') >= 0 and f.endswith('data.txt')]
                if len(info):
                    for fname in info:
                        print('Loading data from', fname)
                        with open(cur_dir+'/'+fname, 'r') as f:
                            data.append(f.read().splitlines())

                    label = gen_label(cur_dir, label_vars)
                    for buf in data:
                        for pts in buf:
                            pts = eval(pts)
                            no, nt = int(pts['no']), int(pts['nt'])
                            if (no,nt) not in exp_data: exp_data[no,nt] = []
                            if (no,nt) not in targets: targets[no,nt] = []
                            pts['exp_name'] = label
                            exp_data[no,nt].append(pts)
                            targs = pts['goal']
                            targinfo = []
                            for targind in range(len(targs)):
                                if targs[targind] == 1:
                                    targinfo.append(targind)
                            targets[no,nt].append([tuple(targinfo)])
                i += 1

    for no, nt in targets:
        print('Plotting', no, nt, exp_name)
        pd_frame = pd.DataFrame(targets[no, nt], columns=['targets']) #'target_{0}'.format(i) for i in range(no)])
        print(pd_frame['targets'].value_counts()[:10])
        sns.set()
        plt.clf()
        sns_plot = sns.countplot(x="value", hue="variable", data=pd.melt(pd_frame))
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        plt.savefig(SAVE_DIR+'/{0}obj_{1}targ_failedtargets{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))
        plt.clf()


def get_hl_tests(keywords=[], exclude=[], pre=False, rerun=False, xvar='time', avg_time=True, tdelta=TDELTA, wind=TWINDOW, lab='', lenthresh=0.99, label_vars=[], include=[], max_t=14400):
    exp_probs = os.listdir(LOG_DIR)
    all_data = []
    exp_data = {}
    exp_len_data = {}
    exp_dist_data = {}
    exp_true_data = {}
    used = []
    for exp_name in exp_probs:
        dir_prefix = LOG_DIR + exp_name + '/'
        if not os.path.isdir(dir_prefix): continue
        exp_dirs = os.listdir(dir_prefix)
        for dir_name in exp_dirs:
            d = dir_name
            if d.find('.') >= 0 or d.find('trained') >= 0: continue
            if len(keywords):
                skip = False
                for k in keywords:
                    if dir_name.find(k) < 0 and dir_prefix.find(k) < 0:
                        skip = True
                if skip: continue

            if len(include):
                skip = True
                for k in include:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = False
                if skip: continue


            print(dir_name)
            if len(exclude):
                skip = False
                for k in exclude:
                    if dir_name.find(k) >= 0 or dir_prefix.find(k) >= 0:
                        skip = True
                        print('skipping', dir_name)
                if skip: continue

            full_dir = dir_prefix + dir_name
            full_exp = full_dir[:full_dir.rfind('_')]
            if full_exp in used: continue
            used.append(full_exp)
            i = 0
            data = []
            while i < 20: 
                cur_dir = '{0}_{1}'.format(full_exp, i)
                if not os.path.isdir(cur_dir):
                    i += 1
                    continue
                fnames = os.listdir(cur_dir)

                if pre:
                    info = [f for f in fnames if f.find('hl_test_pre') >= 0 and f.endswith('pre_log.npy')]
                elif rerun:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                else:
                    info = [f for f in fnames if f.find('hl_test') >= 0 and f.endswith('test_log.npy')]
                if len(info):
                    for fname in info:
                        print('Loading data from', fname)
                        data.append(np.load(cur_dir+'/'+fname))

                    label = gen_label(cur_dir, label_vars)
                    for buf in data:
                        for pts in buf:
                            pt = pts[0]
                            no, nt = int(pt[4]), int(pt[5])
                            if (no,nt) not in exp_data: exp_data[no,nt] = []
                            if (no,nt) not in exp_len_data: exp_len_data[no,nt] = []
                            if (no,nt) not in exp_dist_data: exp_dist_data[no,nt] = []
                            if (no,nt) not in exp_true_data: exp_true_data[no,nt] = []
                             
                            if xvar == 'time':
                                xval = (pt[3] // tdelta) * tdelta
                                if xval < max_t:
                                    exp_data[no, nt].append((label, xval, pt[0]))
                                    exp_data[no, nt].append((label, xval+tdelta, pt[0]))
                                    if pt[0] > lenthresh:
                                        exp_len_data[no, nt].append((label, xval, pt[1]))
                                    if pt[0] <= 0.5: exp_dist_data[no, nt].append((label, xval, pt[2]))
                                    if len(pt) > 7: exp_true_data[no, nt].append((label, xval, pt[7]))
                            elif rerun:
                                exp_data[no, nt].append((label, pt[-1], pt[0]))
                                if pt[0] > lenthresh:
                                    exp_len_data[no, nt].append((label, pt[-1], pt[1]))
                            elif xvar == 'N' or xvar == 'n_data':
                                exp_data[no, nt].append((label, pt[6], pt[0]))
                                if pt[0] > lenthresh:
                                    exp_len_data[no, nt].append((label, pt[6], pt[1]))
                            all_data.append({'time': (pt[3]//tdelta)*tdelta, 'value': pt[0], 'len': pt[1], 'dist': pt[2], 'N': pt[6], 'key': (no, nt), 'description': label, 'ind': i})
                        
                i += 1

    pd_frame = pd.DataFrame(all_data, columns=['time', 'description', 'N', 'value', 'len', 'dist', 'key', 'ind'])
    pd_frame = pd_frame.groupby(['time', 'description', 'key', 'ind'], as_index=False).mean()
    sns.set()
    fig = plt.figure(figsize=(10,6))
    axs = fig.subplots(ncols=3)
    sns_plot = sns.relplot(x=xvar, y='value', hue='description', row='key', kind='line', data=pd_frame)
    sns_plot.fig.set_figwidth(10)
    sns_plot._legend.remove()
    sns_plot.fig.get_axes()[0].legend(loc=(0.25, -0.5))
    sns_plot.fig.axes[0].set_title('value')

    l, b, w, h = sns_plot.fig.axes[0]._position.bounds
    sns_plot.fig.add_axes((l+w+0.1, b, w, h))
    sns_plot_2 = sns.relplot(x=xvar, y='dist', hue='description', row='key', kind='line', data=pd_frame, legend=False, ax=sns_plot.fig.axes[1])
    sns_plot.fig.axes[1].set_title('distance')
   
    l, b, w, h = sns_plot.fig.axes[1]._position.bounds
    sns_plot.fig.add_axes((l+w+0.1, b, w, h))
    sns_plot_2 = sns.relplot(x=xvar, y='len', hue='description', row='key', kind='line', data=pd_frame, legend=False, ax=sns_plot.fig.axes[2])
    sns_plot.fig.axes[2].set_title('length')
    keyid = ''
    for key in keywords:
        keyid += '_'+str(key)
    sns_plot.fig.savefig(SAVE_DIR+'/allgraphs_{0}_{1}.png'.format(keyid, lab), bbox_inches="tight")
    # fig.savefig(SAVE_DIR+'/allgraphs_{0}_{1}.png'.format(keyid, lab), bbox_inches="tight")

    for no, nt in exp_data:
        print('Plotting', no, nt, exp_name)
        pd_frame = pd.DataFrame(exp_data[no, nt], columns=['exp_name', xvar, 'value'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='value', hue='exp_name', kind='line', data=pd_frame)
        sns_plot._legend.remove()
        sns_plot.fig.get_axes()[0].legend(loc=(1.5, 1.5))
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.fig.savefig(SAVE_DIR+'/{0}obj_{1}targ_val{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab), bbox_inches="tight")

    for no, nt in exp_len_data:
        print('Plotting', no, nt, exp_name)
        pd_frame = pd.DataFrame(exp_len_data[no, nt], columns=['exp_name', xvar, 'length'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='length', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_len{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))

    for no, nt in exp_dist_data:
        print('Plotting', no, nt, exp_name)
        pd_frame = pd.DataFrame(exp_dist_data[no, nt], columns=['exp_name', xvar, 'disp'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='disp', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_disp{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))


    for no, nt in exp_true_data:
        print('Plotting', no, nt, exp_name)
        pd_frame = pd.DataFrame(exp_true_data[no, nt], columns=['exp_name', xvar, 'true'])
        sns.set()
        sns_plot = sns.relplot(x=xvar, y='true', hue='exp_name', kind='line', data=pd_frame)
        keyid = ''
        for key in keywords[:-1]:
            keyid += '_{0}'.format(key)
        pre_lab = '_pre' if pre else ''
        if rerun: pre_lab += '_rerun'
        sns_plot.savefig(SAVE_DIR+'/{0}obj_{1}targ_true{2}{3}{4}.png'.format(no, nt, keyid, pre_lab, lab))


def plot(data, columns, descr, separate=True, keyind=3):
    sns.set()
    if not separate:
        d = []
        for k in data:
            d.extend(data[k])
        if not len(d) : return
        pd_frame = pd.DataFrame(d, columns=columns)
        # leg_labels = getattr(pd_frame, columns[0]).unique()
        sns_plot = sns.relplot(x=columns[1], y=columns[2], hue=columns[0], row=columns[keyind], kind='line', data=pd_frame)
        sns_plot.savefig(SAVE_DIR+'/{0}_{1}.png'.format(k, descr))
        sns.set()

    else:
        for k in data:
            if not len(data[k]): continue
            pd_frame = pd.DataFrame(data[k], columns=columns)
            leg_labels = getattr(pd_frame, columns[0]).unique()
            sns_plot = sns.relplot(x=columns[1], y=columns[2], hue=columns[0], col_wrap=3, col=columns[keyind], kind='line', data=pd_frame)
            '''
            for axes in sns_plot.axes.flat:
                box = axes.get_position()
                axes.set_position([box.x0,box.y0,box.width,box.height*0.9])
            handles, _ = sns_plot.plt.get_legend_handles_labels()
            sns_plot.plt.legend(handles, leg_labels, bbox_to_anchor=(0.5, 0), loc='upper center')
            '''
            sns_plot.savefig(SAVE_DIR+'/{0}_{1}_{2}.png'.format(k, descr, columns[keyind]))
            sns.set()
    print('PLOTTED for', descr)

def gen_label(exp_dir, label_vars=[]):
    if not len(label_vars) or not os.path.isfile(exp_dir+'/args.pkl'):
        return exp_dir[:exp_dir.rfind('_')]
    label = ''
    with open(exp_dir+'/args.pkl', 'r') as f:
        args = pickle.load(f)
    args = vars(args)
    for v in label_vars:
        if v not in args: continue
        label += ' {0}_{1}'.format(v, args[v])
    return label 


def gen_data_plots(xvar, yvar, keywords=[], lab='rollout', inter=100, label_vars=[], ylabel='value', separate=True, keyind=3, exclude=[]):
    if lab == 'rollout':
        rd = get_rollout_data(keywords, exclude=exclude)
    else:
        rd = get_policy_data(lab, keywords, exclude=exclude)
    data = {}
    print('Collected data...')
    yvars = [yvar] if type(yvar) is not list else yvar
    for keyword in rd:
        key_data = []
        new_data = rd[keyword]
        for fullexp in rd[keyword]:
            e = new_data[fullexp]
            for ename in e:
                label = gen_label(ename, label_vars)
                curexp = e[ename]
                for rname in curexp:
                    r = curexp[rname]
                    if not len(r): continue
                    if xvar not in r[0] or any([v not in r[0] for v in yvars]): continue
                    for pt in r:
                        xval = (pt[xvar] // inter) * inter
                        for v in yvars:
                            if hasattr(pt[v], '__getitem__'): 
                                for i in range(len(pt[v])):
                                    if keyind != 5:
                                        key_data.append([label+' component {0} in {1}'.format(i, v), xval, pt[v][i], keyword, v, i])
                                    else:
                                        key_data.append([label+' '+v, xval, pt[v][i], keyword, v, i])
                            else:
                                key_data.append([label+' {0}'.format(v), xval, pt[v], keyword, v, 0])
            print('Set data for', keyword, fullexp)
        data[keyword] = key_data

    plot(data, ['exp_name', xvar, ylabel, 'key', 'yvar', 'yvar_ind'], '{0}_vs_{1}'.format(xvar, ylabel), separate=separate, keyind=keyind)

keywords = ['failtrain']
include = [] # ['wed_nocol', 'sun']
label_vars = ['descr', 'hist_len', 'check_col', 'soft_eval'] # ['eta', 'train_iterations', 'lr', 'prim_weight_decay'] # ['prim_dim', 'prim_n_layers', 'prim_weight_decay', 'eta', 'lr', 'train_iterations']
#get_hl_tests(['retrain_2by'], xvar='N', avg_time=False, tdelta=5000, wind=5000, pre=False, exclude=['0001', '10000'])
get_hl_tests(keywords, xvar='time', pre=False, label_vars=label_vars, lenthresh=0.9, exclude=[], include=include)
#get_fail_info(keywords, xvar='time', pre=False, label_vars=label_vars, lenthresh=0.9, exclude=['nocol_det', 'nocol_nohist'], include=include, max_t=5000)
#get_hl_tests(keywords[1:2], xvar='n_data', pre=False, label_vars=label_vars, lenthresh=-1)
#get_hl_tests(keywords[2:3], xvar='n_data', pre=False, label_vars=label_vars, lenthresh=-1)
#get_hl_tests(['valcheck_2'], xvar='time', pre=False, label_vars=['split_nets'], lenthresh=-1)
#get_hl_tests(['compact_base'], xvar='time', pre=True)
#keywords = ['goalpureloss', 'grasppureloss', 'plainpureloss', 'taskpureloss']
#label_vars = ['train_iterations', 'lr', 'prim_weight_decay'] # ['prim_dim', 'prim_n_layers', 'prim_weight_decay', 'eta', 'lr', 'train_iterations']
#gen_data_plots(xvar='n_data', yvar=['train_component_loss', 'val_component_loss'], keywords=keywords, lab='primitive', label_vars=label_vars, separate=True, keyind=5, ylabel='loss_comp_3', exclude=[])
gen_data_plots(xvar='n_data', yvar=['err'], keywords=keywords, lab='primitive', label_vars=label_vars, separate=False, keyind=5, ylabel='loss', exclude=[])
gen_data_plots(xvar='n_data', yvar=['train_component_loss', 'val_component_loss'], keywords=keywords, lab='primitive', label_vars=label_vars, separate=False, keyind=5, ylabel='loss_comp_3', exclude=[])


