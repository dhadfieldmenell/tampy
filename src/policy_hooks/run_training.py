import argparse
import copy
import imp
import importlib
import os
import pickle
import random
import shutil
import sys
import time

from policy_hooks.multiprocess_main import MultiProcessMain


def get_dir_name(base, no, nt, ind, descr, args=None):
    dir_name = base + 'objs{0}_{1}/exp_id{2}_{3}'.format(no, nt, ind, descr)
    if args is not None and not len(descr):
        useq = '_qfunc' if args.qfunc else ''
        useHer = '_her' if args.her else ''
        expand = '_expand' if args.expand else ''
        neg = '_negExs' if args.negative else ''
        onehot = '_onehot' if args.onehot_task else ''
        curric = '_curric{0}_{1}'.format(args.cur_thresh, args.n_thresh) if args.cur_thresh > 0 else ''
        dir_name += '{0}{1}{2}{3}{4}{5}'.format(useq, useHer, expand, curric, neg, onehot)
    return dir_name


def load_multi(exp_list, n_objs=None, n_targs=None, args=None):
    exps = []
    for exp in exp_list:
        configs = []
        for i in range(len(exp)):
            if n_objs is None:
                c = exp[i]
                next_config = config_module.config.copy()
                config_module = importlib.import_module(c)
            elif n_targs is None:
                n_targs = n_objs
            if n_objs is not None:
                c = exp[i]
                config_module = importlib.import_module(c)
                next_config = config_module.refresh_config(n_objs, n_targs)
            if args is not None:
                next_config.update(vars(args))
            next_config['weight_dir'] = get_dir_name(next_config['base_weight_dir'], next_config['num_objs'], next_config['num_targs'], i, next_config['descr'], args)
            next_config['server_id'] = '{0}'.format(str(random.randint(0, 2**16)))
            next_config['mp_server'] = True
            next_config['pol_server'] = True
            next_config['mcts_server'] = True
            next_config['use_local'] = True
            next_config['log_server'] = False
            next_config['view_server'] = False
            next_config['use_local'] = True
            next_config['log_timing'] = False
            configs.append((next_config, config_module))

        exps.append(configs)
    return exps


def load_config(args, config=None, reload_module=None):
    config_file = args.config
    if reload_module is not None:
        config_module = reload_module
        imp.reload(config_module)
    else:
        config_module = importlib.import_module('policy_hooks.'+config_file)
    config = config_module.config
    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds']
    config['common']['num_conds'] = config['num_conds']
    config['algorithm']['conditions'] = config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs'])
    config['log_timing'] = args.timing
    # config['pretrain_timeout'] = args.pretrain_timeout
    config['hl_timeout'] = args.hl_timeout if args.hl_timeout > 0 else config['hl_timeout']
    config['mcts_server'] = args.mcts_server or args.all_servers
    config['mp_server'] = args.mp_server or args.all_servers
    config['pol_server'] = args.policy_server or args.all_servers
    config['log_server'] = args.log_server or args.all_servers
    config['view_server'] = args.view_server
    config['pretrain_steps'] = args.pretrain_steps if args.pretrain_steps > 0 else config['pretrain_steps']
    config['viewer'] = args.viewer
    config['server_id'] = args.server_id if args.server_id != '' else str(random.randint(0,2**32))
    return config, config_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config')
    parser.add_argument('-test', '--test', type=str, default='')
    parser.add_argument('-no', '--nobjs', type=int, default=0)
    parser.add_argument('-nt', '--ntargs', type=int, default=0)
    parser.add_argument('-hist_len', '--hist_len', type=int, default=1)
    parser.add_argument('-obs_del', '--add_obs_delta', action='store_true', default=False)
    parser.add_argument('-act_hist', '--add_action_hist', action='store_true', default=False)
    parser.add_argument('-smooth', '--traj_smooth', action='store_true', default=False)
    parser.add_argument('-hl_retrain', '--hl_retrain', action='store_true', default=False)
    parser.add_argument('-seq', '--seq', action='store_true', default=False)
    parser.add_argument('-hl_only_retrain', '--hl_only_retrain', action='store_true', default=False)

    # Old
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-hlt', '--hl_timeout', type=int, default=0)
    parser.add_argument('-k', '--killall', action='store_true', default=True)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)

    # Server specs
    parser.add_argument('-mcts', '--mcts_server', action='store_true', default=False)
    parser.add_argument('-mp', '--mp_server', action='store_true', default=False)
    parser.add_argument('-pol', '--policy_server', action='store_true', default=False)
    parser.add_argument('-log', '--log_server', action='store_true', default=False)
    parser.add_argument('-vs', '--view_server', action='store_true', default=False)
    parser.add_argument('-all', '--all_servers', action='store_true', default=False)
    parser.add_argument('-ps', '--pretrain_steps', type=int, default=0)
    parser.add_argument('-v', '--viewer', action='store_true', default=False)
    parser.add_argument('-id', '--server_id', type=str, default='')

    # Exp config misc
    parser.add_argument('-f', '--file', type=str, default='')
    parser.add_argument('-descr', '--descr', type=str, default='')
    parser.add_argument('-her', '--her', action='store_true', default=False)
    parser.add_argument('-e', '--expand_process', action='store_true', default=False)
    parser.add_argument('-neg', '--negative', action='store_true', default=False)
    parser.add_argument('-oht', '--onehot_task', action='store_true', default=False)
    parser.add_argument('-render', '--load_render', action='store_true', default=False)
    parser.add_argument('-retime', '--retime', action='store_true', default=False)
    parser.add_argument('-local_retime', '--local_retime', action='store_true', default=False)
    parser.add_argument('-vel', '--velocity', type=float, default=0.3)
    parser.add_argument('-nocol', '--check_col', action='store_false', default=True)
    parser.add_argument('-cond', '--conditional', action='store_true', default=False)

    # Previous policy directories
    parser.add_argument('-llpol', '--ll_policy', type=str, default='')
    parser.add_argument('-hlpol', '--hl_policy', type=str, default='')
    parser.add_argument('-hldata', '--hl_data', type=str, default='')
    parser.add_argument('-hlsamples', '--hl_samples', type=str, default='')

    # Curric args
    parser.add_argument('-cur', '--curric_thresh', type=int, default=-1)
    parser.add_argument('-ncur', '--n_thresh', type=int, default=10)

    # NN args
    parser.add_argument('-spl', '--split_nets', action='store_false', default=True)
    parser.add_argument('-lldim', '--dim_hidden', type=int, default=32)
    parser.add_argument('-lln', '--n_layers', type=int, default=2)
    parser.add_argument('-hldim', '--prim_dim_hidden', type=int, default=32)
    parser.add_argument('-hln', '--prim_n_layers', type=int, default=2)
    parser.add_argument('-llus', '--update_size', type=int, default=2000)
    parser.add_argument('-hlus', '--prim_update_size', type=int, default=5000)
    parser.add_argument('-iters', '--train_iterations', type=int, default=50)
    parser.add_argument('-batch', '--batch_size', type=int, default=500)
    parser.add_argument('-lldec', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-hldec', '--prim_weight_decay', type=float, default=1e-3)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-hllr', '--hllr', type=float, default=1e-3)
    parser.add_argument('-lr_policy', '--lr_policy', type=str, default='fixed')

    # HL args
    parser.add_argument('-check_t', '--check_prim_t', type=int, default=1)
    parser.add_argument('-ff', '--ff_thresh', type=float, default=0)
    parser.add_argument('-ff_only', '--ff_only', action='store_true', default=False)
    parser.add_argument('-fail', '--train_on_fail', action='store_true', default=False)
    parser.add_argument('-failmode', '--fail_mode', type=str, default='start')
    parser.add_argument('-aughl', '--augment_hl', action='store_true', default=False)
    parser.add_argument('-x_select', '--state_select', type=str, default='base')
    parser.add_argument('-prim_decay', '--prim_decay', type=float, default=1.)
    parser.add_argument('-prim_first_wt', '--prim_first_wt', type=float, default=1e0)
    parser.add_argument('-soft', '--soft', action='store_true', default=False)
    parser.add_argument('-eta', '--eta', type=float, default=5.)
    parser.add_argument('-add_noop', '--add_noop', type=int, default=0)
    parser.add_argument('-softev', '--soft_eval', action='store_true', default=False)
    parser.add_argument('-pre', '--check_precond', action='store_true', default=False)
    parser.add_argument('-mask', '--hl_mask', action='store_false', default=True)

    # Q learn args
    parser.add_argument('-qimwt', '--q_imwt', type=float, default=0)
    parser.add_argument('-q', '--use_qfunc', action='store_true', default=False)

    args = parser.parse_args()

    exps = None
    if args.file == "":
        exps = [[args.config]]

    if False:#args.file == "":
        config, config_module = load_config(args)

    else:
        print(('LOADING {0}'.format(args.file)))
        current_id = 0
        if exps is None:
            exps = []
            with open(args.file, 'r+') as f:
                exps = eval(f.read())
        exps_info = exps
        n_objs = args.nobjs if args.nobjs > 0 else None
        n_targs = args.ntargs if args.ntargs > 0 else None
        if len(args.test):
            sys.path.insert(1, 'tf_saved/'+args.test)
            exps_info = [['hyp']]
            old_args = args
            with open('tf_saved/'+args.test+'/args.pkl', 'r') as f:
                args = pickle.load(f)
            args.soft_eval = old_args.soft_eval
            args.test = old_args.test
            args.load_render = old_args.load_render
            args.eta = old_args.eta
            args.descr = old_args.descr
        if args.hl_retrain:
            sys.path.insert(1, 'tf_saved/'+args.hl_data)
            exps_info = [['hyp']]

        exps = load_multi(exps_info, n_objs, n_targs, args)
        for ind, exp in enumerate(exps):
            mains = []
            for ind2, (c, cm) in enumerate(exp):
                if len(args.test):
                    c['weight_dir'] = args.test
                    m = MultiProcessMain(c)
                    m.run_test(c)
                    continue

                print('\n\n\n\n\n\nLOADING NEXT EXPERIMENT\n\n\n\n\n\n')
                while os.path.isdir('tf_saved/'+c['weight_dir']+'_'+str(current_id)):
                    current_id += 1
                c['group_id'] = current_id
                c['weight_dir'] = c['weight_dir']+'_{0}'.format(current_id)
                dir_name = ''
                sub_dirs = ['tf_saved'] + c['weight_dir'].split('/')
                for d_ind, d in enumerate(sub_dirs):
                    dir_name += d + '/'
                    if not os.path.isdir(dir_name):
                        os.mkdir(dir_name)
                if args.hl_retrain:
                    src = 'tf_saved/' + args.hl_data + '/hyp.py'
                else:
                    src = exps_info[ind][ind2].replace('.', '/')+'.py'
                shutil.copyfile(src, 'tf_saved/'+c['weight_dir']+'/hyp.py')
                with open('tf_saved/'+c['weight_dir']+'/__init__.py', 'w+') as f:
                    f.write('')
                with open('tf_saved/'+c['weight_dir']+'/args.pkl', 'wb+') as f:
                    pickle.dump(args, f)

                m = MultiProcessMain(c)
                m.monitor = False # If true, m will wait to finish before moving on
                m.group_id = current_id
                with open('tf_saved/'+c['weight_dir']+'/exp_info.txt', 'w+') as f:
                    f.write(str(cm))

                if args.hl_retrain:
                    m.hl_retrain(c)
                elif args.hl_only_retrain:
                    m.hl_only_retrain(c)
                else:
                    m.start()
                mains.append(m)
                time.sleep(1)
            active = True

            start_t = time.time()
            while active:
                time.sleep(120.)
                print('RUNNING...')
                active = False
                for m in mains:
                    p_info = m.check_processes()
                    print(('PINFO {0}'.format(p_info)))
                    active = active or any([code is None for code in p_info])
                    if active: m.expand_rollout_servers()

                if not active:
                    for m in mains:
                        m.kill_processes()

        print('\n\n\n\n\n\n\n\nEXITING')
        sys.exit(0)

    if not args.nofull:
        main = MultiProcessMain(config)
        main.start(kill_all=args.killall)

if __name__ == '__main__':
    main()
