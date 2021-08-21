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

USE_BASELINES = True 
if USE_BASELINES:
    from policy_hooks.baselines.argparse import argsparser as baseline_argsparser


DIR_KEY = 'experiment_logs/'

def get_dir_name(base, no, nt, ind, descr, args=None):
    dir_name = base + 'objs{0}_{1}/{2}'.format(no, nt, descr)
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
            next_config['base_dir'] = next_config['weight_dir']
            next_config['server_id'] = '{0}'.format(str(random.randint(0, 2**16)))
            next_config['mp_server'] = True
            next_config['pol_server'] = True
            next_config['mcts_server'] = True
            next_config['use_local'] = True
            next_config['log_server'] = False
            next_config['view_server'] = False
            next_config['use_local'] = True
            next_config['log_timing'] = False
            next_config['args'] = args
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
    config = config_module.refresh_config(args.nobjs, args.nobjs)
    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds']
    config['common']['num_conds'] = config['num_conds']
    config['algorithm']['conditions'] = config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    config['weight_dir'] = get_dir_name(next_config['base_weight_dir'], next_config['num_objs'], next_config['num_targs'], i, next_config['descr'], args)
    #config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs'])
    next_config['base_dir'] = config['weight_dir']
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


def run_baseline(args):
    # Retrieve previous information to match how the expert was trained
    exps_info = [[args.config]]
    n_objs = args.nobjs if args.nobjs > 0 else None
    n_targs = args.nobjs if args.nobjs > 0 else None
    #n_targs = args.ntargs if args.ntargs > 0 else None
    if USE_BASELINES and len(args.expert_path):
        sys.path.insert(1, args.expert_path)
        exps_info = [['hyp']]
        with open(args.expert_path+'/args.pkl', 'rb') as f:
            prev_args = pickle.load(f)
        args.add_obs_delta = prev_args.add_obs_delta
        args.hist_len = prev_args.hist_len
        args.add_action_hist = prev_args.add_action_hist
    exps = load_multi(exps_info, n_objs, n_targs, args)
    config = exps[0][0][0]
    config['source'] = args.config
    #current_id = setup_dirs(config, args)
    baseline = args.baseline

    old_dir = config['weight_dir']
    old_file = config['task_map_file']
    config = {'args': args, 'task_map_file': old_file}
    config.update(vars(args))
    config['source'] = exps_info[0][0]
    config['weight_dir'] = old_dir
    current_id = 0 # setup_dirs(c, args)

    if baseline.lower() == 'stable':
        from policy_hooks.baselines.stable import run
        run(config=config)

    elif baseline.lower() == 'hbaselines':
        from policy_hooks.baselines.hbaselines import run
        run(config=config)

    elif baseline.lower() == 'gail':
        from policy_hooks.baselines.gail import run, eval_ckpts
        config['id'] = 0
        if config['task'] == 'evaluate':
            args = config['args']
            eval_ckpts(config, sub_dirs, args.episode_timesteps)
        else:
            run(config=config)
            print('Finished GAIL train')

    elif baseline.lower() == 'hiro':
        from policy_hooks.baselines.hbaselines import run
        config['id'] = 0
        run(config=config)

    elif baseline.lower() == 'example':
        from policy_hooks.baselines.example import run
        config['id'] = 0
        run(config=config)
    else:
        raise NotImplementedError

    sys.exit(0)


def setup_dirs(c, args):
    current_id = 0 if c.get('index', -1) < 0 else c['index']
    if c.get('index', -1) < 0:
        while os.path.isdir(DIR_KEY+c['weight_dir']+'_'+str(current_id)):
            current_id += 1
    c['group_id'] = current_id
    c['weight_dir'] = c['weight_dir']+'_{0}'.format(current_id)
    dir_name = ''
    sub_dirs = [DIR_KEY] + c['weight_dir'].split('/')

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception as e:
        rank = 0
    if rank < 0: rank = 0

    c['rank'] = rank
    if rank == 0:
        for d_ind, d in enumerate(sub_dirs):
            dir_name += d + '/'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
        if args.hl_retrain:
            src = DIR_KEY + args.hl_data + '/hyp.py'
        elif len(args.expert_path):
            src = args.expert_path+'/hyp.py'
        else:
            src = c['source'].replace('.', '/')+'.py'
        shutil.copyfile(src, DIR_KEY+c['weight_dir']+'/hyp.py')
        with open(DIR_KEY+c['weight_dir']+'/__init__.py', 'w+') as f:
            f.write('')
        with open(DIR_KEY+c['weight_dir']+'/args.pkl', 'wb+') as f:
            pickle.dump(args, f, protocol=0)
        with open(DIR_KEY+c['weight_dir']+'/args.txt', 'w+') as f:
            f.write(str(vars(args)))
    else:
        time.sleep(0.1) # Give others a chance to let base set up dirs
    return current_id


def main():
    args = argsparser()
    if args.run_baseline:
        run_baseline(args)

    exps = None
    if args.file == "":
        exps = [[args.config]]

    print(('LOADING {0}'.format(args.file)))
    if exps is None:
        exps = []
        with open(args.file, 'r+') as f:
            exps = eval(f.read())
    exps_info = exps
    n_objs = args.nobjs if args.nobjs > 0 else None
    n_targs = args.nobjs if args.nobjs > 0 else None
    #n_targs = args.ntargs if args.ntargs > 0 else None
    if len(args.test):
        sys.path.insert(1, DIR_KEY+args.test)
        exps_info = [['hyp']]
        old_args = args
        with open(DIR_KEY+args.test+'/args.pkl', 'rb') as f:
            args = pickle.load(f)
        args.soft_eval = old_args.soft_eval
        args.test = old_args.test
        args.use_switch = old_args.use_switch
        args.ll_policy = args.test
        args.hl_policy = args.test
        args.load_render = old_args.load_render
        args.eta = old_args.eta
        args.descr = old_args.descr
        args.easy = old_args.easy
        var_args = vars(args)
        old_vars = vars(old_args)
        for key in old_vars:
            if key not in var_args: var_args[key] = old_vars[key]
    if args.hl_retrain:
        sys.path.insert(1, DIR_KEY+args.hl_data)
        exps_info = [['hyp']]

    exps = load_multi(exps_info, n_objs, n_targs, args)
    for ind, exp in enumerate(exps):
        mains = []
        for ind2, (c, cm) in enumerate(exp):
            #if len(args.test):
            #    # c['weight_dir'] = args.test
            #    m = MultiProcessMain(c)
            #    m.run_test(c)
            #    continue

            print('\n\n\n\n\n\nLOADING NEXT EXPERIMENT\n\n\n\n\n\n')
            old_dir = c['weight_dir']
            old_file = c['task_map_file']
            c = {'args': args, 'task_map_file': old_file}
            c.update(vars(args))
            c['source'] = exps_info[ind][ind2]
            c['weight_dir'] = old_dir
            current_id = 0 # setup_dirs(c, args)

            if args.hl_retrain:
                m = MultiProcessMain(c, load_at_spawn=False)
                m.monitor = False # If true, m will wait to finish before moving on
                m.group_id = current_id
                m.hl_only_retrain()
            elif len(args.test):
                m = MultiProcessMain(c, load_at_spawn=False)
                m.run_test(m.config)
            else:
                m = MultiProcessMain(c, load_at_spawn=True)
                m.monitor = False # If true, m will wait to finish before moving on
                m.group_id = current_id
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


def argsparser():
    parser = argparse.ArgumentParser()

    # General setup
    parser.add_argument('-c', '--config', type=str, default='config')
    parser.add_argument('-test', '--test', type=str, default='')
    parser.add_argument('-no', '--nobjs', type=int, default=1)
    parser.add_argument('-nt', '--ntargs', type=int, default=1)
    parser.add_argument('-motion', '--num_motion', type=int, default=16)
    parser.add_argument('-task', '--num_task', type=int, default=16)
    parser.add_argument('-rollout', '--num_rollout', type=int, default=16)
    parser.add_argument('-num_test', '--num_test', type=int, default=0)
    parser.add_argument('-label', '--label_server', action='store_true', default=False)
    parser.add_argument('-class_label', '--classify_labels', action='store_true', default=False)
    parser.add_argument('-max_label', '--max_label', type=int, default=-1)
    parser.add_argument('-hist_len', '--hist_len', type=int, default=1)
    parser.add_argument('-task_hist_len', '--task_hist_len', type=int, default=1)
    parser.add_argument('-n_gpu', '--n_gpu', type=int, default=4)
    parser.add_argument('-obs_del', '--add_obs_delta', action='store_true', default=False)
    parser.add_argument('-act_hist', '--add_action_hist', action='store_true', default=False)
    parser.add_argument('-task_hist', '--add_task_hist', action='store_true', default=False)
    parser.add_argument('-smooth', '--traj_smooth', action='store_true', default=False)
    parser.add_argument('-seq', '--seq', action='store_true', default=False)
    parser.add_argument('-verbose', '--verbose', action='store_true', default=False)
    parser.add_argument('-backup', '--backup', action='store_true', default=False)
    parser.add_argument('-swap', '--swap', action='store_true', default=False)
    parser.add_argument('-f', '--file', type=str, default='')
    parser.add_argument('-descr', '--descr', type=str, default='')
    parser.add_argument('-save_exp', '--save_expert', action='store_true', default=False)
    parser.add_argument('-render', '--load_render', action='store_true', default=False)
    parser.add_argument('-retime', '--retime', action='store_true', default=False)
    parser.add_argument('-vel', '--velocity', type=float, default=0.3)
    parser.add_argument('-save_data', '--save_data', action='store_true', default=False)

    # Previous policy directories
    parser.add_argument('-llpol', '--ll_policy', type=str, default='')
    parser.add_argument('-hlpol', '--hl_policy', type=str, default='')
    parser.add_argument('-hldata', '--hl_data', type=str, default='')
    parser.add_argument('-hlsamples', '--hl_samples', type=str, default='')
    parser.add_argument('-ref_dir', '--reference_dir', type=str, default='')

    # NN args
    parser.add_argument('-spl', '--split_nets', action='store_false', default=True)
    parser.add_argument('-lldim', '--dim_hidden', type=int, default=32)
    parser.add_argument('-lln', '--n_layers', type=int, default=2)
    parser.add_argument('-hldim', '--prim_dim_hidden', type=int, default=32)
    parser.add_argument('-hln', '--prim_n_layers', type=int, default=2)
    parser.add_argument('-llus', '--update_size', type=int, default=2000)
    parser.add_argument('-hlus', '--prim_update_size', type=int, default=5000)
    parser.add_argument('-iters', '--train_iterations', type=int, default=50)
    parser.add_argument('-batch', '--batch_size', type=int, default=256)
    parser.add_argument('-lldec', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-hldec', '--prim_weight_decay', type=float, default=1e-3)
    parser.add_argument('-contdec', '--cont_weight_decay', type=float, default=1e-3)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-hllr', '--hllr', type=float, default=1e-3)
    parser.add_argument('-lr_policy', '--lr_policy', type=str, default='fixed')
    parser.add_argument('-split_hl', '--split_hl_loss', action='store_true', default=False)
    parser.add_argument('-image', '--add_image', action='store_true', default=False)
    parser.add_argument('-hl_image', '--add_hl_image', action='store_true', default=False)
    parser.add_argument('-cont_image', '--add_cont_image', action='store_true', default=False)
    parser.add_argument('-imwidth', '--image_width', type=int, default=64)
    parser.add_argument('-imheight', '--image_height', type=int, default=64)
    parser.add_argument('-imchannels', '--image_channels', type=int, default=3)
    parser.add_argument('-init_obs', '--incl_init_obs', action='store_true', default=False)
    parser.add_argument('-trans_obs', '--incl_trans_obs', action='store_true', default=False)

    # HL args
    parser.add_argument('-check_t', '--check_prim_t', type=int, default=1)
    parser.add_argument('-n_resample', '--n_resample', type=int, default=5)
    parser.add_argument('-ff', '--ff_thresh', type=float, default=1.)
    parser.add_argument('-hlfeed', '--hl_feedback_thresh', type=float, default=0.)
    parser.add_argument('-ff_only', '--ff_only', action='store_true', default=False)
    parser.add_argument('-hl_post', '--hl_post', action='store_true', default=False)
    parser.add_argument('-opt_wt', '--opt_wt', action='store_true', default=False)
    parser.add_argument('-l1_loss', '--l1_loss', action='store_true', default=False)
    parser.add_argument('-roll_post', '--rollout_post', action='store_true', default=False)
    parser.add_argument('-roll_ll', '--ll_rollout_opt', action='store_true', default=False)
    parser.add_argument('-roll_hl', '--hl_rollout_opt', action='store_true', default=False)
    parser.add_argument('-roll_opt', '--rollout_opt', action='store_true', default=False)
    parser.add_argument('-fail', '--train_on_fail', action='store_true', default=False)
    parser.add_argument('-failmode', '--fail_mode', type=str, default='start')
    parser.add_argument('-aughl', '--augment_hl', action='store_true', default=False)
    parser.add_argument('-x_select', '--state_select', type=str, default='base')
    parser.add_argument('-prim_decay', '--prim_decay', type=float, default=1.)
    parser.add_argument('-prim_first_wt', '--prim_first_wt', type=float, default=1e0)
    parser.add_argument('-end2end', '--end_to_end_prob', type=float, default=0.)
    parser.add_argument('-soft', '--soft', action='store_true', default=False)
    parser.add_argument('-eta', '--eta', type=float, default=5.)
    parser.add_argument('-add_noop', '--add_noop', type=int, default=0)
    parser.add_argument('-flat', '--flat', action='store_true', default=False)
    parser.add_argument('-goal_type', '--goal_type', type=str, default='default')
    parser.add_argument('-softev', '--soft_eval', action='store_true', default=False)
    parser.add_argument('-pre', '--check_precond', action='store_true', default=False)
    parser.add_argument('-post', '--check_postcond', action='store_true', default=False)
    parser.add_argument('-mid', '--check_midcond', action='store_true', default=False)
    parser.add_argument('-random', '--check_random_switch', action='store_true', default=False)
    parser.add_argument('-neg_ratio', '--perc_negative', type=float, default=0)
    parser.add_argument('-opt_ratio', '--perc_optimal', type=float, default=1.)
    parser.add_argument('-dagger_ratio', '--perc_dagger', type=float, default=1.)
    parser.add_argument('-roll_ratio', '--perc_rollout', type=float, default=0.)
    parser.add_argument('-human_ratio', '--perc_human', type=float, default=0.)
    parser.add_argument('-neg', '--negative', type=int, default=0)
    parser.add_argument('-neg_pre', '--neg_precond', action='store_true', default=False)
    parser.add_argument('-neg_post', '--neg_postcond', action='store_true', default=False)
    parser.add_argument('-dwind', '--dagger_window', type=int, default=0)
    parser.add_argument('-mask', '--hl_mask', action='store_false', default=True)
    parser.add_argument('-rs', '--rollout_seed', action='store_true', default=False)
    parser.add_argument('-switch', '--use_switch', action='store_true', default=False)
    parser.add_argument('-permute', '--permute_hl', type=int, default=0)
    parser.add_argument('-ntest', '--num_tests', type=int, default=25)
    parser.add_argument('-col_coeff', '--col_coeff', type=float, default=0.)
    parser.add_argument('-expl_eta', '--explore_eta', type=float, default=5.)
    parser.add_argument('-expl_wt', '--explore_wt', type=float, default=1.)
    parser.add_argument('-expl_n', '--explore_n', type=int, default=10)
    parser.add_argument('-expl_m', '--explore_nmax', type=int, default=1)
    parser.add_argument('-expl_suc', '--explore_success', type=int, default=5)
    parser.add_argument('-warmup', '--warmup_iters', type=int, default=300)
    parser.add_argument('-tfwarmup', '--tf_warmup_iters', type=int, default=0)

    # Old
    parser.add_argument('-her', '--her', action='store_true', default=False)
    parser.add_argument('-hindsight', '--hindsight', action='store_true', default=False)
    parser.add_argument('-e', '--expand_process', action='store_true', default=False)
    parser.add_argument('-oht', '--onehot_task', action='store_true', default=False)
    parser.add_argument('-local_retime', '--local_retime', action='store_true', default=False)
    parser.add_argument('-nocol', '--check_col', action='store_false', default=True)
    parser.add_argument('-cond', '--conditional', action='store_true', default=False)
    parser.add_argument('-easy', '--easy', action='store_true', default=False)
    parser.add_argument('-ind', '--index', type=int, default=-1)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-hlt', '--hl_timeout', type=int, default=0)
    parser.add_argument('-k', '--killall', action='store_true', default=True)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-mcts', '--mcts_server', action='store_true', default=False)
    parser.add_argument('-mp', '--mp_server', action='store_true', default=False)
    parser.add_argument('-pol', '--policy_server', action='store_true', default=False)
    parser.add_argument('-log', '--log_server', action='store_true', default=False)
    parser.add_argument('-vs', '--view_server', action='store_true', default=False)
    parser.add_argument('-all', '--all_servers', action='store_true', default=False)
    parser.add_argument('-ps', '--pretrain_steps', type=int, default=0)
    parser.add_argument('-v', '--viewer', action='store_true', default=False)
    parser.add_argument('-id', '--server_id', type=str, default='')
    parser.add_argument('-hl_retrain', '--hl_retrain', action='store_true', default=False)
    parser.add_argument('-hl_only_retrain', '--hl_only_retrain', action='store_true', default=False)
    parser.add_argument('-cur', '--curric_thresh', type=int, default=-1)
    parser.add_argument('-ncur', '--n_thresh', type=int, default=10)
    parser.add_argument('-view', '--view_policy', action='store_true', default=False)


    # Q learn args
    parser.add_argument('-qimwt', '--q_imwt', type=float, default=0)
    parser.add_argument('-q', '--use_qfunc', action='store_true', default=False)

    ## Baselines - these are passed through to other codebases
    parser.add_argument('-run_baseline', '--run_baseline', action='store_true', default=False)
    if USE_BASELINES:
        parser.add_argument('-baseline', '--baseline', type=str, default='')
        parser.add_argument('-ref_key', '--reference_keyword', type=str, default='')
        parser.add_argument('-reward_type', '--reward_type', type=str, default='binary')
        baseline_argsparser(parser)
    #parser.add_argument('-expert_path', '--expert_path', type=str, default='')
    #parser.add_argument('-n_expert', '--n_expert', type=int, default=-1)
    #parser.add_argument('--adversary_hidden_size', type=int, default=100)
    #parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    #parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    #parser.add_argument('--traj_limitation', type=int, default=-1)
    #parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    #parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    #parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    #parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    #parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    #parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    #parser.add_argument('--max_kl', type=float, default=0.01)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import multiprocessing as mp
    main()

