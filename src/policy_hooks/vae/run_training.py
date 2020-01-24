import argparse
import imp
import importlib
import random

from policy_hooks.vae.vae_main import MultiProcessMain


def load_config(args, reload_module=None):
    config_file = args.config
    if config_file != '':
        if reload_module is not None:
            config_module = reload_module
            imp.reload(config_module)
        else:
            config_module = importlib.import_module('policy_hooks.'+config_file)
        config = config_module.config
    else:
        config_module = None
        config = {}

    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds'] if 'num_conds' in config else 1
    if 'common' in config:
        config['common']['num_conds'] = config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs'] if 'num_objs' in config else 1
    config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs']) if 'base_weight_dir' in config else args.weight_dir
    config['log_timing'] = args.timing
    config['hl_timeout'] = 0
    config['rollout_server'] = args.rollout_server or args.all_servers
    config['vae_server'] = args.vae_server or args.all_servers
    config['viewer'] = args.viewer
    config['server_id'] = args.server_id if args.server_id != '' else str(random.randint(0,2**32))
    config['n_rollout_servers'] = args.n_rollout_servers
    config['no_child_process'] = args.no_child_process
    config['rollout_len'] = args.rollout_len
    config['train_vae'] = args.train_vae
    config['unconditional'] = args.unconditional
    config['train_reward'] = args.train_reward
    config['load_step'] = args.load_step

    config['train_params'] = {
        'use_recurrent_dynamics': args.use_recurrent_dynamics,
        'use_overshooting': args.use_overshooting,
        'data_limit': args.train_samples if args.train_samples > 0 else None,
        'beta': args.beta,
        'overshoot_beta': args.overshoot_beta,
        'dist_constraint': args.dist_constraint,
    }

    return config, config_module


def load_env(args, reload_module=None):
    env_path = args.environment_path
    if reload_module is not None:
        module = reload_module
        imp.reload(module)
    else:
        module = importlib.import_module(env_path)
    env = args.environment

    return getattr(module, env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-wd', '--weight_dir', type=str, default='')
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-o', '--nobjs', type=int, default=0)
    # parser.add_argument('-ptt', '--pretrain_timeout', type=int, default=300)
    parser.add_argument('-hlt', '--hl_timeout', type=int, default=0)
    parser.add_argument('-k', '--killall', action='store_true', default=True)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-vae', '--vae_server', action='store_true', default=False)
    parser.add_argument('-sim', '--rollout_server', action='store_true', default=False)
    parser.add_argument('-all', '--all_servers', action='store_true', default=False)
    parser.add_argument('-v', '--viewer', action='store_true', default=False)
    parser.add_argument('-id', '--server_id', type=str, default='')
    parser.add_argument('-env_path', '--environment_path', type=str, default='')
    parser.add_argument('-env', '--environment', type=str, default='')
    parser.add_argument('-tamp', '--use_tamp', type=str, default='')
    parser.add_argument('-nrs', '--n_rollout_servers', type=int, default=1)
    parser.add_argument('-ncp', '--no_child_process', action='store_true', default=False)
    parser.add_argument('-rl', '--rollout_len', type=int, default=0)
    parser.add_argument('-tv', '--train_vae', action='store_true', default=False)
    parser.add_argument('-uncond', '--unconditional', action='store_true', default=False)
    parser.add_argument('-tr', '--train_reward', action='store_true', default=False)
    parser.add_argument('-loadstep', '--load_step', type=int, default=-1)

    parser.add_argument('-beta', '--beta', type=int, default=1)
    parser.add_argument('-beta_d', '--overshoot_beta', type=int, default=1)
    parser.add_argument('-nts', '--train_samples', type=int, default=-1)
    parser.add_argument('-rnn', '--use_recurrent_dynamics', action='store_true', default=False)
    parser.add_argument('-over', '--use_overshooting', action='store_true', default=False)
    parser.add_argument('-dist', '--dist_constraint', action='store_true', default=False)

    args = parser.parse_args()
    config, config_module = load_config(args)
    if args.config != '':
        main = MultiProcessMain(config)
    else:
        env_cls = load_env(args)
        main = MultiProcessMain.no_config_load(env_cls, args.environment, config)
    main.start(kill_all=args.killall)


if __name__ == '__main__':
    main()
