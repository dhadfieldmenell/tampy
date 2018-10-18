import argparse
import imp
import importlib

from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain


def load_config(args, reload_module=None):
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
    config['pretrain_timeout'] = args.pretrain_timeout
    config['mcts_server'] = args.mcts_server or args.all_servers
    config['mp_server'] = args.mp_server or args.all_servers
    config['pol_server'] = args.policy_server or args.all_servers

    return config, config_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-o', '--nobjs', type=int, default=0)
    parser.add_argument('-ptt', '--pretrain_timeout', type=int, default=300)
    parser.add_argument('-k', '--killall', action='store_true', default=False)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-mcts', '--mcts_server', action='store_true', default=False)
    parser.add_argument('-mp', '--mp_server', action='store_true', default=False)
    parser.add_argument('-pol', '--policy_server', action='store_true', default=False)
    parser.add_argument('-all', '--all_servers', action='store_true', default=False)

    args = parser.parse_args()
    config, config_module = load_config(args)

    if args.pretrain:
        pretrain = MultiProcessPretrainMain(config)
        pretrain.run()
        config, config_module = load_config(args, reload_module=config_module)
        print '\n\n\nPretraining Complete.\n\n\n'

    if not args.nofull:
        main = MultiProcessMain(config)
        main.start(kill_all=args.killall)

if __name__ == '__main__':
    main()
