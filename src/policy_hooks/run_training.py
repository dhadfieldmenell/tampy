import argparse
import importlib

from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-o', '--nobjs', type=int, default=0)
    parser.add_argument('-k', '--killall', action='store_true', default=False)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)

    args = parser.parse_args()
    config_file = args.config
    config_module = importlib.import_module('policy_hooks.'+config_file)
    config = config_module.config
    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs'])
    config['log_timing'] = args.timing

    if args.pretrain:
        pretrain = MultiProcessPretrainMain(config)
        pretrain.run()

    if not args.nofull:
        main = MultiProcessMain(config)
        main.start(kill_all=args.killall)

if __name__ == '__main__':
    main()
