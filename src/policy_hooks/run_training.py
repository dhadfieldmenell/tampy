import argparse
import importlib

from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=-1)
    parser.add_argument('-i', '--niters', type=int, default=-1)
    parser.add_argument('-k', '--killall', action='store_true', default=False)

    args = parser.parse_args()
    config_file = args.config
    config_module = importlib.import_module('policy_hooks.'+config_file)
    config = config_module.config

    if args.pretrain:
        pretrain = MultiProcessPretrainMain(config)
        pretrain.run()

    if not args.nofull:
        main = MultiProcessMain(config)
        main.start(kill_all=args.killall)

if __name__ == '__main__':
    main()
