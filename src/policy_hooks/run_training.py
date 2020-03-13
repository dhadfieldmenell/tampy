import argparse
import copy
import imp
import importlib
import random
import sys
import time

import rospy
from std_msgs.msg import Float32MultiArray, String

from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain


TIME_LIMIT = 3600 # Time allowed per experiment batch

def load_multi(exp_list):
    exps = []
    for exp in exp_list:
        configs = []
        for i in range(len(exp)):
            c = exp[i]
            config_module = importlib.import_module('policy_hooks.'+c)
            next_config = config_module.config.copy()
            if 'num_targs' in next_config:
                next_config['weight_dir'] = next_config['base_weight_dir'] + 'objs{0}_{1}/exp_id{2}'.format(next_config['num_objs'], next_config['num_targs'], i)
            else:
                next_config['weight_dir'] = next_config['base_weight_dir'] + 'objs{0}/exp_id{1}'.format(next_config['num_objs'], i)
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
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-o', '--nobjs', type=int, default=0)
    # parser.add_argument('-ptt', '--pretrain_timeout', type=int, default=300)
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
    parser.add_argument('-f', '--file', type=str, default='')

    args = parser.parse_args()

    if args.file == "":
        config, config_module = load_config(args)

    else:
        print('LOADING {0}'.format(args.file))
        current_id = 0
        exps = []
        with open(args.file, 'r+') as f:
            exps = eval(f.read())
        exps = load_multi(exps)
        for exp in exps:
            mains = []
            for c, cm in exp:
                print('\n\n\n\n\n\nLOADING NEXT EXPERIMENT\n\n\n\n\n\n')
                c['group_id'] = current_id
                c['weight_dir'] = c['weight_dir']+'{0}'.format(current_id)
                m = MultiProcessMain(c)
                m.monitor = False # If true, m will wait to finish before moving on
                m.group_id = current_id
                current_id += 1
                with open('tf_saved/'+c['weight_dir']+'/exp_info.txt', 'w+') as f:
                    f.write(str(cm))
                
                m.start()
                mains.append(m)
                time.sleep(1)
            active = True
            
            start_t = time.time()
            while active:
                time.sleep(60.)
                print('RUNNING...')
                active = False
                for m in mains:
                    p_info = m.check_processes()
                    print('PINFO {0}'.format(p_info))
                    active = active or any([code is None for code in p_info])
                    m.expand_rollout_servers()

                if not active:
                    for m in mains:
                        m.kill_processes()

        print('\n\n\n\n\n\n\n\nEXITING')
        sys.exit(0)

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
