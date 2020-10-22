import numpy as np
import gym

from policy_hooks.agent_env_wrapper import *
from policy_hooks.utils.load_agent import *

def run(config, mode='train'):
    print('Running example setup...')
    args = config['args']
    agent_config = load_agent(config)
    agent = build_agent(agent_config)
    print('Built agent')
    register_env(config, 'exampleEnv-v0')
    env = gym.make('exampleEnv-v0')
    import ipdb; ipdb.set_trace()

