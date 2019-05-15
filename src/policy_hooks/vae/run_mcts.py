from policy_hooks.mcts_explore import MCTSExplore

from policy_hooks.vae.trained_envs import *


env = BlockStackEnv()
mcts = MCSExplore(env)
mcts.run()
