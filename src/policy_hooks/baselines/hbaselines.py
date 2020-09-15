import numpy as np

import gym
from gym.spaces import Box

import hbaselines
from hbaselines.algorithms import OffPolicyRLAlgorithm
from hbaselines.utils.env_util import ENV_ATTRIBUTES

from policy_hooks.agent_env_wrapper import *
from policy_hooks.utils.load_agent import *


def add_env(envtype, inds, acspace):
    ENV_ATTRIBUTES[envtype] = {
            "meta_ac_space": acspace,
            "state_indices": inds,
            "env": lambda evaluate, render, multiagent, shared, maddpg: gym.make(envtype),
            }

def run(config, mode='train'):
    args = config['args']
    envname = 'baselineEnv'
    agent = build_agent(load_agent(config))
    inds = np.concatenate([agent._prim_obs_idx[obj] for obj in objs])

    register_env(config, name=envname, max_ts=args.episode_timesteps)

    acspace = lambda relative_goals: Box(
            low=np.concatenate([[-6, -5] for _ in objs]),
            high=np.concatenate([[6, 3] for _ in objs]),
            dtype=np.float32
            )
    add_env(envname, inds, acspace)
    policy = hbaselines.goal_conditioned.td3.GoalConditionedPolicy
    seed = config['seed']
    total_steps = 1e5
    log_interval = 1000
    eval_interval = 5000
    save_interval = 5000
    initial_exploration_steps = 1000

    alg = OffPolicyRLAlgorithm(
            policy=policy,
            env=envname,
            eval_env=envname,
            policy_kwargs={
                "relative_goals": True,
                "off_policy_corrections": True,
                }
            )

    alg.learn(
            total_steps=steps,
            log_dir=config['weight_dir'],
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            initial_exploration_steps=initial_exploration_steps,
            seed=seed
            )

