import numpy as np
import os
import time
from gym.spaces import Box

from hbaselines.algorithms import RLAlgorithm
from hbaselines.utils.env_util import ENV_ATTRIBUTES

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, sync_envs_normalization

from policy_hooks.agent_env_wrapper import AgentEnvWrapper, gen_agent_env, register_env
from policy_hooks.multiprocess_main import load_config, setup_dirs, DIR_KEY
from policy_hooks.utils.policy_solver_utils import *

def run(config):
    args = config['args']
    setup_dirs(config, args)
    if not os.path.exists(DIR_KEY+config['weight_dir']):
        os.makedirs(DIR_KEY+config['weight_dir'])
    
    log_dir = DIR_KEY + config['weight_dir']
    base_config = config
    new_config, config_module = load_config(args, config)
    new_config.update(config)
    config = new_config

    alg = config.get('algo', 'td3').lower()
    n_envs = config['n_proc']

    def env_fn(process_id=None, evaluate=False):
        new_config, config_module = load_config(args, base_config)
        new_config.update(base_config)
        if process_id is None: process_id = np.random.randint(2**20)
        env = AgentEnvWrapper(config=new_config, max_ts=args.episode_timesteps, process_id=process_id)
        #env = Monitor(env, env.log_file, allow_early_resets=False,
        #              info_keywords=())
        return env

    envname = 'TAMPGym-v0'
    register_env(config, name=envname, max_ts=args.episode_timesteps)
    train_env = AgentEnvWrapper(config=config, max_ts=args.episode_timesteps, process_id='testenv')
    eval_env = AgentEnvWrapper(config=config, max_ts=args.episode_timesteps, process_id='testenv')
    eval_vec_env = SubprocVecEnv([make_env(env_fn, i) for i in range(4)], start_method='spawn')
    check_env(eval_env)
    #vec_env = SubprocVecEnv([make_env(env_fn, i) for i in range(n_envs)])

    n_obj = config['num_objs']
    state_inds = []
    low_meta_ac, high_meta_ac = [], []
    agent = eval_env.agent
    prob = agent.prob
    if EE_ENUM in agent._prim_obs_data_idx:
        new_inds = agent._prim_obs_data_idx[EE_ENUM]
        state_inds = np.r_[state_inds, new_inds]
        low_meta_ac = np.r_[low_meta_ac, -12*np.ones(len(new_inds))]
        high_meta_ac = np.r_[high_meta_ac, 12*np.ones(len(new_inds))]

    if THETA_ENUM in agent._prim_obs_data_idx:
        new_inds = agent._prim_obs_data_idx[THETA_ENUM]
        state_inds = np.r_[state_inds, new_inds]
        low_meta_ac = np.r_[low_meta_ac, -4*np.ones(len(new_inds))]
        high_meta_ac = np.r_[high_meta_ac, 4*np.ones(len(new_inds))]

    if GRIPPER_ENUM in agent._prim_obs_data_idx:
        new_inds = agent._prim_obs_data_idx[GRIPPER_ENUM]
        state_inds = np.r_[state_inds, new_inds]
        low_meta_ac = np.r_[low_meta_ac, -np.ones(len(new_inds))]
        high_meta_ac = np.r_[high_meta_ac, np.ones(len(new_inds))]

    #if THETA_VEC_ENUM in agent._prim_obs_data_idx:
    #    new_inds = agent._prim_obs_data_idx[THETA_VEC_ENUM]
    #    state_inds = np.r_[state_inds, new_inds]
    #    low_meta_ac = np.r_[low_meta_ac, -np.ones(len(new_inds))]
    #    high_meta_ac = np.r_[high_meta_ac, np.ones(len(new_inds))]

    for n, obj in enumerate(prob.get_prim_choices()[OBJ_ENUM]):
        if OBJ_DELTA_ENUMS[n] in agent._prim_obs_data_idx:
            new_inds = agent._prim_obs_data_idx[OBJ_DELTA_ENUMS[n]]
            state_inds = np.r_[state_inds, new_inds]
            low_meta_ac = np.r_[low_meta_ac, -12*np.ones(len(new_inds))]
            high_meta_ac = np.r_[high_meta_ac, 12*np.ones(len(new_inds))]
        elif OBJ_ENUMS[n] in agent._prim_obs_data_idx:
            new_inds = agent._prim_obs_data_idx[OBJ_ENUMS[n]]
            state_inds = np.r_[state_inds, new_inds]
            low_meta_ac = np.r_[low_meta_ac, -12*np.ones(len(new_inds))]
            high_meta_ac = np.r_[high_meta_ac, 12*np.ones(len(new_inds))]

    def meta_ac_fn(relative_goals, multiagent):
        assert relative_goals, 'Not set up to use abs goals'
        return Box(low=low_meta_ac, high=high_meta_ac, dtype=np.float32)

    def state_fn(multiagent):
        return state_inds

    env_gen_fn = lambda evaluate, render, n_levels, multiagent, shared, maddpg: env_fn(evaluate=evaluate)
    ENV_ATTRIBUTES['TampGym-v0'] = {"meta_ac_space": meta_ac_fn, "state_indices": state_fn, "env": env_gen_fn}

    model_params = {"layers": [64,64]}
    if alg == 'td3':
        from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy
        model_params['noise'] = 0.03
        model_params['target_policy_noise'] = 0.05
        model_params['target_noise_clip'] = 0.2
        model_params['use_huber'] = True
    elif alg == 'ppo':
        from hbaselines.fcnet.ppo import FeedForwardPolicy
    elif alg == 'sac':
        from hbaselines.fcnet.sac import FeedForwardPolicy
        from hbaselines.goal_conditioned.sac import GoalConditionedPolicy

    alg = RLAlgorithm(
            policy=GoalConditionedPolicy,
            policy_kwargs={
                           "meta_period": 10,
                           "intrinsic_reward_type": "scaled_negative_distance",
                           "relative_goals": True,
                           "off_policy_corrections": True,
                           "model_params": model_params,
                           },
            env=train_env,
            eval_env=eval_env,
            num_envs=n_envs,
            nb_rollout_steps=5*n_envs,
            total_steps=5000000,
            nb_eval_episodes=10,
            verbose=1,
            )

    alg.learn(log_interval=10000,
              eval_interval=20000,
              save_interval=10000,
              log_dir=log_dir,
              initial_exploration_steps=20000,
              )


def test_run(eval_env, model, n_runs=1, agent=None):
    init_t = time.time()
    rews = []
    rets = []
    goals = []
    iters = []
    dists = []
    smallest_dists = []
    for _ in range(n_runs):
        obs = eval_env.reset()
        done = np.zeros(len(obs))
        ret = np.zeros(len(obs))
        cur_iter = 0
        acts = []
        xs1 = []
        xs2 = []
        xs3 = []
        obs1 = []
        state = None
        closest_dists = -np.ones(len(obs))
        while not np.all(done):
            action, state = model.predict(obs, state=state, deterministic=True)
            acts.append(np.mean(action, axis=0))
            obs, reward, next_done, _info = eval_env.step(action)
            done = np.maximum(next_done.astype(int), done)
            ret += reward
            obs1.append(obs)
            xs1.append(_info[0]['cur_state'])
            xs2.append(_info[1]['cur_state'])
            xs3.append(_info[2]['cur_state'])
            for ind, pt in enumerate(_info):
                if closest_dists[ind] < 0 or pt['distance'] < closest_dists[ind]:
                    closest_dists[ind] = pt['distance']
            cur_iter += 1

        if agent is not None:
            agent.agent.target_vecs[0] = _info[0]['targets']
            agent.agent.save_video([xs1], agent._vid_dir)
            agent.agent.target_vecs[0] = _info[1]['targets']
            agent.agent.save_video([xs2], agent._vid_dir)
            agent.agent.target_vecs[0] = _info[2]['targets']
            agent.agent.save_video([xs3], agent._vid_dir)
        avgact = np.mean(np.abs(acts), axis=0)
        iters.append(cur_iter)
        for ind in range(len(obs)):
            goals.append(_info[ind]['goal'])
            dists.append(_info[ind]['distance'])
            rews.append(reward[ind])
            rets.append(ret[ind])
            smallest_dists.append(closest_dists[ind])

    #print('\n\n\nRollout:', obs1[::10], '\n\n\n')

    print('Time to run {}, rew {}, distance {} {}, act {}'.format(time.time() - init_t, np.mean(rets), np.mean(dists), np.mean(smallest_dists), avgact))
    return rets, rews, goals, dists, smallest_dists



def make_env(env_fn, rank, seed=0):
    def _init():
        env = env_fn(process_id=rank) 
        env.seed(seed + rank)
        return env
    return _init


