'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import os
import logging
from mpi4py import MPI
from tqdm import tqdm
import pickle

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.adversary import TransitionClassifier

from policy_hooks.baselines.mujoco_dset import Mujoco_Dset
from policy_hooks.agent_env_wrapper import *


DIR_PREFIX = 'tf_saved/'

def run(config, mode='train'):
    # os.environ['OPENAI_LOGDIR'] = config['weight_dir']
    args = config['args']
    U.make_session(num_cpu=1).__enter__()
    env = gen_agent_env(config=config)
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    exp_path = args.expert_path
    if os.path.isdir(exp_path):
        fnames = os.listdir(args.expert_path)
        exp_path = list(filter(lambda f: f.find('exp_data.npy') > 0, fnames))
        exp_path = list(map(lambda s: args.expert_path+'/'+s, exp_path))
    fnames = os.listdir(config['expert_path'])
    expert_gen_f = lambda n: env.get_next_batch(n)
    dataset = Mujoco_Dset(expert_path=exp_path, gen_f=expert_gen_f, traj_limitation=args.traj_limitation)
    reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = args.descr 
    args.checkpoint_dir = DIR_PREFIX+config['weight_dir']
    args.log_dir = DIR_PREFIX+config['weight_dir']
    args.pretrained = False

    if mode == 'train':
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.episode_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    else:
        avg_len, avg_ret = runner(env,
                                  policy_fn,
                                  model_path,
                                  timesteps_per_batch=args.num_timesteps,
                                  number_trajs=10,
                                  stochastic_policy=args.stochastic_policy,
                                  save=args.save_sample
                                  )
    env.close()


def eval_ckpts(config, ckpt_dirs, ts=512, n_runs=5):
    print('Evaluating ckpts')
    args = config['args']
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    gym.logger.setLevel(logging.WARN)
    task_name = args.descr 
    args.checkpoint_dir = DIR_PREFIX+config['weight_dir']
    args.log_dir = DIR_PREFIX+config['weight_dir']
    args.pretrained = False
    data = []
    reuse = False
    for i, d in enumerate(ckpt_dirs):
        print('Running eval on', d)
        env = gen_agent_env(config, max_ts=args.episode_timesteps)
        env.seed(args.seed)
        if not os.path.exists(os.path.join(d, 'args.pkl')):
            print('Skipping', d)
            continue
        with open(os.path.join(d, 'args.pkl'), 'rb') as f:
            new_args = pickle.load(f)
        model_path = os.path.join(d, new_args.descr)
        try:
            for t in range(n_runs):
                print('Generating rollout batch', t)
                avg_len, avg_ret = runner(env,
                                          policy_fn,
                                          model_path,
                                          timesteps_per_batch=ts,
                                          number_trajs=1,
                                          stochastic_policy=args.stochastic_policy,
                                          save=args.save_sample,
                                          reuse=reuse
                                          )
                data.append({'num_plans': new_args.traj_limitation,
                             'success at end': avg_ret, 
                             'path length': avg_len,
                             'descr': new_args.descr,
                             'dir': d})
                reuse = True
        except:
            continue
        env.close()
    with open(os.path.join(DIR_PREFIX, config['weight_dir'], 'baseline_data.npy'), 'wb') as f:
        pickle.dump(data, f)
    return data


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, episode_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == 'trpo':
        from policy_hooks.baselines import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=episode_timesteps,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            print(new, horizon, t)
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj

