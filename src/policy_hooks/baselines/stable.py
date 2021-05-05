import numpy as np
import os
import time

from stable_baselines import SAC, PPO2
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines.common.cmd_util import make_vec_env, set_global_seeds 
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, sync_envs_normalization

from policy_hooks.agent_env_wrapper import AgentEnvWrapper, gen_agent_env, register_env
from policy_hooks.multiprocess_main import load_config, setup_dirs, DIR_KEY


def run(config):
    args = config['args']
    setup_dirs(config, args)
    if not os.path.exists(DIR_KEY+config['weight_dir']):
        os.makedirs(DIR_KEY+config['weight_dir'])

    base_config = config
    new_config, config_module = load_config(args, config)
    new_config.update(config)
    config = new_config

    alg = config.get('algo', 'sac').lower()
    n_envs = config['n_proc'] if alg in ['ppo2'] else 1

    def env_fn(process_id=None):
        new_config, config_module = load_config(args, base_config)
        new_config.update(base_config)
        env = AgentEnvWrapper(config=new_config, max_ts=args.episode_timesteps, process_id=process_id)
        #env = Monitor(env, env.log_file, allow_early_resets=False,
        #              info_keywords=())
        return env

    eval_env = AgentEnvWrapper(config=config, max_ts=args.episode_timesteps, process_id='testenv')
    eval_vec_env = SubprocVecEnv([make_env(env_fn, i) for i in range(n_envs)], start_method='spawn')
    check_env(eval_env)
    eval_callback = EvalAgentCallback(eval_vec_env, eval_env, eval_freq=2048)
    #envname = 'TAMPEnv-v0'
    #register_env(config, name=envname, max_ts=args.episode_timesteps)
    vec_env = SubprocVecEnv([make_env(env_fn, i) for i in range(n_envs)])

    model_cls = PPO2
    if alg == 'sac':
        from stable_baselines.sac import MlpPolicy, CnnPolicy
        model_cls = SAC
        policy_cls = MlpPolicy
        if config['add_hl_image'] or config['add_image']:
            policy_cls = CnnPolicy
        model = model_cls(policy_cls, vec_env, verbose=1)
    elif alg == 'ddpg':
        from stable_baselines import DDPG
        from stable_baselines.ddpg.policies import MlpPolicy, CnnPolicy
        policy_cls = MlpPolicy
        if config['add_hl_image'] or config['add_image']:
            policy_cls = CnnPolicy
        n_actions = eval_env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        model_class = DDPG
        model = model_cls(policy_cls, vec_env, verbose=1)
    else:
        from stable_baselines.common.policies import MlpPolicy, CnnPolicy
        policy_cls = MlpPolicy
        if config['add_hl_image'] or config['add_image']:
            policy_cls = CnnPolicy
        model = model_cls(policy_cls, vec_env, verbose=1, cliprange_vf=-1, n_steps=512)

    test_run(eval_vec_env, model)
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(args.descr)


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
    set_global_seeds(seed)
    return _init


class EvalAgentCallback(EventCallback):
    def __init__(self, eval_env,
                 base_env,
                 callback_on_new_best=None,
                 n_eval_episodes=1,
                 eval_freq=1000,
                 log_path=None,
                 best_model_save_path=None,
                 deterministic=True,
                 render=False,
                 verbose=1):
        super(EvalAgentCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.base_env = base_env 
        self._last_call = eval_freq + 1

        # Convert to VecEnv for consistency
        #if not isinstance(eval_env, VecEnv):
        #    eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []


    def _on_step(self):
        self._last_call += 1
        if self.eval_freq > 0 and self._last_call > self.eval_freq:
            self._last_call = 0.
            init_t = time.time()
            print('Running eval...')
            sync_envs_normalization(self.training_env, self.eval_env)
            print('Time to sync {}'.format(time.time() - init_t))

            self.base_env._goal = []
            self.base_env._rews = []
            rets, rews, goals, dists, smallest_dists = test_run(self.eval_env, self.model, self.n_eval_episodes, self.base_env)
            episode_rewards = rets
            #episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
            #                                                   n_eval_episodes=self.n_eval_episodes,
            #                                                   render=False,
            #                                                   deterministic=self.deterministic,
            #                                                   return_episode_rewards=True)
            print('Finished eval for {} in {} seconds with mean reward {}, saving...'.format(self.n_eval_episodes, time.time() - init_t, np.mean(episode_rewards)))
            #goals = self.base_env._goal
            #rews = self.base_env._rews
            self.base_env._goal = []
            self.base_env._rews = []
            #mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            #mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            for ind, rew in enumerate(episode_rewards):
                self.base_env.add_test_info(rew, goals[ind], rews[ind], dists[ind], smallest_dists[ind])
            self.base_env.save_log()

        return True


