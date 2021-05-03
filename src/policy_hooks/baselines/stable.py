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
    n_envs = config['n_proc']

    def env_fn(process_id=None):
        new_config, config_module = load_config(args, base_config)
        new_config.update(base_config)
        env = AgentEnvWrapper(config=new_config, max_ts=args.episode_timesteps, process_id=process_id)
        #env = Monitor(env, env.log_file, allow_early_resets=False,
        #              info_keywords=())
        return env

    eval_env = AgentEnvWrapper(config=config, max_ts=args.episode_timesteps, process_id='testenv')
    check_env(eval_env)
    eval_callback = EvalAgentCallback(eval_env, eval_freq=10000)
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
        model = model_cls(policy_cls, vec_env, verbose=1, cliprange_vf=-1, n_steps=eval_env.horizon)

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(args.descr)


def make_env(env_fn, rank, seed=0):
    def _init():
        env = env_fn(process_id=rank) 
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


class EvalAgentCallback(EventCallback):
    def __init__(self, eval_env,
                 callback_on_new_best=None,
                 n_eval_episodes=5,
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
        self.base_env = eval_env.envs[0] if hasattr(eval_env, 'envs') else eval_env
        self._last_call = 0.

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

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

            self.base_env._goal = []
            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)
            print('Finished eval in {} seconds with mean reward {}, saving...'.format(time.time() - init_t, np.mean(episode_rewards)))
            goals = self.base_env._goal
            self.base_env._goal = []
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            for ind, rew in enumerate(episode_rewards):
                self.base_env.add_test_info(rew, goals[ind])
            self.base_env.save_log()

        return True


