from stable_baselines import SAC, PPO2
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.vc_env import set_global_seeds, make_vec_env
from stable_baelines.vev_enc import SubprocVecEnv

from policy_hooks.agent_env_wrapper import gen_agent_env, register_env
from policy_hooks.multiprocess_main import load_config


def run(config):
    args = config['args']
    new_config, config_module = load_config(args, config)
    new_config.update(config)
    config = new_config

    alg = config.get('algo', 'sac').lower()
    n_envs = config['n_proc']

    def env_fn():
        env = gen_agent_env(config, max_ts=args.episode_timesteps)
        return env

    envname = 'TAMPEnv-v0'
    register_env(config, name=envname, max_ts=args.episode_timesteps)
    vec_env = make_vec_env(envname, n_envs, vec_env_cls=SubprocVecEnv)

    model_cls = SAC
    if alg == 'ppo2':
        from stable_baselines.common.policies import MlpPolicy, CnnPolicy
        model_cls = PPO2
    else:
        from stable_baselines.sac import MlpPolicy, CnnPolicy

    policy_cls = MlpPolicy
    if config['add_hl_image'] or config['add_image']:
        policy_cls = CnnPolicy

    model = model_cls(policy_cls, env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, log_interval=10)
    model.save(args.descr)


