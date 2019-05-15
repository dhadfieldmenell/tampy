from gym.envs.registration import register

register(
    id='BlockSortEnv-v0',
    entry_point='policy_hooks.vae.trained_envs:BlockSortEnv',
    max_episode_steps=20,
)
