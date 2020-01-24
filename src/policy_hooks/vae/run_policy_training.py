import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import A2C

from policy_hooks.vae.vae import VAE
from policy_hooks.vae.vae_env import VAEEnvWrapper


class PolicyTrainer(object):
    def __init__(self, config):
        config['vae']['data_read_only'] = True
        self.vae = VAE(config['vae'])
        self.rollout_len = config['rollout_len']
        env = config['env']()
        self.env = VAEEnvWrapper(env=sub_env, agent=None, use_solver=False)
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.model = A2C(MlpPolicy, self.env, ent_coef=0.1, verbose=0)
        self.runner = A2CRunner(self.model.env, self.model, n_steps=self.model.n_steps, gamma=self.model.gamma)

    def train_policy(self, total_timesteps=10000):
        for n in range(total_timesteps // model.n_batch + 1):
            obs, states, rewards, masks, actions, values, ep_infos, true_reward = self.runner.run()


# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=0)
# runner = A2CRunner(model.env, model, n_steps=model.n_steps, gamma=model.gamma)

# for update in range(1, total_timesteps // model.n_batch + 1):
#     # true_reward is the reward without discount
#     obs, states, rewards, masks, actions, values, ep_infos, true_reward = runner.run()
#     ep_info_buf.extend(ep_infos)
#     _, value_loss, policy_entropy = model._train_step(obs, states, rewards, masks, actions, values,
#                                                      model.num_timesteps // (model.n_batch + 1), None)


# best_mean_reward, n_steps = -np.inf, 0

# def callback(_locals, _globals):
#     """
#     Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
#     :param _locals: (dict)
#     :param _globals: (dict)
#     """
#     global n_steps, best_mean_reward
#     # Print stats every 1000 calls
#     if (n_steps + 1) % 1000 == 0:
#         # Evaluate policy training performance
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#         if len(x) > 0:
#             mean_reward = np.mean(y[-100:])
#             print(x[-1], 'timesteps')
#             print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

#             # New best model, you could save the agent here
#             if mean_reward > best_mean_reward:
#                 best_mean_reward = mean_reward
#                 # Example for saving best model
#                 print("Saving new best model")
#                 _locals['self'].save(log_dir + 'best_model.pkl')
#     n_steps += 1
#     # Returning False will stop training early
#     return True

#     # Logs will be saved in log_dir/monitor.csv
# env = Monitor(env, log_dir, allow_early_resets=True)
# def plot_results(log_folder, title='Learning Curve'):
#     """
#     plot the results

#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     x, y = ts2xy(load_results(log_folder), 'timesteps')
#     y = moving_average(y, window=50)
#     # Truncate x
#     x = x[len(x) - len(y):]

#     fig = plt.figure(title)
#     plt.plot(x, y)
#     plt.xlabel('Number of Timesteps')
#     plt.ylabel('Rewards')
#     plt.title(title + " Smoothed")
#     plt.show()

# def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="A2C",
#               reset_num_timesteps=True):

#         new_tb_log = self._init_num_timesteps(reset_num_timesteps)

#         with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
#                 as writer:
#             self._setup_learn(seed)

#             self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
#                                                     schedule=self.lr_schedule)

#             runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
#             self.episode_reward = np.zeros((self.n_envs,))
#             # Training stats (when using Monitor wrapper)
#             ep_info_buf = deque(maxlen=100)

#             t_start = time.time()
#             for update in range(1, total_timesteps // self.n_batch + 1):
#                 # true_reward is the reward without discount
#                 obs, states, rewards, masks, actions, values, ep_infos, true_reward = runner.run()
#                 ep_info_buf.extend(ep_infos)
#                 _, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
#                                                                  self.num_timesteps // (self.n_batch + 1), writer)
#                 n_seconds = time.time() - t_start
#                 fps = int((update * self.n_batch) / n_seconds)

#                 if writer is not None:
#                     self.episode_reward = total_episode_reward_logger(self.episode_reward,
#                                                                       true_reward.reshape((self.n_envs, self.n_steps)),
#                                                                       masks.reshape((self.n_envs, self.n_steps)),
#                                                                       writer, self.num_timesteps)

#                 self.num_timesteps += self.n_batch + 1

#                 if callback is not None:
#                     # Only stop training if return value is False, not when it is None. This is for backwards
#                     # compatibility with callbacks that have no return statement.
#                     if callback(locals(), globals()) is False:
#                         break
