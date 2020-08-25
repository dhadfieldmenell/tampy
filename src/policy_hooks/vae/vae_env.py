import numpy as np

from gym import spaces

from policy_hooks.agent_env_wrapper import AgentEnvWrapper
from policy_hooks.vae.vae import VAE


class VAEEnvWrapper(AgentEnvWrapper):
    def __init__(self, config, agent=None, env=None, use_solver=False):
        assert agent is not None or env is not None
        self.agent = agent
        self.use_solver = use_solver
        self.seed = 1234
        self.sub_env = agent.mjc_env if env is None else env

        # VAE specific
        self.sub_env.im_height = 80
        self.sub_env.im_wid = 107

        self.task_options = agent.prob.get_prim_choices() if agent is not None else {}
        self.num_tasks = len(agent.task_list) if agent is not None \
                         else env.action_space.n if hasattr(env.action_space, 'n') \
                         else env.action_space.nvec[0]
        self.tol = 0.03
        self.init_state = agent.x0[0] if agent is not None else env.physics.data.qpos.copy()
        self.action_space = env.action_space if env is not None else spaces.MultiDiscrete([len(opts) for opts in list(self.task_options.values())])
        self.observation_space = spaces.Box(0, 255, [self.sub_env.im_height, self.sub_env.im_wid, 3], dtype='uint8')
        self.cur_state = env.physics.data.qpos.copy() if env is not None else self.agent.x0[0]
        self.goal = None
        config['vae']['data_read_only'] = True
        self.vae = VAE(config['vae'])
        self.vae.fit_prior()
        self.cur_latent_obs = self.vae.get_latents([self.render()])
        self._t = 0
        self.max_t = 20
        self.reset_goal()
        self.reset()


    def step(self, action, real_act=False, encoded=False):
        # if encoded:
        #     action = self.decode_action(action)

        if real_act:
            obs, reward, done, info = self.sub_env.step(action)
            next_latent = self.vae.get_latents(np.array([obs]))[0]
        else:
            if self.vae.use_recurrent_dynamics:
                next_latent, self.h = self.vae.get_next_latents([self.cur_latent_obs], [self.encode_action(action)], self.h)[0]
            else:
                next_latent, _ = self.vae.get_next_latents([self.cur_latent_obs], [self.encode_action(action)])[0]

        self.cur_latent_obs = next_latent
        obs = np.r_[next_latent, self.goal_latent]
        reward = self.check_goal()
        done = False
        info = {}
        self._t += 1
        if self._t > 0:
            self.reset_goal()
            self.reset_init_state()

        return obs, reward, done, info


    def get_obs(self):
        obs = self.cur_latent_obs.copy()
        return np.r_[obs, self.goal_latent]


    def check_goal(self, test=False):
        if test:
            return self.sub_env.check_goal()

        return -np.sum((self.goal_latent-self.cur_latent_obs)**2)


    def reset(self):
        if self.agent is not None:
            self.agent.reset(0)
        self.sub_env.reset()
        self.cur_state = self.sub_env.physics.data.qpos.copy() if self.agent is None else self.agent.x0[0]
        self.render()
        self.cur_latent_obs = self.vae.get_latents([self.render()])[0]
        self._t = 0
        if self.vae.use_recurrent_dynamics:
            self.h = self.vae.zero_state


    def reset_init_state(self):
        if hasattr(self.sub_env, 'randomize_init_state'):
            self.sub_env.randomize_init_state()

        if self.agent is not None:
            self.agent.randomize_init_state()
        self.reset()


    def reset_goal(self, test=False):
        super(VAEEnvWrapper, self).reset_goal()
        if test:
            self.goal_latent = self.vae.get_latents([self.goal_obs])[0]
        else:
            self.goal_latent = self.vae.sample_prior()


    def render(self, mode='rgb_array'):
        if self.agent is not None:
            return self.sub_env.render(camera_id=self.agent.main_camera_id, mode=mode, view=False)
        else:
            return self.sub_env.render(camera_id=self.sub_env.main_camera_id, mode=mode, view=False)


    def cost_f(self, state, p, condition, active_ts=(0,0), debug=False):
        if self.agent is not None:
            return self.agent.cost_f(state, p, condition, active_ts, debug)

        return 0.
