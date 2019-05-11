import numpy as np

from gym import spaces


class AgentEnvWrapper(object):
    def __init__(self, agent=None, env=None, use_solver=False):
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
        self.action_space = env.action_space if env is not None else spaces.MultiDiscrete([len(opts) for opts in self.task_options.values()])
        self.observation_space = spaces.Box(0, 255, [self.sub_env.im_height, self.sub_env.im_wid, 3], dtype='uint8')
        self.cur_state = env.physics.data.qpos.copy() if env is not None else self.agent.x0[0]
        self.goal = None


    def encode_action(self, task):
        if self.agent is not None:
            # act = np.zeros(self.num_tasks*np.prod([len(opts) for opts in self.task_options.values()]))
            # act = np.zeros((self.num_tasks+np.sum([len(opts) for opts in self.task_options.values()])))
            # act[task[0]] += 1.
            # cur_ind = self.num_tasks

            act = np.zeros(np.sum([len(opts) for opts in self.task_options.values()]))
            cur_ind = 0
            for i, opt in enumerate(self.task_options.keys()):
                act[cur_ind+task[i]] = 1.
                cur_ind += len(self.task_options[opt])
        elif type(task) is not int:
            act = np.zeros(np.prod(self.sub_env.action_space.nvec))
            cur_ind = 0
            for i, d in enumerate(self.sub_env.action_space.nvec):
                act[cur_ind+task[i]] = 1.
                cur_ind += d
        else:
            act = np.zeros(self.num_tasks)
            act[task] = 1.

        return act


    def decode_action(self, action):
        task = [np.argmax(action[:self.num_tasks])]
        action = action[self.num_tasks:]
        for opt in self.task_options:
            dim = len(self.task_options[opt])
            task.append(np.argmax(action[:dim]))
            action = action[dim:]
        return tuple(task)


    def step(self, action, encoded=False):
        # if encoded:
        #     action = self.decode_action(action)

        if self.use_solver:
            sample, _, success = self.agent.solve_sample_opt_traj(self.cur_state, action, condition=0)
            x = sample.get_X(sample.T-1) if success else sample.get_X(0)
            obs = self.agent.get_mjc_obs(x)
        else:
            obs, _, _, _ = self.sub_env.step(action)
            x = self.sub_env.physics.data.qpos.copy()

        obs = np.array(obs).reshape(self.sub_env.im_height, self.sub_env.im_wid, 3)

        done = False
        info = {'cur_state': x}
        self.cur_state = x
        reward = self.check_goal()
        return obs, reward, done, info


    def get_obs(self):
        return self.agent.get_mjc_obs(self.cur_state) if self.agent is not None else self.sub_env.get_obs()


    def check_goal(self):
        return 0.

        state = self.sub_env.physics.data.qpos
        success = True
        for (name, attr) in self.goal:
            if attr == 'pos':
                cur_pos = self.sub_env.get_item_pos(name, mujoco_frame=False)
                success = success and np.all(np.abs(cur_pos - self.goal[name, attr]) < self.tol)
        return 1 if success else 0


    def reset(self):
        if self.agent is not None:
            self.agent.reset(0)
        self.sub_env.reset()
        self.cur_state = self.sub_env.physics.data.qpos.copy() if self.agent is None else self.agent.x0[0]
        self.render()


    def reset_init_state(self):
        if hasattr(self.sub_env, 'randomize_init_state'):
            self.sub_env.randomize_init_state()

        if self.agent is not None:
            self.agent.randomize_init_state()


    def render(self, mode='rgb_array'):
        if self.agent is not None:
            return self.sub_env.render(camera_id=self.agent.main_camera_id, mode=mode, view=False)
        else:
            return self.sub_env.render(mode=mode, view=False)


    def close(self):
        if self.sub_env is not None:
            self.sub_env.close()


    def seed(self, seed=None):
        if seed is None:
            seed = 1234
        self.seed = seed
        return [seed]


    def get_obs(self):
        # return self.sub_env.get_obs(view=False)
        return self.render()


    def cost_f(self, state, p, condition, active_ts=(0,0), debug=False):
        if self.agent is not None:
            return self.agent.cost_f(state, p, condition, active_ts, debug)

        return 0.
