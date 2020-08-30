import numpy as np

from gym import Env
from gym import spaces

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *



def gen_agent_env(config):
    config['weight_dir'] = config['base_weight_dir'] + '_{0}'.format(config['current_id'])
    m = MultiProcessMain(config)
    agent = m.agent
    env = AgentEnvWrapper(agent=agent)
    return env


class AgentEnvWrapper(Env):
    metadata = {'render.modes': ['rgb_array', 'human']}
    def __init__(self, agent=None, env=None, use_solver=False, seed=1234):
        assert agent is not None or env is not None
        self.agent = agent
        self.use_solver = use_solver
        self.seed = seed
        self.sub_env = agent.mjc_env if env is None else env

        # VAE specific
        self.sub_env.im_height = 80
        self.sub_env.im_wid = 107

        self.task_options = agent.prob.get_prim_choices() if agent is not None else {}
        self.num_tasks = len(agent.task_list) if agent is not None \
                         else env.action_space.n if hasattr(env.action_space, 'n') \
                         else env.action_space.nvec[0]
        self.tol = 0.03
        if self.agent is None:
            self.action_space = env.action_space if env is not None else spaces.MultiDiscrete([len(opts) for opts in list(self.task_options.values())])
            self.observation_space = spaces.Box(0, 255, [self.sub_env.im_height, self.sub_env.im_wid, 3], dtype='uint8')
        else:
            self.action_space = spaces.Box(-10, 10, [self.agent.dU], dtype='float32')
            self.observation_space = spaces.Box(-1e3, 1e3, [self.agent.dO], dtype='float32')
        self.cur_state = env.physics.data.qpos.copy() if env is not None else self.agent.x0[0]
        if self.agent is None or self.agent.master_config['load_render']:
            self.render()
            self.render()

        self.expert_paths = []


    def encode_action(self, task):
        if self.agent is not None:
            # act = np.zeros(self.num_tasks*np.prod([len(opts) for opts in self.task_options.values()]))
            # act = np.zeros((self.num_tasks+np.sum([len(opts) for opts in self.task_options.values()])))
            # act[task[0]] += 1.
            # cur_ind = self.num_tasks

            act = np.zeros(np.sum([len(opts) for opts in list(self.task_options.values())]))
            cur_ind = 0
            for i, opt in enumerate(self.task_options.keys()):
                act[cur_ind+task[i]] = 1.
                cur_ind += len(self.task_options[opt])
        elif type(task) is not int:
            act = np.zeros(np.sum(self.sub_env.action_space.nvec))
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
            obs = np.array(obs).reshape(self.sub_env.im_height, self.sub_env.im_wid, 3)
        elif self.agent is not None:
            x = self.agent.get_state()
            self.agent.run_policy_step(action, x)
            x = self.agent.get_state()
            s = Sample(self.agent)
            self.agent.fill_sample(0, s, x[self.agent._x_data_idx[STATE_ENUM]], 0, list(self.agent.plans.keys())[0], fill_obs=True)
            obs = s.get_prim_obs()
        else:
            obs, _, _, _ = self.sub_env.step(action)
            x = self.sub_env.physics.data.qpos.copy()
            obs = np.array(obs).reshape(self.sub_env.im_height, self.sub_env.im_wid, 3)

        info = {'cur_state': x}
        self.cur_state = x
        reward = self.check_goal(x)
        done = reward > 0.99
        return obs, reward, done, info


    def reset_goal(self):
        if self.agent is not None:
            self.agent.replace_targets()
            self.agent.set_to_targets()
            if self.agent.master_config['load_render']:
                obs = self.render()
        else:
            obs = self.sub_env.reset_goal()
        self.reset()
        self.goal_obs = obs


    def check_goal(self, x):
        if self.agent is not None:
            return self.agent.goal_f(0, x)
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
            self.agent.replace_cond(0)
            self.agent.reset(0)
        else:
            self.sub_env.reset()
        self.cur_state = self.sub_env.physics.data.qpos.copy() if self.agent is None else self.agent.x0[0]
        if self.agent is not None:
            x = self.agent.get_state()
            s = Sample(self.agent)
            self.agent.fill_sample(0, s, x[self.agent._x_data_idx[STATE_ENUM]], 0, list(self.agent.plans.keys())[0], fill_obs=True)
            obs = s.get_prim_obs()
            return obs
        else:
            return self.render()


    def reset_to_state(self, x):
        self.reset()
        self.agent.x0[0] = x
        self.agent.reset(0)


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
        elif self.agent is not None:
            self.agent.mjc_env.close()


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


    def load_data(self, fname):
        raise NotImplementedError


    def get_next_batch(self, n=0):
        while sum([s.T for s in self.expert_paths])-n <= 0:
            self.agent.replace_cond(0)
            path = self.agent.run_pr_graph(self.agent.x0[0], cond=0)
            self.agent.expert_paths.extend(path)
        if n == 0: n = sum([s.T for s in self.expert_paths])
        obs, acs = np.zeros((n, self.agent.dO)), np.zeros((n, self.agent.dU))
        t = 0
        while t < n and len(self.expert_paths):
            ind = np.random.choice(range(len(self.expert_paths)))
            s = self.expert_paths.pop(ind)
            m = min(n-t, s.T)
            inds = np.random.choice(range(s.T), m, replace=False)
            obs[t:t+m] = s.get_prim_obs()[:,inds]
            acs[t:t+m] = s.get(ACTION_ENUM)[:,inds]
            t += s.T
        return obs, acs

