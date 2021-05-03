import numpy as np
import time

from gym import Env
from gym import spaces
from gym.envs.registration import register

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.load_agent import *


DIR_KEY = 'experiment_logs/'

def register_env(config, name='TampGym-v0', max_ts=500):
    register(
            id=name,
            entry_point='policy_hooks.agent_env_wrapper:AgentEnvWrapper',
            kwargs={
                'config':config.copy(),
                'max_ts':max_ts
                }
            )
    return name


def gen_agent_env(config=None, max_ts=500):
    env = AgentEnvWrapper(config=config, max_ts=max_ts)
    return env


class AgentEnvWrapper(Env):
    metadata = {'render.modes': ['rgb_array', 'human']}
    def __init__(self, agent=None, config=None, env=None, use_solver=False, seed=1234, max_ts=500, process_id=None):
        if process_id is None:
            process_id = str(os.getpid())
        config['id'] = process_id
        self._process_id = process_id
        if agent is None:
            agent_config = load_agent(config)
            agent = build_agent(agent_config)
        self._log_dir = DIR_KEY + config['weight_dir'] + '/'
        self._log_file = self._log_dir + 'AgentEnv{}_hl_test_log.npy'.format(self._process_id)
        self.agent = agent
        self.dummy_sample = Sample(self.agent)
        self._seed = seed
        self.sub_env = agent.mjc_env if env is None else env
        self._max_time = max_ts
        self._cur_time = 0
        self._ret = 0.
        self._goal = []
        self.horizon = agent.hor * agent.rlen
        self.start_t = time.time()
        self.n_step = 0.
        self.log_dir = DIR_KEY + config['weight_dir']
        self._rollout_data = []

        self.action_space = spaces.Box(-4, 4, [self.agent.dU], dtype='float32')
        self.observation_space = spaces.Box(-3e2, 3e2, [self.agent.dPrim], dtype='float32')
        self.cur_state = self.agent.x0[0]

        self.n_runs = 0
        self.n_suc = 0
        self.n_goal = 0
        self._reset_since_goal = True
        self._reset_since_done = True

        self.expert_paths = []


    def step(self, action):
        self.n_step += 1
        self._cur_time += 1
        x = self.agent.get_state()
        self.agent.run_policy_step(action, x)
        self.agent.fill_sample(0, self.dummy_sample, x[self.agent._x_data_idx[STATE_ENUM]], 0, list(self.agent.plans.keys())[0], fill_obs=True)
        obs = self.dummy_sample.get_prim_obs(t=0).flatten()
        self.cur_state = x
        targets = self.agent.target_vecs[0]
        reward = self.agent.reward(x, targets) / self.horizon
        goal = self.agent.goal_f(0, x, targets=targets)
        if self._reset_since_goal and goal == 0:
            self.n_goal += 1
            self._reset_since_goal = False
        done = goal == 0 or self._cur_time >= self.horizon
        if done and self._reset_since_done:
            self._goal.append(1.-goal)
            self._reset_since_done = False
            if reward > 0:
                reward *= (self.horizon - self._cur_time)
        elif not self._reset_since_done:
            reward = 0.

        self._ret += reward
        info = {'cur_state': x, 'goal': 1-goal}
        return obs, reward, done, info


    def add_test_info(self, reward, goal):
        res = [np.zeros(21)]
        next_pt = [res]
        res[0][0] = goal
        res[0][3] = time.time() - self.start_t
        res[0][18] = reward
        res[0][19] = self.agent.reward() / self.horizon
        self._rollout_data.append(next_pt)


    def save_log(self):
        np.save(self._log_file, np.array(self._rollout_data))


    def reset(self):
        self._cur_time = 0
        self._ret = 0.
        self._reset_since_goal = True
        self._reset_since_done = True
        self.agent.replace_cond(0)
        self.agent.reset(0)
        self.cur_state = self.agent.x0[0]
        x = self.agent.get_state()
        self.agent.fill_sample(0, self.dummy_sample, x[self.agent._x_data_idx[STATE_ENUM]], 0, list(self.agent.plans.keys())[0], fill_obs=True)
        obs = self.dummy_sample.get_prim_obs(t=0)
        return obs.flatten()


    def render(self, mode='rgb_array'):
        x = self.agent.get_state()
        return self.agent.get_image(x)


    def close(self):
        if self.agent is not None:
            self.agent.mjc_env.close()


    def seed(self, seed=None):
        if seed is None:
            seed = 1234
        self._seed = seed
        return [seed]


