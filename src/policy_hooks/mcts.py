from copy import copy
import numpy as np

class MCTSNode():
    def __init__(self, label, value, parent):
        self.label = label
        self.value = value
        self.children = []
        self.parent = parent
        self.n_explored = 1.0

    def is_leaf(self):
        return len(self.children) == 0

    def update_value(self, new_value):
        self.value = (self.value*(self.n_explored-1) + new_value) / self.n_explored

class MCTS:
    def __init__(self, tasks, init_cost, state_func, rollout_policy, condition, task_durations, agent, soft_decision=1.0, C=2, max_depth=10):
        self.tasks = tasks
        self.root = MCTSNode('root', -1, None)
        self.max_depth = max_depth
        self.state_func = state_func
        self.rollout_policy = rollout_policy
        self.condition = condition
        self.task_durations = task_durations
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C

    # def select_next(self, prob_func, state):
    #     current_node = self.root
    #     path = []

    #     while not current_node.is_leaf():
    #         task_distr = np.array(prob_func(state)) * map(lambda n: n.cost, current_node.children)
    #         task_distr /= np.linalg.norm(task_distr)
    #         next_task = np.random.choice(range(len(self.tasks)), p=task_distr)
    #         current_node = current_node.children[next_task]
    #         path.append(current_node)

    #     return path

    def select_next(self, prob_func, state):
        current_node = self.root
        path = []

        while not current_node.is_leaf():
            next_ind = p.argmax(map(lambda n: n.value+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored)), current_node.children)
            task_distr = np.array(prob_func(state)) * map(lambda n: n.cost, current_node.children)
            task_distr /= np.linalg.norm(task_distr)
            next_task = np.random.choice(range(len(self.tasks)), p=task_distr)
            current_node = current_node.children[next_task]
            path.append(current_node)

        return path

    def expand_next(self, node, prob_func, state):
        prob_distr = prob_func(state)
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            cost = simulate_from_next(i, state, prob_func)
            node.children.append(MCTSNode(task, cost, node))

    def simulate_to_next(self, path_to, state):
        samples = []
        for node in path_to:
            task = node.label
            sample = self.agent.sample_task(self.rollout_policy[task], self.condition, state, noisy=False)
            state = sample.get_X(sample.T-1)
            node.n_explored += 1
            samples.append(sample)

        return state, samples

    def simulate_from_next(self, task_ind, state, prob_func):
        return self._simulate_from_next(task_ind, state, prob_func, [])

    def _simulate_from_next(self, task_ind, state, prob_func, samples):
        task = self.tasks[task_ind]
        sample = self.agent.sample_task(self.rollout_policy[task], self.condition, state, noisy=False)
        duration = self.task_durations[]
        end_state = sample.get_X(sample.T-1)
        cost, terminate = self.state_func(end_state)
        samples.append(sample)
        if terminate or len(samples) == self.max_depth:
            return cost, samples

        prob_distr = prob_func(end_state)
        if np.random.sample() > self.soft_decision:
            next_task = np.random.choice(range(len(self.tasks)), prob_distr)
        else:
            next_task = np.argmax(prob_distr)

        return self._simulate_from_next(next_task, end_state, prob_func, samples)
