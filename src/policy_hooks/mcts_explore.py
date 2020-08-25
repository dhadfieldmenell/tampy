from copy import copy, deepcopy
from datetime import datetime
import itertools
import numpy as np

from policy_hooks.mcts import MCTS, MCTSNode
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


class MCTSNode():
    def __init__(self, label, value, parent, nvec):
        self.label = label
        self.value = value
        self.is_leaf = True
        self.children = {}
        label_options = itertools.product(*[list(range(n)) for n in nvec])
        for option in label_options:
            self.children[option] = None
        self.parent = parent
        self.n_explored = 1.0
        self.n_child_explored = {label:0 for label in self.children}
        self.sample_links = {}
        self.sample_to_traj = {}
        self.depth = parent.depth + 1 if parent != None else 0
        if parent is not None:
            parent.add_child(self)


    def is_leaf(self):
        return self.is_leaf


    def get_task(self):
        return self.label[0]


    def update_value(self, new_value):
        # self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)
        # if new_value == 0:
        #     new_value = 1
        # else:
        #     new_value = 0

        self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)


    def update_n_explored(self):
        self.n_explored += 1


    def update_child_explored(self, child_label):
        child_label = tuple(child_label)
        self.n_child_explored[child_label] += 1


    def get_child(self, label):
        return self.children[tuple(label)]


    def add_child(self, child):
        self.children[child.label] = child
        child.parent = self
        self.is_leaf = False


    def get_explored_children(self):
        return [n for n in list(self.children.values()) if n is not None]


    def has_unexplored(self):
        for child in list(self.children.values()):
            if child is None: return True
        return False


    def __repr__(self):
        return str(self.label)


class MCTSExplore(MCTS):
    def __init__(self, env):
        self.env = env
        self.nvec = self.env.action_space.nvec
        self.root = MCTSNode(np.zeros(len(self.nvec)), 0, None, self.nvec)
        self.C = 2
        self.explore = True


    def node_check_f(self, label, state, parent):
        child = parent.get_child(label)
        if not self.explore:
            return -np.inf if child is None else child.value
        q_value = 0 if child is None else child.value
        return q_value + np.sqrt(np.log(parent.n_explored) / (1 + parent.n_child_explored[label]))


    def run(self, num_rollouts=100):
        paths = []
        for n in range(num_rollouts):
            self.env.reset()
            self.n_runs += 1
            next_path = self.simulate()
            paths.append(next_path)

        return paths


    def _simulate_from_unexplored(self, state, node, label=None, debug=False):
        # if debug:
        #     print 'Simulating from unexplored children.'

        if label is None:
            label = tuple(self.env.action_space.sample())

        next_node = MCTSNode(tuple(label),
                             0,
                             node,
                             self.env.action_space.nvec)
        path = self.simulate_from_next(next_node, state, debug=debug)
        # next_node.update_value(int(cost==0))
        node.add_child(next_node)
        # node.update_child_explored(label)
        # while node != self.root:
        #     node.update_value(int(cost==0))
        #     node = node.parent
        # return next_node

        return None, path


    def _select_from_explored(self, state, node, exclude_hl=[], label=None, debug=False):
        # if debug:
        #     print "Selecting from explored children."
        #     print "State: ", state

        if label is None:
            children = node.get_explored_children()
            children_distr = list(map(self.node_check_f, children))
        else:
            children = [node.get_child(label)]
            assert children[0] is not None
            children_distr = np.ones(1)

        next_ind = np.random.choice(list(range(len(children))), 1, p=children_distr)
        next_node = children[next_ind]
        label = next_node.label
        # plan = self.agent.plans[label]
        # if debug:
        #     print 'Chose explored child.'
        return children[next_ind], None


    def _choose_next(self, state, node, debug=False):
        # if debug:
        #     print 'Choosing next node.'
        parameterizations, values = [], []
        for label in itertools.product(list(range(self.num_tasks)), *[list(range(n)) for n in self.num_prims]):
            label = tuple(label)
            parameterizations.append(label)
            values.append(self.node_check_f(label, state, node))

        values = np.array(values)
        next_ind = np.argmax(values)
        # next_ind = np.random.choice(range(len(parameterizations)),1, p=values)
        p = parameterizations[next_ind]

        child = node.get_child(p)
        i = 0
        while child is None and not self.explore and i < len(values):
            values[next_ind] = -np.inf
            next_ind = np.argmax(values)
            p = parameterizations[next_ind]
            child = node.get_child(p)
            i += 1

        node.update_child_explored(p)

        if child is None:
            new_node, next_path = self._simulate_from_unexplored(state, node, label=p, debug=debug)
        else:
            new_node, next_path = self._select_from_explored(state, node, label=p, debug=debug)

        return new_node, next_path


    def sample(self, task, debug=False):
        # if debug:
        #     print "SAMPLING"
        samples = []
        s, success = None, False
        obs, reward, done, info = self.env.step(task, encoded=False)
        return obs, info['cur_state']


    def simulate(self, state, use_distilled=True, early_stop_prob=0.0, debug=False):
        current_node = self.root
        path = []
        samples = []

        success = True
        cur_state = state.copy()
        prev_sample = None
        terminated = False
        iteration = 0
        exclude_hl = []
        path_value = None

        path = []
        cur_obs = self.env.get_obs()
        while True:
            # if debug:
            #     print "Taking simulation step"

            current_node.update_n_explored()
            if current_node.depth >= self.max_depth:
                break

            next_node, next_path = self._choose_next(cur_state, current_node, debug=debug)
            next_node.init_state = cur_state
            if len(next_path):
                path.extend(next_path)

            if next_node == None:
                break

            label = next_node.label
            next_obs, next_state = self.sample(label)

            if next_obs is None:
                break

            path.append((label, cur_obs, self.env.check_goal())) # Repeat observations?

            current_node = next_node
            # path.append(current_node)
            # exclude_hl += [self._encode_f(cur_state, plan, self.agent.targets[self.condition])]

            iteration += 1
            # if np.random.uniform() < early_stop_prob:
            #     break
            cur_obs, cur_state = next_obs, next_state


        # if path_value is None:
        #     path_value = 1 - self.agent.goal_f(self.condition, cur_state)

        # path = []
        path_value = path[-1][-1]
        i = 0
        while current_node is not self.root:
            i += 1
            path_value = np.maximum(path_value, path[-i][-1])
            current_node.update_value(path_value)

        # path.reverse()
        return path


    def simulate_from_next(self, node, state, prev_sample, num_samples=1, save=False, exclude_hl=[], use_distilled=True, debug=False):
        # if debug:
        #     print "Running simulate from next"
        path = self._default_simulate_from_next(node.label, node.depth, node.depth, state, self.prob_func, [], num_samples, save, exclude_hl, use_distilled, [], debug=debug)
        return path


    def _default_simulate_from_next(self, label, depth, init_depth, state, prob_func, path, num_samples=1, save=True, exclude_hl=[], use_distilled=True, exclude=[], debug=False):
        # print 'Entering simulate call:', datetime.now()
        next_obs, end_state = self.sample(label)
        if next_obs is None:
            return path

        val = self.env.check_goal()
        path.append((label, next_obs, val))
        if depth >= init_depth + self.explore_depth or depth >= self.max_depth: # or hl_encoding in exclude_hl:
            return path

        next_label = []
        for n in self.nvec:
            next_label.append(np.random.choice(list(range(n))))

        next_label = tuple(next_label)
        return self._default_simulate_from_next(next_label, depth+1, init_depth, end_state, prob_func, samples, num_samples=num_samples, save=False, use_distilled=use_distilled, debug=debug)
