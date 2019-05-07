from copy import copy, deepcopy
from datetime import datetime
import itertools
import numpy as np

from policy_hooks.mcts import MCTS, MCTSNode
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


class MixedPolicy:
    def __init__(self, pol, dU, action_inds, state_inds, opt_traj, opt_strength):
        self.pol = pol
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj
        self.opt_strength = opt_strength


    def act(self, X, O, t, noise):
        if self.opt_strength == 0: return self.pol.act(X, O, t, noise)
        opt_u = np.zeros(self.dU)
        for param, attr in self.action_inds:
            opt_u[self.action_inds[param, attr]] = self.opt_traj[t, self.action_inds[param, attr]]

        if self.opt_strength == 1: return opt_u

        return self.opt_strength * opt_u + (1 - self.opt_strength) * self.pol.act(X, O, t, noise)


class MCTSExplore(MCTS):
    def node_check_f(self, label, state, parent):
        child = parent.get_child(label)
        q_value = 0 if child is None else child.value
        return q_value + np.sqrt(np.log(parent.n_explored) / (1 + parent.n_child_explored[label]))


    def run(self, num_rollouts=50):
        paths = []
        for n in range(num_rollouts):
            self.agent.reset()
            self.n_runs += 1
            # if not self.n_runs % 10 and self.n_success == 0:
            #     self.max_depth += 1
            # self.agent.reset_hist()
            print "MCTS Rollout {0} for condition {1}.\n".format(n, self.condition)
            next_path = self.simulate(self.agent.init_state.copy())
            print "Finished Rollout {0} for condition {1}.\n".format(n, self.condition)
            paths.append(next_path)

        return paths


    def _simulate_from_unexplored(self, state, node, label=None, debug=False):
        if debug:
            print 'Simulating from unexplored children.'

        if label is None:
            task = [np.random.choice(range(len(self.tasks)))]
            for p in self.prim_dims:
                task.append(np.random.choice(range(self.prim_dims[p])))
            label = tuple(task)
            precond_cost = self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug)
            if node.get_child(label) is None and precond_cost == 0: pass

        precond_cost = self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug)
        if precond_cost > 0:
            return None

        next_node = MCTSNode(tuple(label), 
                             0, 
                             node, 
                             len(self.tasks), 
                             self.prim_dims)
        path = self.simulate_from_next(next_node, state, debug=debug)
        # next_node.update_value(int(cost==0))
        node.add_child(next_node)
        # node.update_child_explored(label)
        # while node != self.root:
        #     node.update_value(int(cost==0))
        #     node = node.parent
        # return next_node

        return next_node, path


    def _select_from_explored(self, state, node, exclude_hl=[], label=None, debug=False):
        if debug:
            print "Selecting from explored children."
            print "State: ", state

        if label is None:
            children = node.get_explored_children() 
            children_distr = map(self.node_check_f, children)
        else:
            children = [node.get_child(label)]
            assert children[0] is not None
            children_distr = np.ones(1)

        next_ind = np.random.choice(range(len(children)), 1, p=children_distr)
        next_node = children[next_ind]
        label = next_node.label
        # plan = self.agent.plans[label]
        if self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug) == 0:
            if debug:
                print 'Chose explored child.'
            return children[next_ind], None
        else:
            return None, None

        # while np.any(children_distr > -np.inf):
        #     next_ind = np.argmax(children_distr)
        #     children_distr[next_ind] = -np.inf
        #     next_node = children[next_ind]
        #     label = next_node.label
        #     plan = self.agent.plans[label]
        #     if self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug) == 0:
        #         if debug:
        #             print 'Chose explored child.'
        #         return children[next_ind]
        # return None


    def _default_choose_next(self, state, node, debug=False):
        if debug:
            print 'Choosing next node.'
        parameterizations, values = [], []
        for label in itertools.product(range(self.num_tasks), *[range(n) for n in self.num_prims]):
            label = tuple(label)
            parameterizations.append(label)
            values.append(self.node_check_f(label, state, node))

        values = np.array(values)
        # next_ind = np.argmax(values)
        next_ind = np.random.choice(range(len(parameterizations)),1, p=values)
        p = parameterizations[next_ind]
        values[next_ind] = -np.inf
        cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
        while cost > 0 and np.any(values > -np.inf):
            # next_ind = np.argmax(values)
            next_ind = np.random.choice(range(len(parameterizations)),1, p=values)
            p = parameterizations[next_ind]
            values[next_ind] = -np.inf
            cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)

        p = parameterizations[np.argmax(values)]
        child = node.get_child(p)
        node.update_child_explored(p)

        if debug:
            print 'Chose to explore ', p

        if cost > 0:
            if debug:
                print 'Failed all preconditions for next nodes'
            return None, -np.inf

        if child is None:
            new_node, next_path = self._simulate_from_unexplored(state, node, label=p, debug=debug)
        else:
            new_node, next_path = self._select_from_explored(state, node, label=p, debug=debug)

        return new_node, next_path


    def sample(self, task, cur_state, debug=False):
        if debug:
            print "SAMPLING"
        samples = []
        # old_traj_hist = self.agent.get_hist()
        task_name = self.tasks[task[0]]

        s, success = None, False
        obs, reward, done, info = self.agent.step(task, encoded=False)
        return obs, info['cur_state']

        # self.agent.reset_hist(lowest_cost_sample.get_U()[-self.agent.hist_len:].tolist())

        # if self.log_file is not None:
        #     mp_state = []
        #     x = cur_state[self.agent._x_data_idx[STATE_ENUM]]
        #     for param_name, attr in self.agent.state_inds:
        #         inds = self.agent.state_inds[param_name, attr]
        #         if inds[-1] < len(x):
        #             mp_state.append((param_name, attr, x[inds]))
        #     cost_info = self.agent.cost_info(cur_state, task, self.condition, active_ts=(-1,-1))
        #     task_name = self.tasks[task[0]]
        #     with open(self.log_file, 'w+') as f:
        #         f.write('Data for MCTS after step for {0} on {1}:'.format(task_name, task))
        #         f.write(str(mp_state))
        #         f.write(str(cost_info))
        # #         f.write('\n\n')
            
        # return lowest_cost_sample, cur_state


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
        cur_obs = self.agent.get_obs()
        while True:
            if debug:
                print "Taking simulation step"

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
            if self.agent.cost_f(cur_state, label, self.condition, active_ts=(0,0)) > 0:
                break

            plan = self.agent.plans[label]
            next_obs, next_state = self.sample(label, cur_state, plan)

            if next_obs is None:
                break

            # path.append((label, cur_obs, self.agent.check_goal())) # Repeat observations?
 
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
        if debug:
            print "Running simulate from next"
        path = self._default_simulate_from_next(node.label, node.depth, node.depth, state, self.prob_func, [], num_samples, save, exclude_hl, use_distilled, [], debug=debug)
        return path


    def _default_simulate_from_next(self, label, depth, init_depth, state, prob_func, path, num_samples=1, save=True, exclude_hl=[], use_distilled=True, exclude=[], debug=False):
        # print 'Entering simulate call:', datetime.now()
        task = self.tasks[label[0]]
        new_samples = []

        # if self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition], (task, obj.name, targ.name)) in exclude_hl:
        #     return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task, obj.name]), samples 

        if self.agent.cost_f(state, label, self.condition, active_ts=(0,0)) > 0:
            return path

        next_obs, end_state = self.sample(label, state)
        if next_obs is None:
            return path

        val = self.agent.check_goal()
        path.append((label, next_obs, val))
        if depth >= init_depth + self.explore_depth or depth >= self.max_depth or val == 1.: # or hl_encoding in exclude_hl:
            return path

        next_label = [np.random.choice(len(range(self.tasks)))]
        for p in self.prim_dims:
            next_label.append(np.random.choice(len(self.prim_dims[p])))

        next_label = tuple(next_label)
        return self._default_simulate_from_next(next_label, depth+1, init_depth, end_state, prob_func, samples, num_samples=num_samples, save=False, use_distilled=use_distilled, debug=debug)


    def log_path(self, path):
        if self.log_file is None: return
        tasks = []
        for sample in path:
            tasks.append((self.tasks[sample.task[0]], sample.task, 'Value: {0}'.format(sample.task_cost)))

        with open(self.log_file, 'a+') as f:
            f.write('Path explored:')
            f.write(str(tasks))
            f.write('\n')

