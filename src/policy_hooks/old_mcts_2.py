from copy import copy, deepcopy
from datetime import datetime
import numpy as np

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


class MCTSNode():
    def __init__(self, label, value, parent, num_tasks, prim_dims):
        self.label = label
        self.value = value
        self.num_tasks = num_tasks
        self.prim_dims = prim_dims
        self.prim_order = list(prim_dims.keys())
        self.num_prims = list(prim_dims.values())
        self.is_leaf = True
        self.children = [None for _ in range(num_tasks*np.prod([list(prim_dims.values())]))]
        self.parent = parent
        self.n_explored = 1.0
        self.sample_links = {}
        self.sample_to_traj = {}
        self.depth = parent.depth + 1 if parent != None else 0
        if parent is not None:
            parent.add_child(self)


    def is_leaf(self):
        return self.is_leaf()


    def get_task(self):
        return self.label[0]


    def get_prim(self, prim_name):
        return self.label[self.prim_order.index(prim_name)+1]


    def update_value(self, new_value):
        # self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)
        # if new_value == 0:
        #     new_value = 1
        # else:
        #     new_value = 0

        self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)
        self.n_explored += 1


    def get_child(self, label):
        ind = 0
        m = 1
        for i in range(len(self.prim_order), 0, -1):
            ind += m * label[i]
            m *= self.num_prims[i-1]
        ind += self.num_tasks * label[0]
        return self.children[ind]


    def add_child(self, child):
        ind = 0
        m = 1
        for i in range(len(self.prim_order), 0, -1):
            ind += m * child.label[i]
            m *= self.num_prims[i-1]
        ind += self.num_tasks * child.label[0]
        self.children[ind] = child
        self.is_leaf = False
        child.parent = self


    def get_explored_children(self):
        return [n for n in self.children if n is not None]


    def has_unexplored(self):
        for child in self.child:
            if child is None: return True
        return False


    def __repr__(self):
        return str(self.label)


class MCTS:
    def __init__(self, tasks, prim_dims, gmms, value_f, condition, agent, branch_factor, num_samples, num_distilled_samples, choose_next=None, soft_decision=1.0, C=2, max_depth=20, explore_depth=5, opt_strength=0.0):
        self.tasks = tasks
        self.prim_dims = prim_dims
        self.prim_order = list(prim_dims.keys())
        self.num_prims = list(prim_dims.values())
        self.root = MCTSNode((-1, -1, -1), 0, None, len(tasks), prim_dims)
        self.max_depth = max_depth
        self.explore_depth = explore_depth
        self.condition = condition
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C
        self.branch_factor = branch_factor
        self.num_samples = 1
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = choose_next if choose_next != None else self._default_choose_next
        self._value_f = value_f
        # self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf

        self._opt_cache = {}
        self.opt_strength = opt_strength

        self.n_success = 0
        self.n_runs = 0


    def prob_func(self, prim_obs):
        prim_obs = prim_obs.reshape((1, -1))
        return self._prob_func(prim_obs)


    def value_func(self, obs):
        obs = obs.reshape((1, -1))
        return self._value_f(obs)


    def update_vals(self, path, success):
        node = self.root
        for step in path:
            node = node.get_child(*step)
            if node is None:
                node = MCTSNode(step,
                                int(success),
                                node,
                                len(self.tasks),
                                self.prim_dims)

            else:
                node.update_value(int(success))


    def node_check_f(self, label, state, parent):
        child = parent.get_child(label)
        sample = Sample(self.agent)
        sample.set_X(state.copy(), 0)
        # sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        task_vec = np.zeros((len(self.tasks)), dtype=np.float32)
        task_vec[label[0]] = 1.
        sample.set(TASK_ENUM, task_vec, 0)
        for i in range(len(list(self.prim_dims.keys()))):
            prim = self.prim_order[i]
            vec = np.zeros(self.prim_dims[prim], dtype='float32')
            vec[label[i+1]] = 1.0
            sample.set(prim, vec, 0)

        self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=True)

        prim_obs = sample.get_prim_obs(t=0)
        val_obs = sample.get_val_obs(t=0)
        q_value = self.value_func(val_obs)[1] if child is None else child.value
        # policy_distrs = self.prob_func(prim_obs)
        # prob = np.product([policy_distrs[ind][label[ind]] for ind in range(len(label))])
        child_explored = child.n_explored if child is not None else 0
        # return self.value_func(val_obs)[1] + self.C * np.sqrt(parent.n_explored) / (1 + child_explored)
        # return q_value + self.C * self.value_func(obs)[1] / (1 + child_explored)
        return q_value + self.C * np.sqrt(np.log(parent.n_explored) / (1 + child_explored))


    def print_run(self, state, use_distilled=True):
        path = self.simulate(state.copy(), use_distilled, debug=False)
        print('Testing rollout of MCTS')
        for sample in path:
            task = self.tasks[np.argmax(sample.get(TASK_ENUM, t=0))]
            targ = self.agent.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]
            print(task, targ)
            print(sample.get_X())
        print('End of MCTS rollout.\n\n')


    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = np.inf
        paths = []
        for n in range(num_rollouts):
            self.n_runs += 1
            # if not self.n_runs % 10 and self.n_success == 0:
            #     self.max_depth += 1
            self.agent.reset_hist()
            print("MCTS Rollout {0} for condition {1}.\n".format(n, self.condition))
            next_path = self.simulate(state.copy(), use_distilled, debug=debug)
            print("Finished Rollout {0} for condition {1}.\n".format(n, self.condition))
            if len(next_path):
                end = next_path[-1]
                new_opt_value = self.agent.goal_f(self.condition, state)
                if new_opt_value == 0:
                    paths.append(next_path)
                    self.n_success += 1
                opt_val = np.minimum(new_opt_value, opt_val)

        self.agent.add_task_paths(paths)
        return opt_val


    def _simulate_from_unexplored(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, label=None, debug=False):
        if debug:
            print('Simulating from unexplored children.')
        if label is None:
            sample = Sample(self.agent)
            sample.set_X(state.copy(), 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
            dummy_label = tuple(np.zeros(len(self.num_prims)+1, dtype='int32'))
            self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, dummy_label, fill_obs=True)
            distrs = self.prob_func(sample.get_prim_obs(t=0))
            distr = np.ones(1)
            for i in range(len(distrs)):
                distr = np.outer(distr, distrs[i]).flatten()
        else:
            distr = np.zeros(len(self.tasks)*np.product(self.num_prims))
            ind = 0
            m = 1
            for i in range(len(label), 1, -1):
                ind += m * label[i-1]
                m *= self.num_prims[i-2]
            ind += m * label[0]
            distr[ind] = 1.0

        next_node = None
        old_traj_hist = self.agent.get_hist()
        while np.any(distr > 0):
            ind = np.argmax(distr)
            distr[ind] = 0.
            label = []
            m = np.product(self.num_prims)
            label.append(int(ind / m))
            ind = ind % m
            for i in range(len(self.num_prims)):
                m /= self.num_prims[i]
                label.append(int(ind / m))
                ind = ind % m
            label = tuple(label)

            if node.get_child(label) is not None:
                continue

            precond_cost = self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug)
            if precond_cost > 0:
                continue

            self.agent.reset_hist(deepcopy(old_traj_hist))
            next_node = MCTSNode(tuple(label),
                                 0,
                                 node,
                                 len(self.tasks),
                                 self.prim_dims)
            cost, _ = self.simulate_from_next(next_node, state, prev_sample, num_samples=5, use_distilled=use_distilled, save=True, exclude_hl=exclude_hl, debug=debug)
            next_node.update_value(int(cost==0))
            node.add_child(next_node)
            while node != self.root:
                node.update_value(int(cost==0))
                node = node.parent
            return None

        self.agent.reset_hist(old_traj_hist)

        if debug:
            print('Rejected all unexplored child nodes.')

        return None

    def _select_from_explored(self, state, node, exclude_hl=[], label=None, debug=False):
        if debug:
            print("Selecting from explored children.")
            print("State: ", state)

        if label is None:
            children = node.get_explored_children()
            children_distr = list(map(self.node_check_f, children))
        else:
            children = [node.get_child(label)]
            assert children[0] is not None
            children_distr = np.ones(1)

        while np.any(children_distr > -np.inf):
            next_ind = np.argmax(children_distr)
            children_distr[next_ind] = -np.inf
            next_node = children[next_ind]
            label = next_node.label
            plan = self.agent.plans[label]
            if self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug) == 0:
                if debug:
                    print('Chose explored child.')
                return children[next_ind]
        return None


    def _default_choose_next(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print('Choosing next node.')
        parameterizations, values = [], []
        for i in range(len(self.tasks) * np.product(self.num_prims)):
            ind = i
            label = []
            m = np.product(self.num_prims)
            label.append(int(ind / m))
            ind = ind % m
            for j in range(0, len(self.num_prims)):
                m /= self.num_prims[j]
                label.append(int(ind / m))
                ind = ind % m
            label = tuple(label)
            parameterizations.append(label)
            values.append(self.node_check_f(label, state, node))

        values = np.array(values)
        p = parameterizations[np.argmax(values)]
        values[np.argmax(values)] = -np.inf
        cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
        while cost > 0 and np.any(values > -np.inf):
            p = parameterizations[np.argmax(values)]
            values[np.argmax(values)] = -np.inf
            cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)

        child = node.get_child(p)
        if debug:
            print('Chose to explore ', p)
        if child is None:
            new_node = self._simulate_from_unexplored(state, node, prev_sample, exclude_hl, use_distilled, label=p, debug=debug)
        else:
            new_node = self._select_from_explored(state, node, exclude_hl, label=p, debug=debug)
        if new_node is None:
            return new_node, -np.inf
        return new_node, new_node.value


    def sample(self, task, cur_state, plan, num_samples, use_distilled=True, node=None, save=True, debug=False):
        samples = []
        old_traj_hist = self.agent.get_hist()
        task_name = self.tasks[task[0]]

        s, success = None, False
        pol = MixedPolicy(self.rollout_policy[task_name], self.agent.dU, self.agent.action_inds, self.agent.state_inds, None, self.opt_strength)
        if self.opt_strength > 0:
            gmm = self.gmms[task_name] if self.gmms is not None else None
            inf_f = None if gmm is None or gmm.sigma is None else lambda s: gmm.inference(np.concatenate[s.get(STATE_ENUM), s.get_U()])
            s, failed, success = self.agent.solve_sample_opt_traj(cur_state, task, self.condition, inf_f=inf_f)
            if success:
                pol.opt_traj = s.get(ACTION_ENUM)
            else:
                pol.opt_strength = 0

        for n in range(self.num_samples):
            self.agent.reset_hist(deepcopy(old_traj_hist))
            samples.append(self.agent.sample_task(pol, self.condition, cur_state, task, noisy=True))
            if success:
                samples[-1].set_ref_X(s.get_ref_X())
                samples[-1].set_ref_U(s.get_ref_U())

        lowest_cost_sample = samples[0]

        opt_fail = False

        if save:
            self.agent.add_sample_batch(samples, task)
        cur_state = lowest_cost_sample.get_X(t=lowest_cost_sample.T-1)
        self.agent.reset_hist(lowest_cost_sample.get_U()[-self.agent.hist_len:].tolist())

        return lowest_cost_sample, cur_state


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
        while True:
            if debug:
                print("Taking simulation step")
            if self.agent.goal_f(self.condition, state) == 0: # or current_node.depth >= self.max_depth:
                break

            next_node, _ = self._choose_next(cur_state, current_node, prev_sample, exclude_hl, use_distilled, debug=debug)

            if next_node == None:
                break

            label = next_node.label
            if self.agent.cost_f(cur_state, label, self.condition, active_ts=(0,0)) > 0:
                break

            plan = self.agent.plans[label]
            next_sample, cur_state = self.sample(label, cur_state, plan, self.num_samples, use_distilled, debug=debug)

            if next_sample is None:
                break

            current_node.sample_links[next_sample] = prev_sample # Used to retrace paths
            prev_sample = next_sample

            current_node = next_node
            path.append(current_node)
            # exclude_hl += [self._encode_f(cur_state, plan, self.agent.targets[self.condition])]

            iteration += 1
            if np.random.uniform() < early_stop_prob:
                break


        if path_value is None:
            path_value = self.agent.goal_f(self.condition, cur_state)
        path = []
        while current_node is not self.root:
            path.append(prev_sample)
            prev_sample.task_cost = path_value
            prev_sample = current_node.parent.sample_links[prev_sample]
            current_node.update_value(int(path_value==0))
            current_node = current_node.parent

        path.reverse()
        return path


    def simulate_from_next(self, node, state, prev_sample, num_samples=1, save=False, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print("Running simulate from next")
        label = node.label
        cost, samples = self._simulate_from_next(label, node.depth, node.depth, state, self.prob_func, [], num_samples, save, exclude_hl, use_distilled, [], debug=debug)
        if len(samples):
            node.sample_links[samples[0]] = prev_sample
        else:
            samples.append(None)
        return cost, samples[0]


    def _simulate_from_next(self, label, depth, init_depth, state, prob_func, samples, num_samples=1, save=True, exclude_hl=[], use_distilled=True, exclude=[], debug=False):
        # print 'Entering simulate call:', datetime.now()
        task = self.tasks[label[0]]
        new_samples = []

        # if self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition], (task, obj.name, targ.name)) in exclude_hl:
        #     return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task, obj.name]), samples

        plan = self.agent.plans[label]
        if self.agent.cost_f(state, label, self.condition, active_ts=(0,0)) > 0:
            return self.agent.goal_f(self.condition, state), samples

        next_sample, end_state = self.sample(label, state, plan, num_samples=num_samples, use_distilled=use_distilled, debug=debug)

        if next_sample is None:
            path_value = self.agent.goal_f(self.condition, state)
            for sample in samples:
                sample.task_cost = path_value
                sample.success = SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples
        samples.append(next_sample)

        path_value = self.agent.goal_f(self.condition, end_state)
        # hl_encoding = self._encode_f(end_state, self.agent.plans.values()[0], self.agent.targets[self.condition])
        if path_value == 0 or depth >= init_depth + self.explore_depth or depth >= self.max_depth: # or hl_encoding in exclude_hl:
            for sample in samples:
                sample.task_cost = path_value
                sample.success = SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples

        # exclude_hl = exclude_hl + [hl_encoding]

        sample = Sample(self.agent)
        sample.set_X(end_state.copy(), t=0)
        self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=True)
        distrs = self.prob_func(sample.get_prim_obs(t=0))
        next_label = []
        if self.soft_decision:
            for i in range(len(label)):
                if sum(distrs[i]) > 0:
                    distrs[i] = distrs[i] / sum(distrs[i])
                    next_label.append(np.random.choice(list(range(len(distrs[i]))), p=distrs[i]))
                else:
                    next_label.append(np.argmax(distrs[i]))
        else:
            for distr in distrs:
                next_label.append(np.argmax(distr))

        next_label = tuple(next_label)
        return self._simulate_from_next(next_label, depth+1, init_depth, end_state, prob_func, samples, num_samples=num_samples, save=False, use_distilled=use_distilled, debug=debug)
