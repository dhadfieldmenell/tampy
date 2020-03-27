from copy import copy, deepcopy
from datetime import datetime
import itertools
import numpy as np
import pprint
import random
import time

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


MAX_OPT_DEPTH = 30 # TODO: Make this more versatile


class MixedPolicy:
    def __init__(self, pol, dU, action_inds, state_inds, opt_traj, opt_strength):
        self.pol = pol
        self.dU = dU
        self.action_inds = action_inds
        self.state_inds = state_inds
        self.opt_traj = opt_traj
        self.opt_strength = opt_strength


    def act(self, X, O, t, noise):
        if self.opt_strength < 1e-2: return self.pol.act(X, O, t, noise)
        # opt_u = np.zeros(self.dU)
        # for param, attr in self.action_inds:
        #     opt_u[self.action_inds[param, attr]] = self.opt_traj[t, self.action_inds[param, attr]]

        if noise is not None:
            if len(self.pol.chol_pol_covar.shape) > 2:
                opt_u = self.opt_traj[t] + self.pol.chol_pol_covar[t].T.dot(noise)
            else:
                opt_u = self.opt_traj[t] + self.pol.chol_pol_covar.T.dot(noise)
        else:
            opt_u = self.opt_traj[t]

        assert not np.any(np.isnan(opt_u))
        if np.any(np.isnan(opt_u)):
            print('ERROR NAN IN ACTION FOR OPT', t, self.opt_strength, self.opt_traj[t])
            opt_u[np.where(np.isnan(opt_u))] = 0.
        if self.opt_strength > 1 - 1e-2: return opt_u.copy()

        return self.opt_strength * opt_u + (1 - self.opt_strength) * self.pol.act(X, O, t, noise)


class MCTSNode():
    def __init__(self, label, value, parent, num_tasks, prim_dims, tree=None):
        self.label = label
        self.value = value
        self.num_tasks = num_tasks
        self.prim_dims = prim_dims
        self.prim_order = list(prim_dims.keys())
        self.num_prims = list(prim_dims.values())
        self.is_leaf = True
        self.children = {}
        label_options = itertools.product(range(num_tasks), *[range(n) for n in self.num_prims])
        for option in label_options:
            self.children[option] = None
        self.parent = parent
        self.n_explored = 1.0
        self.n_child_explored = {label:0 for label in self.children}
        self.sample_links = {}
        self.sample_to_traj = {}
        self.depth = parent.depth + 1 if parent != None else 0
        self.tree = tree
        if parent is not None:
            parent.add_child(self)
            self.tree = parent.tree
        self.valid = True
        self.failures = {}


    def erase(self):
        self.valid = False
        self.tree = None
        self.sampke_links = {}
        self.sample_to_traj = {}
        for child in self.children.values():
            if child is not None:
                child.erase()
        self.children = {}
        self.failures = []


    def is_root(self):
        return self.parent is None or self.parent is self


    def is_leaf(self):
        return self.is_leaf


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
        if self.tree is not None: self.tree.n_explored[self.label] += 1


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
        return filter(lambda n: n is not None, list(self.children.values()))


    def has_unexplored(self):
        for child in self.children.values():
            if child is None: return True
        return False


    def __repr__(self):
        return str(self.label)


class MCTS:
    def __init__(self, tasks, prim_dims, gmms, value_f, prob_f, condition, agent, branch_factor, num_samples, num_distilled_samples, choose_next=None, sim_from_next=None, soft_decision=False, C=2e-1, max_depth=20, explore_depth=5, opt_strength=0.0, log_prefix=None, tree_id=0, curric_thresh=-1, her=False):
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.prim_dims = prim_dims
        self.prim_order = list(prim_dims.keys())
        self.num_prims = list(prim_dims.values())
        self.max_depth = max_depth
        self._max_depth = max_depth
        self.explore_depth = explore_depth
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C # Standard is to use 2 but given difficulty of finding good paths, using smaller
        self.branch_factor = branch_factor
        self.num_samples = 1
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = self._default_choose_next if choose_next is None else choose_next
        self._simulate_from_next = self._default_simulate_from_next if sim_from_next is None else sim_from_next 
        self._value_f = value_f
        self._prob_f = prob_f
        # self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self.opt_strength = opt_strength
        self.her = her
        self.curric_thresh = curric_thresh
        self.cur_curric = 1 if curric_thresh > 0 else 0
        if self.cur_curric != 0:
            self.max_depth = 3

        self.n_resets = 0
        self.reset(gmms, condition)

        label_options = list(itertools.product(range(self.num_tasks), *[range(n) for n in self.num_prims]))
        self.n_explored = {tuple(l): 0 for l in label_options}

        self.label_options = label_options
        self.log_file = log_prefix + '_paths.txt' if log_prefix is not None else None
        self.verbose_log_file = log_prefix + '_verbose.txt' if log_prefix is not None else None
        self.log_prefix = log_prefix
        if self.log_file is not None:
            init_state = []
            x = self.agent.x0[self.condition][self.agent._x_data_idx[STATE_ENUM]]
            for param_name, attr in self.agent.state_inds:
                inds = self.agent.state_inds[param_name, attr]
                if inds[-1] < len(x):
                    init_state.append((param_name, attr, x[inds]))
            with open(self.log_file, 'w+') as f:
                f.write('Data for MCTS on initial state:')
                f.write(str(init_state))
                f.write('\n\n')


    def add_log_file(self, log_prefix):
        self.log_file = log_prefix + '_paths.txt' if log_prefix is not None else None
        self.verbose_log_file = log_prefix + '_verbose.txt' if log_prefix is not None else None
        self.log_prefix = log_prefix
        if self.log_file is not None:
            init_state = []
            x = self.agent.x0[self.condition][self.agent._x_data_idx[STATE_ENUM]]
            for param_name, attr in self.agent.state_inds:
                inds = self.agent.state_inds[param_name, attr]
                if inds[-1] < len(x):
                    init_state.append((param_name, attr, x[inds]))
            with open(self.log_file, 'w+') as f:
                f.write('\n')
            with open(self.verbose_log_file, 'w+') as f:
                f.write('\n')

    
    def mark_failure(self, node, task):
        if node is self.root:
            self.root.failures[tuple(task)] = True
            if len(self.root.failures.keys()) == len(self.root.children.keys()):
                print('BAD ROOT STATE; RESETING ON {0}'.format(self.agent.x0[self.condition]))
                self.reset()


    def reset(self, gmms=None, condition=None):
        if hasattr(self, 'root'):
            self.root.erase()
        self.root = MCTSNode((-1, -1, -1), 0, None, len(self.tasks), self.prim_dims, self)
        self.root.parent = self.root
        self.gmms = gmms
        self.condition = condition if condition is not None else self.condition
        self.n_success = 0
        self.n_runs = 0
        self.n_fixed_rollouts = 0
        self.n_samples = 1
        self.val_per_run = []
        self.first_success = 3000 # TODO: Make this mor eversatile
        self.x0 = None
        self.node_history = {}
        self.start_t = time.time()
        self.n_resets += 1
        if 1.0 in self.val_per_run and self.val_per_run.find(1.0) < self.curric_thresh:
            self.cur_curric += 1
            # self.max_depth = min(self._max_depth, int(2 * self.max_depth))
            self.max_depth = min(self._max_depth, self.max_depth + 3)
        self.agent.replace_cond(self.condition, curric_step=self.cur_curric)


    def get_new_problem(self):
        self.reset()
        self.agent.replace_conditions([self.condition])


    def prob_func(self, prim_obs):
        prim_obs = prim_obs.reshape((1, -1))
        return self._prob_f(prim_obs)


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
        self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=True)

        # q_value = 0 if child is None else child.value
        # prim_obs = sample.get_prim_obs(t=0)
        val_obs = sample.get_val_obs(t=0)
        q_value = self.value_func(val_obs)[0] if child is None else child.value
        # policy_distrs = self.prob_func(prim_obs)
        # prob = np.product([policy_distrs[ind][label[ind]] for ind in range(len(label))])
        # child_explored = child.n_explored if child is not None else 0
        # return self.value_func(val_obs)[1] + self.C * np.sqrt(parent.n_explored) / (1 + child_explored)
        # return q_value + self.C * self.value_func(obs)[1] / (1 + child_explored)
        return q_value + self.C * np.sqrt(np.log(parent.n_explored) / (1 + parent.n_child_explored[label]))

        # child_value = child.value if child is not None else q_value
        # return child_value + self.C * q_value / (1 + parent.n_child_explored[label])


    def multi_node_check_f(self, labels, state, parent):
        sample = Sample(self.agent)
        sample.set_X(state.copy(), 0)
        # sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        # sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)

        # self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, labels[0], fill_obs=True)

        vals = []
        for label in labels:
            # self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=False)
            child = parent.get_child(label)
            # val_obs = sample.get_val_obs(t=0)
            # q_value = self.value_func(val_obs)[0] if child is None else child.value
            q_value = 0 if child is None else child.value
            vals.append(q_value + self.C * np.sqrt(np.log(parent.n_explored) / (1 + parent.n_child_explored[label])))
            # vals.append(q_value + \
            #             self.C * np.sqrt(np.log(parent.n_explored) / (1 + parent.n_child_explored[label])) + \
            #             self.C * np.sqrt(np.log(self.n_samples) / (1 + self.n_explored[label])))

        return vals


    def print_run(self, state, use_distilled=True):
        value, path = self.simulate(state.copy(), use_distilled, debug=False)
        print('Testing rollout of MCTS')
        for sample in path:
            task = self.tasks[np.argmax(sample.get(TASK_ENUM, t=0))]
            targ = self.agent.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]
            print(task, targ)
            print(sample.get_X())

        print('End of MCTS rollout.\n\n')


    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, fixed_paths=[], debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = -np.inf
        paths = []
        self.x0 = state

        for n in range(num_rollouts):
            self.agent.reset_to_state(state)
            # if not self.n_runs % 10 and self.n_success == 0:
            #     self.max_depth += 1
            # self.agent.reset_hist()
            # print("MCTS Rollout {0} for condition {1}.\n".format(n, self.condition))
            new_opt_val, next_path = self.simulate(state.copy(), use_distilled, fixed_paths=fixed_paths, debug=debug)
            # print("Finished Rollout {0} for condition {1}.\n".format(n, self.condition))

            opt_val = np.maximum(new_opt_val, opt_val)

            '''
            if len(next_path) and new_opt_value > 1. - 1e-3:
                paths.append(next_path)
                self.n_success += 1
                end = next_path[-1]
                print('\nSUCCESS! Tree {0}\n'.format(state))
    
        self.agent.add_task_paths(paths)
            '''

        return opt_val


    def _simulate_from_unexplored(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, label=None, debug=False):
        if debug:
            print 'Simulating from unexplored children.'

        if label is None:
            dummy_label = tuple(np.zeros(len(self.num_prims)+1, dtype='int32'))
            label = self.iter_labels(state, dummy_label)

            if label is None: 
                return 0, None, None

            precond_cost = self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug)
            if precond_cost > 0:
                return 0, None, None

        # self.agent.reset_hist(deepcopy(old_traj_hist))
        next_node = MCTSNode(tuple(label), 
                             0, 
                             node, 
                             len(self.tasks), 
                             self.prim_dims)
        value, next_sample = self.simulate_from_next(next_node, state, prev_sample, num_samples=5, use_distilled=use_distilled, save=True, exclude_hl=exclude_hl, debug=debug)
        node.add_child(next_node)
        next_node.update_value(value)
        if next_sample is not None:
            next_sample.node = next_node.parent
        # node.update_child_explored(label)
        # while node != self.root:
        #     node.update_value(int(cost==0))
        #     node = node.parent
        # return next_node

        return value, None, next_sample

    
    def _simulate_fixed_path(self, cur_state, node, task, fixed_path):
        next_node = node.get_child(task)
        if next_node is None:
            next_node = MCTSNode(tuple(task),
                                 0,
                                 node,
                                 len(self.tasks),
                                 self.prim_dims)
        plan = self.agent.plans[task]
        if fixed_path is None:
            next_sample = None
            end_state = cur_state
            value = 0
        else:
            next_sample, end_state = self.sample(task, cur_state, plan, num_samples=1, use_distilled=False, fixed_path=fixed_path, debug=False)
            value = 1 - self.agent.goal_f(self.condition, end_state)
        self.n_fixed_rollouts += 1
        return value, next_node, next_sample


    def _select_from_explored(self, state, node, exclude_hl=[], label=None, debug=False):
        if debug:
            print("Selecting from explored children.")
            print("State: ", state)

        if label is None:
            children = node.get_explored_children() 
            # children_distr = map(self.node_check_f, children)
            children_distr = self.multi_node_check_f([c.label for c in children], state, node)
        else:
            children = [node.get_child(label)]
            assert children[0] is not None
            children_distr = np.ones(1)

        next_ind = np.argmax(children_distr)
        next_node = children[next_ind]
        label = next_node.label
        plan = self.agent.plans[label]
        if self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug) == 0:
            if debug:
                print('Chose explored child.')
            plan = self.agent.plans[label]
            next_sample, next_state = self.sample(label, state, plan, self.num_samples, node=node, debug=debug)
            value = 1 - self.agent.goal_f(self.condition, next_state)
            return value, next_node, next_sample
        else:
            return 0, None, None

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


    def _default_choose_next(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        stochastic = False
        if debug:
            print('Choosing next node.')
        parameterizations, values = [], []
        for label in itertools.product(range(self.num_tasks), *[range(n) for n in self.num_prims]):
            label = tuple(label)
            parameterizations.append(label)
            # values.append(self.node_check_f(label, state, node))

        cost = 1.
        if all([node.get_child(p) is None for p in parameterizations]):
            p = self.iter_labels(state, label)
            if p is None:
                cost = 1.
            else:
                cost = 0.

        if cost > 0:
            values = self.multi_node_check_f(parameterizations, state, node)
            values = np.array(values)
            if stochastic:
                eta = 1e1
                exp_cost = np.exp(eta * (values - np.max(values)))
                exp_cost /= np.sum(exp_cost)
                ind = np.random.choice(range(len(values)), p=exp_cost)
            else:
                inds = np.array(range(len(values)))[values >= np.max(values) - 1e-3]
                ind = np.random.choice(inds)

            p = parameterizations[ind]
            values[ind] = -np.inf
            cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
            while cost > 0 and np.any(values > -np.inf):
                child = node.get_child(p)
                # if child is not None:
                #     child.update_value(0) # If precondtions are violated, this is a bad path

                inds = np.array(range(len(values)))[values == np.max(values)]
                ind = np.random.choice(inds)
                p = parameterizations[ind]
                values[ind] = -np.inf
                cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
                if cost > 0: node.failures[tuple(p)] = True
        
        child = node.get_child(p)
        node.update_child_explored(p)

        if debug:
            print('Chose to explore ', p)

        if cost > 0:
            if debug:
                print('Failed all preconditions for next nodes')
            if child is not None:
                child.update_value(0)
            return 0, None, None

        if child is None:
            return self._simulate_from_unexplored(state, node, prev_sample, exclude_hl, use_distilled, label=p, debug=debug)
        else:
            return self._select_from_explored(state, node, exclude_hl, label=p, debug=debug)


    def sample(self, task, cur_state, plan, num_samples, use_distilled=True, node=None, save=True, fixed_path=None, debug=False):
        if debug:
            print("SAMPLING")
        samples = []
        # old_traj_hist = self.agent.get_hist()
        task_name = self.tasks[task[0]]

        self.n_samples += 1
        s, success = None, False
        if not hasattr(self.rollout_policy[task_name], 'scale') or self.rollout_policy[task_name].scale is None:
            new_opt_strength = 1.
        else:
            new_opt_strength = self.opt_strength

        use = True
        pol = MixedPolicy(self.rollout_policy[task_name], self.agent.dU, self.agent.action_inds, self.agent.state_inds, None, new_opt_strength)
        if new_opt_strength > 0:
            gmm = self.gmms[task_name] if self.gmms is not None else None
            inf_f = None # if gmm is None or gmm.sigma is None else lambda s: gmm.inference(np.concatenate[s.get(STATE_ENUM), s.get_U()])
            s, failed, success = self.agent.solve_sample_opt_traj(cur_state.copy(), task, self.condition, inf_f=inf_f, targets=self.agent.target_vecs[self.condition].copy())
            pol.opt_traj = s.get(ACTION_ENUM).copy()
            s.opt_strength = new_opt_strength
            if success:
                samples.append(s)
            else:
                s = None
                pol.opt_strength = 0
                use = False

        # if success:
        #     samples.append(s)

        if fixed_path is None:
            for n in range(num_samples):
                # self.agent.reset_hist(deepcopy(old_traj_hist))
                samples.append(self.agent.sample_task(pol, self.condition, cur_state, task, noisy=(n > 0)))
                # samples.append(self.agent.sample_task(pol, self.condition, cur_state, task, noisy=True))
                if success:
                    samples[-1].set_ref_X(s.get_ref_X())
                    samples[-1].set_ref_U(s.get_ref_U())
        else:
            samples.append(self.agent.sample_optimal_trajectory(cur_state, task, self.condition, fixed_path))

        lowest_cost_sample = samples[0]
        opt_fail = False
        lowest_cost_sample.opt_strength = new_opt_strength

        # if np.random.uniform() > 0.99: print(lowest_cost_sample.get(STATE_ENUM), task)
        for s in samples:
            s.node = node
            if not use:
                s.use_ts[:] = 0.

        if save and fixed_path is None:
            self.agent.add_sample_batch(samples, task)
        cur_state = lowest_cost_sample.end_state # get_X(t=lowest_cost_sample.T-1)
        # self.agent.reset_hist(lowest_cost_sample.get_U()[-self.agent.hist_len:].tolist())

        '''
        if self.log_file is not None:
            mp_state = []
            x = cur_state[self.agent._x_data_idx[STATE_ENUM]]
            for param_name, attr in self.agent.state_inds:
                inds = self.agent.state_inds[param_name, attr]
                if inds[-1] < len(x):
                    mp_state.append((param_name, attr, x[inds]))
            cost_info = self.agent.cost_info(cur_state, task, self.condition, active_ts=(-1,-1))
            task_name = self.tasks[task[0]]
            with open(self.log_file, 'w+') as f:
                f.write('Data for MCTS after step for {0} on {1}:'.format(task_name, task))
                f.write('Using fixed path: {0}'.format(fixed_path is not None))
                f.write(str(mp_state))
                f.write(str(cost_info))
                f.write('\n\n')
        '''
        # assert not np.any(np.isnan(lowest_cost_sample.get_obs()))
        return lowest_cost_sample, cur_state


    def get_path_info(self, x0, node, task, traj):
        return [x0, node, task, traj]


    def simulate(self, state, use_distilled=False, early_stop_prob=0.0, fixed_paths=[], debug=False):
        current_node = self.root
        path = []
        samples = []

        self.n_runs += 1
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

            if len(fixed_paths) <= iteration:
                value, next_node, next_sample = self._choose_next(cur_state, current_node, prev_sample, exclude_hl, use_distilled, debug=debug)
            else:
                path_info = fixed_paths[iteration]
                value, next_node, next_sample = self._simulate_fixed_path(*path_info)
            
            # if np.random.uniform() > 0.9 and current_node is self.root: print(next_sample.get(STATE_ENUM))
            path_value = np.maximum(value, path_value)
            self.node_history[tuple(cur_state)] = current_node
    
            # print('Got nodes:', next_node, next_sample, 'n_fixed:', len(fixed_paths))
            if next_node is None or next_sample is None or next_node.depth > self.max_depth:
                break

            # if len(fixed_paths) <= iteration and np.random.uniform() > 0.9:
            #     print(next_sample.get_X(), '<---- sampled path in tree')

            next_sample.node = next_node.parent
            next_node.sample_links[next_sample] = prev_sample # Used to retrace paths
            prev_sample = next_sample
            cur_state = next_sample.end_state # get_X(t=next_sample.T-1)
            current_node = next_node
            # exclude_hl += [self._encode_f(cur_state, plan, self.agent.targets[self.condition])]

            iteration += 1
            # if np.random.uniform() < early_stop_prob:
            #     break

        if path_value is None:
            path_value = 1 - self.agent.goal_f(self.condition, cur_state)

        end_sample = next_sample 

        path = []
        while current_node is not self.root and prev_sample in current_node.sample_links:
            path.append(prev_sample)
            cur_state = prev_sample.end_state # get_X(t=prev_sample.T-1)
            path_value = np.maximum(path_value, 1-self.agent.goal_f(self.condition, cur_state))
            prev_sample.task_cost = 1-path_value
            prev_sample.success = path_value
            prev_sample = current_node.sample_links[prev_sample]
            current_node.sample_links = {}
            current_node.update_value(path_value)
            current_node = current_node.parent

        path.reverse()

        if end_sample is not None:
            if end_sample not in path:
                path.append(end_sample)
                end_sample.success = path_value
                end_sample.task_cost = 1. - path_value
            n = end_sample
            while hasattr(n, 'next_sample') and n.next_sample is not None:
                next_n = n.next_sample
                n.next_sample = None
                n = next_n
                n.success = path_value
                n.task_cost = 1. - path_value
                path.append(n)

        self.val_per_run.append(path_value)
        if len(path) and path_value > 1. - 1e-3:
            self.n_success += 1
            self.first_success = np.minimum(self.first_success, self.n_runs)
            end = path[-1]
            print('\nSUCCESS! Tree {0} {1} using fixed: {2}\n'.format(self.log_file, state, len(fixed_paths) != 0))
            self.agent.add_task_paths([path])
        elif self.her:
            old_nodes = [path[i].node for i in range(len(path))]
            for s in path:
                s.node = None
                s.next_sample = None
            new_path = self.agent.relabel_path(path)
            self.agent.add_task_paths([new_path])
            for i in range(len(path)):
                path[i].node = old_nodes[i]

        self.log_path(path, len(fixed_paths))
        return path_value, path


    def simulate_from_next(self, node, state, prev_sample, num_samples=1, save=False, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print("Running simulate from next")
        label = node.label
        value, samples = self._default_simulate_from_next(label, node.depth, node.depth, state, self.prob_func, [], num_samples, save, exclude_hl, use_distilled, [], debug=debug)
        if len(samples):
            pass
            # node.sample_links[samples[0]] = prev_sample
        else:
            samples.append(None)

        for i in range(len(samples) - 1):
            samples[i].next_sample = samples[i+1]

        return value, samples[0]


    def test_run(self, state, targets, max_t=20):
        old_opt = self.opt_strength
        self.opt_strength = 1.
        path = []
        val = 0
        l = (0,0,0)
        t = 0
        while t < max_t and val < 1-1e-2 and l is not None:
            l = self.iter_labels(state, l, targets=targets)
            plan = self.agent.plans[l]
            s = sef.sample_task(l, state, plan, 1)
            val = self.agent.goal_f(0, s.get_X(s.T-1), targets)
            t += 1
        self.opt_strength = old_opt
        return val, path


    def iter_labels(self, end_state, label, exclude=[], targets=None, debug=False):
        sample = Sample(self.agent)
        sample.set_X(end_state.copy(), t=0)
        self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=True, targets=targets)
        distrs = self.prob_func(sample.get_prim_obs(t=0))
        for d in distrs:
            for i in range(len(d)):
                d[i] = round(d[i], 3)
        next_label = []

        cost = 1.
        labels = [l for l in self.label_options]
        distr = [np.prod([distrs[i][l[i]] for i in range(len(l))]) for l in labels]
        distr = np.array(distr)
        for l in exclude:
            ind = labels.index(tuple(l))
            distr[ind] = 0.

        while cost > 0 and np.any(distr > 0): 
            next_label = []
            if self.soft_decision:
                expcost = self.n_runs * distr
                expcost = expcost - np.max(expcost)
                expcost = np.exp(expcost)
                expcost = expcost / np.sum(expcost)
                ind = np.random.choice(range(len(distr)), p=expcost)
                next_label = tuple(labels[ind])
                distr[ind] = -np.inf
            else:
                val = np.max(distr)
                inds = np.array(range(len(distr)))[distr >= val - 1e-3]
                ind = np.random.choice(inds)
                # ind = np.argmax(distr)
                next_label = tuple(labels[ind])
                distr[ind] = 0.
            cost = self.agent.cost_f(end_state, next_label, self.condition, active_ts=(0,0), debug=debug)
        if cost > 0:
            print('NO PATH FOR:', end_state, 'excluding:', exclude)
            return None
        return next_label


    def _default_simulate_from_next(self, label, depth, init_depth, state, prob_func, samples, num_samples=1, save=True, exclude_hl=[], use_distilled=True, exclude=[], debug=False):
        # print 'Entering simulate call:', datetime.now()
        task = self.tasks[label[0]]
        new_samples = []

        if depth > MAX_OPT_DEPTH:
            self.opt_strength = 0.
        # if self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition], (task, obj.name, targ.name)) in exclude_hl:
        #     return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task, obj.name]), samples 

        plan = self.agent.plans[label]
        if self.agent.cost_f(state, label, self.condition, active_ts=(0,0)) > 0:
            return 1 - self.agent.goal_f(self.condition, state), samples

        next_sample, end_state = self.sample(label, state, plan, num_samples=num_samples, save=save, use_distilled=use_distilled, debug=debug)
        if next_sample is None:
            path_value = 0 # 1 - self.agent.goal_f(self.condition, state)
            for sample in samples:
                sample.task_cost = 1-path_value
                sample.success = path_value # (path_value, 1-path_value) # SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples

        if self.check_change(next_sample):
            samples.append(next_sample)
        else:
            exclude = [label]
        path_value = 1. - self.agent.goal_f(self.condition, end_state)
        # hl_encoding = self._encode_f(end_state, self.agent.plans.values()[0], self.agent.targets[self.condition])
        # if path_value >= 1. - 1e-3 or depth >= init_depth + self.explore_depth or depth >= self.max_depth: # or hl_encoding in exclude_hl:
        if path_value >= 1. - 1e-3 or depth >= min(self.n_runs, self.max_depth): # or hl_encoding in exclude_hl:
            for sample in samples:
                sample.task_cost = 1-path_value
                sample.success = path_value # (path_value, 1-path_value) # SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples

        next_label = self.iter_labels(end_state, label, exclude=exclude)
        
        if next_label is None:
            next_label = random.choice(self.label_options)

        cost = 0 # TODO: find a better way to do this

        next_label = tuple(next_label)
        next_path_value, samples = self._default_simulate_from_next(next_label, depth+1, init_depth, end_state, prob_func, samples, num_samples=num_samples, save=True, use_distilled=use_distilled, debug=debug)
        return max(path_value, next_path_value), samples


    def get_data(self):
       data = {'x0': self.x0,
               'data': self.val_per_run,
               'n_success': self.n_success,
               'n_runs': self.n_runs,
               'goal': self.agent.get_goal(self.condition),
               'targets:': self.agent.get_target_dict(self.condition),
               'opt_strength': self.opt_strength
               } 
       return data


    def get_path_data(self, path, n_fixed=0, verbose=False):
        data = []
        for sample in path:
            X = [{(pname, attr): sample.get_X(t=t)[self.agent.state_inds[pname, attr]] for pname, attr in self.agent.state_inds if self.agent.state_inds[pname, attr][-1] < self.agent.symbolic_bound} for t in range(sample.T)]
            info = {'X': X, 'task': sample.task, 'time_from_start': time.time() - self.start_t, 'n_runs': self.n_runs, 'n_resets': self.n_resets, 'value': 1.-sample.task_cost, 'fixed_samples': n_fixed, 'root_state': self.agent.x0[self.condition], 'opt_strength': sample.opt_strength if hasattr(sample, 'opt_strength') else 'N/A'}
            if verbose:
                info['obs'] = sample.get_obs()
                info['targets'] = {tname: sample.targets[self.agent.target_inds[tname, attr]] for tname, attr in self.agent.target_inds}
            data.append(info)
        return data


    def log_path(self, path, n_fixed=0):
        if self.log_file is None: return
        with open(self.log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed)
            pp_info = pprint.pformat(info, depth=60)
            f.write(pp_info)
            f.write('\n')

        with open(self.verbose_log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed, True)
            pp_info = pprint.pformat(info, depth=60)
            f.write(pp_info)
            f.write('\n')

    def check_change(self, sample):
        return np.linalg.norm(sample.get_X(0) - sample.get_X(sample.T-1)) > 1e-1

        '''    
        tasks = []
        for sample in path:
            tasks.append((self.tasks[sample.task[0]], sample.task, 'Value: {0}'.format(sample.task_cost)))

        with open(self.log_file, 'a+') as f:
            f.write('\n\n')
            f.write('Path explored:')
            f.write(str(tasks))
            f.write('\n')
            f.write('Trajectories:')
            for sample in path:
                f.write(self.tasks[sample.task[0]])
                info = self.agent.get_trajectories(sample)
                pp_info = pprint.pformat(info, depth=60)
                f.write(pp_info)

                info = self.agent.get_target_dict(self.condition)
                pp_info = pprint.pformat(info, depth=60)
                f.write(pp_info)

                f.write('\n')
        '''

