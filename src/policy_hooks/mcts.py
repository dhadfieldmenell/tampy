from copy import copy, deepcopy
from datetime import datetime
import itertools
import numpy as np
import pprint
import random
import time

from pma.pr_graph import *
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


MAX_OPT_DEPTH = 30 # TODO: Make this more versatile
MCTS_WEIGHT = 10


class ixedPolicy:
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
            print(('ERROR NAN IN ACTION FOR OPT', t, self.opt_strength, self.opt_traj[t]))
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
        label_options = itertools.product(list(range(num_tasks)), *[list(range(n)) for n in self.num_prims])
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
        for child in list(self.children.values()):
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
        return [n for n in list(self.children.values()) if n is not None]


    def has_unexplored(self):
        for child in list(self.children.values()):
            if child is None: return True
        return False


    def __repr__(self):
        return str(self.label)


class MCTS:
    def __init__(self, tasks, prim_dims, gmms, value_f, prob_f, condition, agent, branch_factor, num_samples, num_distilled_samples, choose_next=None, sim_from_next=None, soft_decision=False, C=2e-1, max_depth=20, explore_depth=5, opt_strength=0.0, log_prefix=None, tree_id=0, curric_thresh=-1, n_thresh=-1, her=False, onehot_task=False, soft=False, ff_thresh=0, eta=1.):
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
        self._soft = soft
        self.eta = eta
        self.ff_thresh = ff_thresh
        self.C = C # Standard is to use 2 but given difficulty of finding good paths, using smaller
        self.branch_factor = branch_factor
        self.num_samples = 1
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = self._default_choose_next if choose_next is None else choose_next
        self._simulate_from_next = self._default_simulate_from_next if sim_from_next is None else sim_from_next
        self._value_f = value_f
        self._prob_f = prob_f
        self._switch_f = None
        self._permute = 0
        # self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self.start_t = time.time()
        self.opt_strength = opt_strength
        self.her = her
        self.onehot_task = onehot_task
        self.curric_thresh = curric_thresh
        self.n_thresh = n_thresh
        self.cur_curric = 1 if curric_thresh > 0 else 0
        if self.cur_curric != 0:
            self.max_depth = curric_thresh

        self.val_per_run = []
        self.first_suc_buf = []
        self.use_q = False
        self.discrete_prim = True
        self.n_resets = 0
        self.n_runs = 0
        self.reset(gmms, condition)
        self.first_success = self.max_depth * 50
        self.hl_suc = 0
        self.hl_fail = 0

        label_options = list(itertools.product(list(range(self.num_tasks)), *[list(range(n)) for n in self.num_prims]))
        self.n_explored = {tuple(l): 0 for l in label_options}

        self.label_options = label_options
        self.log_file = log_prefix + '_paths.txt' if log_prefix is not None else None
        self.verbose_log_file = log_prefix + '_verbose.txt' if log_prefix is not None else None
        self.log_prefix = log_prefix
        self._n_plans = None
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
            if len(list(self.root.failures.keys())) == len(list(self.root.children.keys())):
                print(('BAD ROOT STATE; RESETING ON {0}'.format(self.agent.x0[self.condition])))
                self.reset()


    def reset(self, gmms=None, condition=None):
        if hasattr(self, 'root'):
            self.root.erase()
        self.root = MCTSNode((-1, -1, -1), 0, None, len(self.tasks), self.prim_dims, self)
        self.root.parent = self.root
        self.gmms = gmms
        self.condition = condition if condition is not None else self.condition
        self.n_success = 0
        self.n_fixed_rollouts = 0
        self.n_samples = 1
        self.bad_tree = False
        self.post_cond = []
        self.prim_pre_cond = []
        self.x0 = None
        self.node_history = {}
        self.n_resets += 1
        if 1.0 in self.val_per_run:
            self.first_success = self.val_per_run.index(1.0)
            self.first_suc_buf.append(self.first_success)
            if self.agent.check_curric(self.first_suc_buf, self.n_thresh, self.curric_thresh, self.cur_curric):
                self.first_suc_buf = []
                self.cur_curric += 1
                self.max_depth = min(self._max_depth, int(2 * self.max_depth))
                print(('{0} updated curriculum to {1}'.format(self.log_file, self.cur_curric)))
            # self.max_depth = min(self._max_depth, self.max_depth + 3)
        else:
            self.first_success = self.n_runs
            self.first_suc_buf.append(max(10, self.first_success))
        self.n_runs = 0
        self.val_per_run = []
        self.agent.replace_cond(self.condition, curric_step=self.cur_curric)
        self.agent.reset(self.condition)


    def get_new_problem(self):
        self.reset()
        self.agent.replace_conditions([self.condition])


    def prob_func(self, prim_obs, soft=False):
        prim_obs = prim_obs.reshape((1, -1))
        distrs = self._prob_f(prim_obs)
        if not soft: return distrs
        out = []
        for d in distrs:
            new_d = np.zeros_like(d)
            eta = 1e-1
            exp = np.exp((d-np.max(d))/eta)
            p = exp / np.sum(exp)
            ind = np.random.choice(list(range(len(d))), p=p)
            new_d[ind] = 1.
            out.append(new_d)
        return new_d


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
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        # sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)

        # self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, labels[0], fill_obs=True)

        vals = []
        for label in labels:
            # self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=False)
            child = parent.get_child(label)
            # val_obs = sample.get_val_obs(t=0)
            if False: # self.use_q:
                self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=False)
                val_obs = sample.get_val_obs(t=0)
                q_value = self.value_func(val_obs)[0] if child is None else child.value
            else:
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
            print((task, targ))
            print((sample.get_X()))

        print('End of MCTS rollout.\n\n')


    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, fixed_paths=[], debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = -np.inf
        paths = []
        self.x0 = state

        for n in range(num_rollouts):
            self.agent.reset_to_state(state)
            new_opt_val, next_path = self.simulate(state.copy(), use_distilled, fixed_paths=fixed_paths, debug=debug)
            paths.append(next_path)

            opt_val = np.maximum(new_opt_val, opt_val)

        return opt_val, paths


    def eval_pr_graph(self, state=None, targets=None, reset=True, save=True):
        plan = None
        if targets is None:
            targets = self.agent.target_vecs[self.condition]
        if state is None:
            state, initial, goal = self.agent.sample_hl_problem()
        else:
            if self.agent.goal_f(self.condition, state) == 0:
                print(('WARNING! Init state success', state, targets))
                print(state[self.agent.state_inds['can0', 'pose']])
                plan = 'EMPTY PLAN'
            initial, goal = self.agent.get_hl_info(state, cond=self.condition, targets=targets)
            # initial = None
        if state is None: return 0, []
        domain = list(self.agent.plans.values())[0].domain
        prob = list(self.agent.plans.values())[0].prob
        for pname, attr in self.agent.state_inds:
            p = prob.init_state.params[pname]
            if p.is_symbol(): continue
            getattr(p, attr)[:,0] = state[self.agent.state_inds[pname, attr]]
        old_targs = self.agent.target_vecs[0]
        self.agent.target_vecs[0] = targets
        for targ, attr in self.agent.target_inds:
            if targ in prob.init_state.params:
                p = prob.init_state.params[targ]
                getattr(p, attr)[:,0] = targets[self.agent.target_inds[targ, attr]].copy()
        if plan is None:
            max_iter = 10 * self.agent.num_objs
            plan, descr = p_mod_abs(self.agent.hl_solver, self.agent, domain, prob, initial=initial, goal=goal, label=self.agent.process_id, n_resamples=5, max_iter=max_iter)
            #if self._n_plans is not None:
            #    with self._n_plans.get_lock():
            #        self._n_plans.value = self._n_plans.value + 1
        self.n_runs += 1
        self.agent.n_hl_plan += 1
        success = 0
        old_hist = self.agent.get_hist()
        if plan is not None and type(plan) is not str:
            #assert len(plan.get_failed_preds(tol=1e-3)) == 0
            path = self.agent.run_plan(plan, targets=targets, reset=reset, save=save)#, permute=self._permute>0)
            for s in path: s.source_label = 'n_plans'
            if len(path): success = path[-1].success
            self.hl_suc += 1
            self.log_path(path, 10)
            '''
            for _ in range(self._permute-1):
                self.agent.reset_to_state(state)
                self.agent.store_hist(old_hist)
                new_path = self.agent.run_plan(plan, targets=targets, reset=False, permute=True)
                self.log_path(new_path, 20)
            '''
        else:
            self.agent.n_hl_fail += 1
            self.hl_fail += 1
            print(('No plan found for', state, goal, targets, self.agent.process_id, self.agent.hl_pol))
            path = []
        self.n_success += success
        self.val_per_run.append(success)
        self.agent.target_vecs[0] = old_targs
        # self.reset()
        return success, path, plan


    def _simulate_from_unexplored(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, label=None, debug=False):
        if debug:
            print('Simulating from unexplored children.')

        if label is None:
            dummy_label = tuple(np.zeros(len(self.num_prims)+1, dtype='int32'))
            label = self.iter_labels(state, dummy_label)

            if label is None:
                return 0, None, None

            precond_cost = self.agent.cost_f(state, label, self.condition, active_ts=(0,0), debug=debug)
            if precond_cost > 0:
                return 0, None, None

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
            print(("State: ", state))

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
        for label in itertools.product(list(range(self.num_tasks)), *[list(range(n)) for n in self.num_prims]):
            label = tuple(label)
            parameterizations.append(label)
            # values.append(self.node_check_f(label, state, node))

        cost = 1.
        if self.discrete_prim and all([node.get_child(p) is None for p in parameterizations]):
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
                ind = np.random.choice(list(range(len(values))), p=exp_cost)
            else:
                inds = np.array(list(range(len(values))))[values >= np.max(values) - 1e-3]
                ind = np.random.choice(inds)

            p = parameterizations[ind]
            values[ind] = -np.inf
            cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
            while cost > 0 and np.any(values > -np.inf):
                child = node.get_child(p)
                # if child is not None:
                #     child.update_value(0) # If precondtions are violated, this is a bad path

                inds = np.array(list(range(len(values))))[values == np.max(values)]
                ind = np.random.choice(inds)
                p = parameterizations[ind]
                values[ind] = -np.inf
                cost = self.agent.cost_f(state, p, self.condition, active_ts=(0,0), debug=False)
                if cost > 0: node.failures[tuple(p)] = True

        child = node.get_child(p)
        node.update_child_explored(p)

        if debug:
            print(('Chose to explore ', p))

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


    def sample(self, task, cur_state, plan, num_samples, use_distilled=True, node=None, save=True, fixed_path=None, debug=False, hl=False, hl_check=False, skip_opt=False):
        if debug:
            print("SAMPLING")
        samples = []
        # old_traj_hist = self.agent.get_hist()
        task_name = self.tasks[task[0]]

        self.n_samples += 1
        s, success = None, False
        new_opt_strength = self.opt_strength

        use = True
        success = False
        pol = self.rollout_policy[task_name] # MixedPolicy(self.rollout_policy[task_name], self.agent.dU, self.agent.action_inds, self.agent.state_inds, None, new_opt_strength)
        if fixed_path is None and new_opt_strength > 0:
            hl = False
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
                self.bad_tree = True

        if s is None:
            if fixed_path is None:
                for n in range(num_samples):
                    task_f = None
                    if hl:
                        def task_f(s, t, curtask):
                            if self._switch_f is not None:
                                p = self._switch_f(s.get_obs(t=t))
                                if np.random.uniform() > p:
                                    return curtask
                            return self.run_hl(s, t, curtask, s.targets, check_cost=hl_check)
                        # task_f = lambda o, t, task: self.prob_func(o, self._soft, self.eta, t, task)
                    samples.append(self.agent.sample_task(pol, self.condition, cur_state, task, noisy=(n > 0), task_f=task_f, skip_opt=skip_opt))
                    # samples.append(self.agent.sample_task(pol, self.condition, cur_state, task, noisy=True))
                    if success:
                        samples[-1].set_ref_X(s.get_ref_X())
                        samples[-1].set_ref_U(s.get_ref_U())
                # if new_opt_strength < 1-1e-2:
                #     self.post_cond.append(samples[0].post_cost)
            else:
                samples.append(self.agent.sample_optimal_trajectory(cur_state, task, self.condition, fixed_path))

        lowest_cost_sample = samples[0]
        opt_fail = False
        lowest_cost_sample.opt_strength = new_opt_strength
        lowest_cost_sample.opt_suc = success

        # if np.random.uniform() > 0.99: print(lowest_cost_sample.get(STATE_ENUM), task)
        for s in samples:
            s.node = node
            if not use:
                s.use_ts[:] = 0.

        if save and fixed_path is None:
            self.agent.add_sample_batch(samples, task)
        # cur_state = lowest_cost_sample.end_state # get_X(t=lowest_cost_sample.T-1)
        cur_state = lowest_cost_sample.get_X(t=lowest_cost_sample.T-1)
        lowest_cost_sample.success = 1 - self.agent.goal_f(self.condition, cur_state)
        lowest_cost_sample.done = int(lowest_cost_sample.success)

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


    def run_ff_solve(self, state, node=None, targets=None, opt=False):
        old_opt_strength = self.opt_strength
        if targets is None:
            targets = self.agent.target_vecs[self.condition]
        if opt:
            self.opt_strength = 1.
        task_path = self.agent.task_from_ff(state, targets)
        path = []
        val = 0.
        if task_path is None: return val, path
        init_state = state
        for label in task_path:
            label = tuple(label)
            plan = self.agent.plans[label]
            if self.agent.cost_f(state, label, self.condition, active_ts=(0,0)) > 0:
                break

            next_sample, state = self.sample(label, state, plan, num_samples=1, save=True)
            T = next_sample.T - 1
            post = 1. # self.agent.cost_f(state, label, self.condition, active_ts=(T,T))

            '''
            if post > 0:
                old_opt = self.opt_strength
                self.opt_strength = 1.
                next_sample, state = self.sample(label, state, plan, num_samples=1, save=True)
                self.opt_strength = old_opt
            '''

            next_sample.node = node
            next_sample.success = 1 - self.agent.goal_f(self.condition, state)
            if node is not None:
                node = node.get_child(label)
            t = next_sample.T - 1
            next_sample.success = 1 - self.agent.goal_f(self.condition, state)
            path.append((next_sample, node))
        val = 1 - self.agent.goal_f(self.condition, state)
        for i in range(len(path)):
            if path[i][1] is not None:
                path[i][1].update_value(val)
            path[i][0].success = max(val, path[i][0].success)
            path[i][0].discount = 0.9**(len(path)-i-1)
        path = [step[0] for step in path]
        if len(path) and val > 1. - 1e-3:
            print(('\nSUCCESS! Tree {0} {1} using ff solve\n'.format(self.log_file, state)))
            self.agent.add_task_paths([path])
        else:
            print('FAILED with ff solve')
            print(('FF out:', task_path, 'Ran:', [step.task for step in path], 'End state:', state, 'Targets:', targets, 'Init state', init_state))
        self.opt_strength = old_opt_strength
        self.val_per_run.append(val)
        self.log_path(path, -1)
        return val, path


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
        path_value = 0. # None
        next_sample = None
        self.agent.reset_to_state(state)
        if np.random.uniform() < self.ff_thresh:
            val, path, _ = self.eval_pr_graph(state)
            return val, path

        while True:
            if debug:
                print("Taking simulation step")

            if self.agent.goal_f(self.condition, cur_state) == 0: # or current_node.depth >= self.max_depth:
                if not iteration:
                    print((state, self.agent.targets[self.condition]))
                    print('WARNING: Should not succeed without sampling')
                break

            if len(fixed_paths) <= iteration:
                value, next_node, next_sample = self._choose_next(cur_state, current_node, prev_sample, exclude_hl, use_distilled, debug=debug)
            else:
                path_info = fixed_paths[iteration]
                value, next_node, next_sample = self._simulate_fixed_path(*path_info)

            # if np.random.uniform() > 0.9 and current_node is self.root: print(next_sample.get(STATE_ENUM))
            path_value = np.maximum(value, path_value)
            self.node_history[tuple(cur_state)] = current_node

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

        if path_value is 0:
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
            print(('\nSUCCESS! Tree {0} {1} using fixed: {2} {3}\n'.format(self.log_file, state, len(fixed_paths) != 0, self.n_runs)))
            for s in path:
                s.prim_use_ts *= MCTS_WEIGHT
            self.agent.add_task_paths([path])
        elif len(path) and self.her:
            old_nodes = [path[i].node for i in range(len(path))]
            for s in path:
                s.node = None
                s.next_sample = None
            new_path = self.agent.relabel_path(path)
            self.agent.add_task_paths([new_path])
            for i in range(len(path)):
                path[i].node = old_nodes[i]

        self.log_path(path, len(fixed_paths))
        for n in range(len(path)):
            path[n].discount = 0.9**(len(path)-n-1)
        if self.bad_tree:
            print(('Bad tree for state', state))
            self.reset()
        if len(path): path[-1].done = 1
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


    def rollout_with_postcond(self, state, targets, max_t=10, task_ts=20, soft=False, eta=None, mode=''):
        prev_tasks = []
        cur_run = [0]
        def task_f(s, t, curtask):
            next_task = self.run_hl(s, t, curtask, targets, check_cost=False)
            if len(prev_tasks) and tuple(next_task) != tuple(prev_tasks[-1]):
                s.targets = targets
                postcost = self.agent.postcond_cost(s, prev_tasks[-1], t)
                if postcost > 0:
                    next_task = prev_tasks[-1]
            if len(prev_tasks) and tuple(next_task) == tuple(prev_tasks[-1]):
                cur_run.append(cur_run[-1]+1)
            else:
                cur_run.append(0)
            prev_tasks.append(next_task)
            return next_task

        self.agent.reset_to_state(state)
        old_opt = self.opt_strength
        path = []
        val = 0
        t = 0
        old_soft = self._soft
        self._soft = soft
        old_eta = self.eta
        if eta is not None: self.eta = eta
        l = list(self.agent.plans.keys())[0]
        l = self.iter_labels(state, l, targets=targets, debug=False, check_cost=False)
        s, t = 0, 0
        col_s, col_ts = -1, -1
        while t < max_t and val < 1-1e-2 and l is not None:
            l = self.iter_labels(state, l, targets=targets, debug=False, check_cost=False)
            if l is None: break
            task_name = self.tasks[l[0]]
            pol = self.rollout_policy[task_name]
            plan = self.agent.plans[l]
            s = self.agent.sample_task(pol, self.condition, state, l, task_f=task_f, skip_opt=True)
            val = 1 - self.agent.goal_f(0, s.get_X(s.T-1), targets)
            state = s.end_state # s.get_X(s.T-1)
            path.append(s)
            if mode == 'collision' and 1 in s.col_ts:
                col_s = t
                col_t = s.col_ts.tolist().index(1)
            t += 1
            if cur_run[-1] >= task_ts:
                break

        if col_ts >= 0:
            task = tuple(path[col_ts].get(FACTOREDTASK_ENUM, t=col_ts))
            ts = col_ts - 2
            if ts < 0:
                col_s -= 1
                if col_s < 0:
                    col_s, col_ts = 0, 0
                else:
                    ts = path[col_s].T + ts - 1
            x = path[col_s].get_X(t=ts)
            plan = self.agent.plans[task]
            success = self.agent.backtrack_solve(plan, x0=x)
            if success:
                new_samples = self.agent.run_plan(plan, targets, record=False, save=False)
                for s in new_samples:
                    self.optimal_samples[self.agent.task_list[task[0]]].append(s)
            print('OPT on collision in rollout:', success, task, x)

        if val < 1-1e-3:
            last_task = tuple(path[-1].get(FACTOREDTASK_ENUM, t=path[-1].T-1))
            t = len(prev_tasks)-1
            while t >= 0 and tuple(last_task) == tuple(prev_tasks[t]):
                t -= 1
            ind = 0
            while t >= path[ind].T:
                ind += 1
                t -= path[ind].T
            s, t = ind, t

        self.opt_strength = old_opt
        self.eta = old_eta
        self.log_path(path, -50)
        self._soft = old_soft
        # (s, t) indexes the switch where it failed postconditions
        return val, path, s, t


    def test_run(self, state, targets, max_t=20, hl=False, soft=False, check_cost=True, eta=None):
        self.agent.reset_to_state(state)
        old_opt = self.opt_strength
        # self.opt_strength = 1.
        path = []
        val = 0
        l = (0,0,0,0)
        t = 0
        old_soft = self._soft
        self._soft = soft
        old_eta = self.eta
        if eta is not None: self.eta = eta 
        debug = np.random.uniform() < 0.1
        while t < max_t and val < 1-1e-2 and l is not None:
            l = self.iter_labels(state, l, targets=targets, debug=debug, check_cost=check_cost)
            if l is None: break
            plan = self.agent.plans[l]
            s, _ = self.sample(l, state, plan, 1, hl=hl, hl_check=check_cost, save=False, skip_opt=True)
            val = 1 - self.agent.goal_f(0, s.get_X(s.T-1), targets)
            t += 1
            state = s.end_state # s.get_X(s.T-1)
            path.append(s)
        self.opt_strength = old_opt
        self.eta = old_eta
        self.log_path(path, -5)
        self._soft = old_soft
        return val, path


    def get_bad_labels(self, state):
        labels = [l for l in self.label_options]
        bad = []
        for l in labels:
            cost = self.agent.cost_f(state, l, self.condition, active_ts=(0,0), debug=debug)
            if cost > 1e-3:
                bad.append(l)
        return bad


    def run_hl(self, sample, t=0, task=None, targets=None, check_cost=False, debug=False):
        next_label, distr = self.eval_hl(sample, t, targets, debug, True)
        # if t== 0:
            # distrs = self.prob_func(sample.get_prim_obs(t=t), False, eta=1.)
            # print(distrs, sample.get(STATE_ENUM, t=t), sample.get(ONEHOT_GOAL_ENUM, t=t))
        if not check_cost: return next_label
        return self.iter_distr(next_label, distr, self.label_options, sample.get_X(t), sample)


    def eval_hl(self, sample, t=0, targets=None, debug=False, find_distr=False):
        labels = [l for l in self.label_options]
        if self.use_q:
            obs = sample.get_val_obs(t=t)
            opts = self.agent.prob.get_prim_choices(self.agent.task_list)
            distr = np.zeros(len(labels))
            dact = np.sum([len(opts[e]) for e in opts])
            for i in range(len(labels)):
                l = labels[i]
                act = np.zeros(dact)
                cur_ind = 0
                for j, e in enumerate(opts):
                    act[cur_ind + l[j]] = 1.
                    cur_ind += len(opts[e])
                distr[i:i+1] = self.value_func(obs, act)

            if self._soft:
                exp_wt = np.exp(self.eta*(distr - np.max(distr)))
                wt = exp_wt / np.sum(exp_wt)
                ind = np.random.choice(list(range(len(labels))), p=wt)
            else:
                ind = np.argmax(distr)
            next_label = tuple(labels[ind])

        elif self.discrete_prim:
            task = sample.task if hasattr(sample, 'task') else None
            distrs = self.prob_func(sample.get_prim_obs(t=t), self._soft, eta=self.eta, t=t, task=task)
            for d in distrs:
                for i in range(len(d)):
                    d[i] = round(d[i], 5)

            if self.onehot_task:
                distr = distrs[0]
                val = np.max(distr)
                ind = np.random.choice([i for i in range(len(distr)) if distr[i] >= val])
                next_label = self.agent.task_to_onehot[ind]
            else:
                distr = [np.prod([distrs[i][l[i]] for i in range(len(l))]) for l in labels]
                distr = np.array(distr)
                ind = []
                for d in distrs:
                    val = np.max(d)
                    ind.append(np.random.choice([i for i in range(len(d)) if d[i] >= val]))
                next_label = tuple(ind) # tuple([ind[d] for d in range(len(distrs))])
        if find_distr: return tuple(next_label), distr
        return tuple(next_label)


    def iter_distr(self, next_label, distr, labels, end_state, sample, debug=False):
        cost = self.agent.cost_f(end_state, next_label, self.condition, active_ts=(0,0), debug=debug)
        post = 1. # self.agent.cost_f(end_state, next_label, self.condition, active_ts=(sample.T-1,sample.T-1), debug=debug)
        init_label = next_label
        T = self.agent.plans[next_label].horizon - 1

        while (cost > 0 or post < 1e-3) and np.any(distr > -np.inf):
            next_label = []
            if self.soft_decision:
                expcost = self.n_runs * distr
                expcost = expcost - np.max(expcost)
                expcost = np.exp(expcost)
                expcost = expcost / np.sum(expcost)
                ind = np.random.choice(list(range(len(distr))), p=expcost)
                next_label = tuple(labels[ind])
                distr[ind] = -np.inf
            else:
                val = np.max(distr)
                inds = [i for i in range(len(distr)) if distr[i] >= val]
                ind = np.random.choice(inds)
                # ind = np.argmax(distr)
                if self.onehot_task:
                    print('ONEHOT?')
                    next_label = self.agent.task_to_onehot[ind]
                else:
                    next_label = tuple(labels[ind])
                distr[ind] = -np.inf
            cost = self.agent.cost_f(end_state, next_label, self.condition, active_ts=(0,0), debug=debug)
            post = 1. # self.agent.cost_f(end_state, next_label, self.condition, active_ts=(T,T), debug=debug)
        if cost > 0:
            return init_label
        return next_label


    def iter_labels(self, end_state, label, exclude=[], targets=None, debug=False, find_bad=False, check_cost=True):
        sample = Sample(self.agent)
        sample.set_X(end_state.copy(), t=0)
        self.agent.fill_sample(self.condition, sample, sample.get(STATE_ENUM, 0), 0, label, fill_obs=True, targets=targets)
        labels = [l for l in self.label_options]
        cost = 1.
        bad = []
        next_label, distr = self.eval_hl(sample, 0, targets, debug, find_distr=True)
        if not check_cost: return tuple(next_label)
        T = self.agent.plans[next_label].horizon - 1
        cost = self.agent.cost_f(end_state, next_label, self.condition, active_ts=(0,0), debug=debug, targets=targets)
        post = 1. # self.agent.cost_f(end_state, next_label, self.condition, active_ts=(T,T), debug=debug, targets=targets)

        for l in exclude:
            if self.onehot_task:
                ind = self.agent.task_to_onehot[tuple(l)]
            else:
                ind = labels.index(tuple(l))
            distr[ind] = 0.
        self.prim_pre_cond.append(cost)
        costs = {}
        while (cost > 0 or post < 1e-5) and np.any(distr > -np.inf):
            next_label = []
            if self.soft_decision:
                expcost = self.n_runs * distr
                expcost = expcost - np.max(expcost)
                expcost = np.exp(expcost)
                expcost = expcost / np.sum(expcost)
                ind = np.random.choice(list(range(len(distr))), p=expcost)
                next_label = tuple(labels[ind])
                distr[ind] = -np.inf
            else:
                val = np.max(distr)
                inds = [i for i in range(len(distr)) if distr[i] >= val]
                ind = np.random.choice(inds)
                # ind = np.argmax(distr)
                if self.onehot_task:
                    print('ONEHOT?')
                    next_label = self.agent.task_to_onehot[ind]
                else:
                    next_label = tuple(labels[ind])
                distr[ind] = -np.inf
            T = self.agent.plans[next_label].horizon - 1
            cost = self.agent.cost_f(end_state, next_label, self.condition, active_ts=(0,0), debug=debug, targets=targets)
            post = 1. # self.agent.cost_f(end_state, next_label, self.condition, active_ts=(T,T), debug=debug, targets=targets)
            if cost > 0:
                bad.append(next_label)
                next_label = None
            costs[next_label] = (cost, post, end_state)
        if cost > 0 or post < 1e-5:
            print(('NO PATH FOR:', end_state, 'excluding:', exclude))
        if find_bad:
            return next_label, bad
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
        if path_value >= 1. - 1e-3 or depth >= self.max_depth: # or hl_encoding in exclude_hl:
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
            X = [{(pname, attr): sample.get_X(t=t)[self.agent.state_inds[pname, attr]].round(3) for pname, attr in self.agent.state_inds if self.agent.state_inds[pname, attr][-1] < self.agent.symbolic_bound} for t in range(sample.T)]
            if hasattr(sample, 'col_ts'):
                U = [{(pname, attr): (sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4), sample.col_ts[t]) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            else:
                U = [{(pname, attr): sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            info = {'X': X, 'task': sample.task, 'time_from_start': time.time() - self.start_t, 'n_runs': self.n_runs, 'n_resets': self.n_resets, 'value': 1.-sample.task_cost, 'fixed_samples': n_fixed, 'root_state': self.agent.x0[self.condition], 'opt_strength': sample.opt_strength if hasattr(sample, 'opt_strength') else 'N/A'}
            if verbose:
                info['obs'] = sample.get_obs().round(3)
                # info['prim_obs'] = sample.get_prim_obs().round(3)
                info['targets'] = {tname: sample.targets[self.agent.target_inds[tname, attr]] for tname, attr in self.agent.target_inds}
                info['cur_curric'] = self.cur_curric
                info['opt_success'] = sample.opt_suc
                info['tasks'] = sample.get(FACTOREDTASK_ENUM)
                info['hl_suc'] = self.hl_suc
                info['hl_fail'] = self.hl_fail
                info['end_state'] = sample.end_state
                # info['prim_obs'] = sample.get_prim_obs().round(4)
            data.append(info)
        return data


    def log_path(self, path, n_fixed=0):
        if self.log_file is None: return
        with open(self.log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed)
            pp_info = pprint.pformat(info, depth=120, width=120)
            f.write(pp_info)
            f.write('\n')

        with open(self.verbose_log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed, True)
            pp_info = pprint.pformat(info, depth=120, width=120)
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
