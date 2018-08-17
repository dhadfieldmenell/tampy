from copy import copy, deepcopy
import numpy as np

from gps.sample.sample import Sample

from policy_hooks.utils.policy_solver_utils import *

class MCTSNode():
    def __init__(self, label, value, parent, num_tasks):
        self.label = label
        self.value = value
        self.children = [None for _ in range(num_tasks)]
        self.parent = parent
        self.n_explored = 1.0
        self.sample_links = {}
        self.sample_to_traj = {}
        self.depth = parent.depth + 1 if parent != None else 0

    def is_leaf(self):
        return len(self.children) == 0

    def update_value(self, new_value):
        self.value = (self.value*(self.n_explored-1) + new_value) / self.n_explored

class MCTS:
    def __init__(self, tasks, prob_func, plan_f, cost_f, goal_f, target_f, rollout_policy, distilled_policy, condition, agent, num_samples, num_distilled_samples, choose_next=None, soft_decision=1.0, C=2, max_depth=20):
        self.tasks = tasks
        self._prob_func = prob_func
        self.root = MCTSNode(-1, -1, None, len(tasks))
        self.max_depth = max_depth
        self.rollout_policy = rollout_policy
        self.distilled_policy = distilled_policy
        self.condition = condition
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C
        self.num_samples = num_samples
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = choose_next if choose_next != None else self._default_choose_next
        self._plan_f = plan_f
        self._cost_f = cost_f 
        self._goal_f = goal_f
        self._target_f = target_f
        self.node_check_f = lambda n: n.value+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self.targets = np.zeros((self.agent.target_dim))
        for target_name in self.agent.targets[self.condition]:
            target = self.agent.plans.values()[0].params[target_name]
            self.targets[self.agent.target_inds[target.name, 'value']] = self.agent.targets[condition][target.name]

        self._opt_cache = {}

    def prob_func(self, state):
        state = state.reshape((1, -1))
        return self._prob_func(state).flatten()

    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = np.inf
        paths = []
        if hl_plan == None:
            for n in range(num_rollouts):
                self.agent.reset_hist()
                print "MCTS Rollout {0} for condition {1}.\n".format(n, self.condition)
                next_path = self.simulate(state.copy(), use_distilled)
                if not len(next_path):
                    continue
                paths.append(next_path)
                end = next_path[-1]
                opt_val = np.minimum(self._goal_f(end.get_X(t=end.T-1), self.agent.targets[self.condition], self.agent.plans.values()[0]), opt_val)
        else:
            cur_state = state
            paths = [[]]
            cur_sample = None
            opt_val = np.inf
            for step in hl_plan:
                targets = [self.agent.plans.values()[0].params[p_name] for p_name in step[1]]
                if len(targets) < 2:
                    targets.append(self.agent.plans.values()[0].params['{0}_init_target'.format(p_name)])
                plan = self._plan_f(step[0], targets)
                # next_sample, cur_state = self.sample(step[0], cur_state, targets, plan)
                next_sample = self.agent.sample_optimal_trajectory(cur_state, step[0], self.condition, targets)
                if next_sample == None:
                    break
                cur_sample = next_sample
                cur_state = cur_sample.get_X(t=cur_sample.T-1)
                paths[0].append(cur_sample)

            if cur_sample != None: 
                opt_val = self._goal_f(cur_sample.get_X(t=cur_sample.T-1), self.agent.targets[self.condition], self.agent.plans.values()[0])
                for path in paths:
                    for sample in path:
                        sample.task_cost = opt_val

        self.agent.add_task_paths(paths)
        return opt_val

    '''
    def _default_cost_f(self, X, task, target, t_range=None):
        plan = self.agent.plans[task]
        self.agent.set_target(plan, target, self.condition)
        if t_range == None: t_range = (0, plan.horizon)
        for t in range(t_range[0], t_range[1]):
            set_params_attrs(plan.params, plan.params.state_inds, X[t], t)

        failed = [(pred, negated) for negated, pred, t in plan.get_failed_preds(active_ts=t_range, priority=3, tol=1e-3)]
        if len(failed):
            return sum([np.max(pred.check_pred_violation(tol=1e-3, negated=negated)) for pred, negated in failed])

        return 0
    '''

    def _simulate_from_unexplored(self, state, node, prev_sample, use_distilled=True):
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        prob_distr = self.prob_func(sample.get_obs(t=0))

        for i in range(len(prob_distr)):
            if node.children[i] != None: prob_distr[i] = 0

        next_node = None
        old_traj_hist = self.agent.get_hist()
        while np.any(prob_distr > 0):
            next_ind = np.argmax(prob_distr)
            prob_distr[next_ind] = 0
            target = self._target_f(self.agent.plans.values()[0], state, self.tasks[next_ind], self.agent.targets[self.condition])
            if target[0] != None:
                self.agent.reset_hist(deepcopy(old_traj_hist))
                next_node = MCTSNode(next_ind, 0, node, len(self.tasks))
                cost = self.simulate_from_next(next_node, state, prev_sample, use_distilled)
                next_node.value -= cost
                node.children[next_ind] = next_node
                while node != self.root:
                    node.n_explored += 1
                    node.value -= cost
                    node = node.parent
                return next_node

        self.agent.reset_hist(old_traj_hist)

        return None

    def _select_from_explored(self, state, node):
        children = copy(node.children)
        children = filter(lambda n: n != None, children)
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        prob_distr = self.prob_func(sample.get_obs(t=0))
        
        for i in range(len(prob_distr)):
            if node.children[i] == None: prob_distr[i] = 0

        while np.any(prob_distr > 0):
            next_ind = np.argmax(prob_distr)
            prob_distr[next_ind] = 0
            target = self._target_f(self.agent.plans.values()[0], state, self.tasks[next_ind], self.agent.targets[self.condition])
            if target[0] != None:
                return node.children[next_ind]

        return None

    '''
    def _default_choose_next(self, state, node):
        unexplored_inds = [i for i in range(len(node.children)) if node.children[i] == None]
        explored_inds = [i for i in range(len(node.children)) if node.children[i] != None]

        prob_distr = self.prob_func(state)
        explored_prob = np.sum(prob_distr[explored_inds]) if len(explored_inds) else 0
        unexplored_prob = np.sum(prob_distr[unexplored_inds]) if len(unexplored_inds) else 0

        choose_unexplored = np.random.choice([0, 1], p=[explored_prob, unexplored_prob])

        loops = 0
        while loops < 2:
            if choose_unexplored:
                node = self._select_from_unexplored(state, node)
            else:
                node = self._select_from_explored(state, node)

            if node != None:
                return node

            choose_unepxplored = not choose_unexplored
            loops += 1

        return None
    '''

    def _default_choose_next(self, state, node, prev_sample, use_distilled=True):
        for _ in range(len(node.children)):
            self._simulate_from_unexplored(state, node, prev_sample, use_distilled)

        next_node = self._select_from_explored(state, node)
        return next_node

    def sample(self, task, cur_state, target, plan, use_distilled=True, node=None):
        samples = []

        old_traj_hist = self.agent.get_hist()

        for n in range(self.num_samples):
            self.agent.reset_hist(deepcopy(old_traj_hist))
            samples.append(self.agent.sample_task(self.rollout_policy[task], self.condition, cur_state, (task, target[0].name), noisy=True))

        if use_distilled:
            for n in range(self.num_distilled_samples):
                self.agent.reset_hist(deepcopy(old_traj_hist))
                samples.append(self.agent.sample_task(self.distilled_policy, self.condition, cur_state, (task, target[0].name), noisy=True))

        sample_costs = {}
        for sample in samples:
            # Note: the cost function needs to describe any dynamics not enforced by the sim (such as collisions in openrave)
            sample_costs[sample] = self._cost_f(sample.get_X(), task, target, self.agent.targets[self.condition], plan)
            sample.plan = plan

        lowest_cost_ind = np.argmin(sample_costs.values())
        lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]

        opt_fail = False

        '''
        # If the motion policy is not optimal yet, want to collect optimal trajectories from the motion planner
        if sample_costs[lowest_cost_sample] > 0:
            if node not in self._opt_cache:
                opt_sample = self.agent.sample_optimal_trajectory(cur_state, task, self.condition, target)
                if opt_sample != None:
                    opt_X, opt_U = opt_sample.get_X(), opt_sample.get_U()
                    samples.append(opt_sample)
                    for sample in samples:
                        sample.set_ref_X(opt_X)
                        sample.set_ref_U(opt_U)
                    if node != None:
                        self._opt_cache[node] = (opt_X, opt_U)
                else:
                    opt_fail = True
            else:
                for sample in samples:
                    opt_X, opt_U = self._opt_cache[node]
                    sample.set_ref_X(opt_X)
                    sample.set_ref_U(opt_U)

        if sample_costs[lowest_cost_sample] == 0 or opt_fail:
            for sample in samples:
                # When setting the reference trajectory, only refer for those samples that are not optimal already
                if sample_costs[sample] > 0:
                    sample.set_ref_X(lowest_cost_sample.get_X().copy())
                    sample.set_ref_U(lowest_cost_sample.get_U().copy())
                else:
                    sample.set_ref_X(sample.get_X().copy())
                    sample.set_ref_U(sample.get_U().copy())
        '''

        self.agent.add_sample_batch(samples, task)
        cur_state = lowest_cost_sample.get_X(t=lowest_cost_sample.T-1)
        self.agent.reset_hist(lowest_cost_sample.get_U()[-self.agent.hist_len:].tolist())

        return lowest_cost_sample, cur_state

    def simulate(self, state, use_distilled=True):
        current_node = self.root
        path = []
        samples = []

        success = True
        cur_state = state.copy()
        prev_sample = None
        terminated = False
        while True:
            if self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0]) == 0 or current_node.depth >= self.max_depth:
                break

            next_node = self._choose_next(cur_state, current_node, prev_sample, use_distilled)

            if next_node == None:
                break

            task = self.tasks[next_node.label]
            target = self._target_f(self.agent.plans.values()[0], cur_state, task, self.agent.targets[self.condition])
            if target[0] is None:
                break

            plan = self._plan_f(task, target)
            if self._cost_f(cur_state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
                break
            # if target[0] == None:
            #     break

            next_sample, cur_state = self.sample(task, cur_state, target, plan, use_distilled)

            current_node.sample_links[next_sample] = prev_sample # Used to retrace paths
            prev_sample = next_sample
 
            current_node = next_node
            path.append(current_node)


        path_value = self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0])
        path = []
        while current_node is not self.root:
            path.append(prev_sample)
            prev_sample.task_cost = path_value
            prev_sample = current_node.parent.sample_links[prev_sample]
            current_node.n_explored += 1
            current_node.value -= path_value
            current_node = current_node.parent

        path.reverse()
        return path

    def simulate_from_next(self, node, state, prev_sample, use_distilled=True):
        cost, samples = self._simulate_from_next(node.label, node.depth, state, self.prob_func, [], use_distilled)
        if len(samples):
            node.sample_links[samples[0]] = prev_sample
        return cost

    def _simulate_from_next(self, task_ind, depth, state, prob_func, samples, use_distilled=True):
        task = self.tasks[task_ind]
        new_samples = []
        target = self._target_f(self.agent.plans.values()[0], state, task, self.agent.targets[self.condition])
        if target[0] is None:
            return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans.values()[0]), samples

        plan = self._plan_f(task, target)
        if self._cost_f(state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
            return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans.values()[0]), samples
        # if target[0] == None:
        #     return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task]), samples 

        next_sample, end_state = self.sample(task, state, target, plan, use_distilled)

        samples.append(next_sample)

        # Refer to what the end state would have been moving optimally
        path_value = self._goal_f(end_state, self.agent.targets[self.condition], self.agent.plans.values()[0])
        if path_value == 0 or depth >= self.max_depth:
            for sample in samples:
                sample.task_cost = path_value
            return self._goal_f(end_state, self.agent.targets[self.condition], self.agent.plans.values()[0]), samples

        
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, end_state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        prob_distr = self.prob_func(sample.get_obs(t=0))
        
        if np.random.sample() > self.soft_decision:
            next_task = np.random.choice(range(len(self.tasks)), prob_distr)
        else:
            next_task = np.argmax(prob_distr)

        return self._simulate_from_next(next_task, depth+1, end_state, prob_func, samples, use_distilled)

