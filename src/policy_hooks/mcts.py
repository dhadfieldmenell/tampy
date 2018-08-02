from copy import copy
import numpy as np

class MCTSNode():
    def __init__(self, label, value, parent):
        self.label = label
        self.value = value
        self.children = []
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
    def __init__(self, tasks, state_func, prob_func, goal_f, target_f, rollout_policy, condition, agent, num_samples, choose_next=None, cost_f=None, soft_decision=1.0, C=2, max_depth=20):
        self.tasks = tasks
        self.state_func = state_func
        self.prob_func = prob_func
        self.root = MCTSNode('root', -1, None)
        self.max_depth = max_depth
        self.rollout_policy = rollout_policy
        self.condition = condition
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C
        self.num_samples = num_samples
        self._choose_next = choose_next if choose_next != None else self._default_choose_next
        self._cost_f = cost_f if cost_f != None else self._default_cost_f
        self._goal_f = goal_f
        self._target_f = target_f
        self.node_check_f = lambda n: n.value+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf

    def run(self, state, num_rollouts=20):
        opt_val = np.inf
        paths = []
        for _ in range(num_rollouts):
            paths.append(self.simulate(state))
            end = paths[-1][-1]
            opt_val = np.minimum(self._goal_f(end.get_ref_X(t=end.T-1)), opt_val)

        return paths, opt_val

    def _default_cost_f(self, X, task, target, t_range=None):
        plan = self.agent.plans[task]
        self.agent.set_target(plan, target)
        if t_range == None: t_range = (0, plan.horizon)
        for t in range(t_range[0], t_range[1]):
            set_params_attrs(plan.params.values(), plan.params.state_inds, X[t], t)

        failed = [(pred, negated) for negated, pred, t in plan.get_failed_preds(active_ts=t_range, priority=3, tol=1e-3)]
        if len(failed):
            return sum([np.max(pred.check_pred_violation(tol=1e-3, negated=negated)) for pred, negated in failed])

        return 0

    def _simulate_from_unexplored(self, state, node, prev_sample):
        prob_distr = self.prob_func(state)
        for i in range(len(prob_distr)):
            if node.children[i] != None: prob_distr[i] = 0

        next_node = None
        while np.any(prob_distr > 0):
            next_ind = np.argmax(prob_distr)
            prob_distr[next_ind] = 0
            target = self._target_f(self.agent.plans[self.tasks[next_ind]], self.tasks[next_ind], state)
            if target[0] != None:
                next_node = Node(next_ind, 0, node)
                cost = self.simulate_from_next(node, state, prev_sample)
                next_node.value -= cost
                node.children[next_ind] = next_node
                while node != self.root:
                    node.n_explored += 1
                    node.value -= cost
                    node = node.parent
                return next_node

        return None

    def _select_from_explored(self, state, node):
        children = copy(node.children)
        children = filter(lambda n: n != None, children)
        prob_distr = self.prob_func(state)
        for i in range*ken(prob_distr)):
            if node.children[i] == None: prob_distr[i] = 0

        while np.any(prob_distr > 0):
            next_ind = np.argmax(prob_distr)
            prob_distr[next_ind] = 0
            target = self._target_f(self.agent.plans[self.tasks[next_ind]], self.tasks[next_ind], state)
            if target != None:
                return node.children[next_ind]

        return None

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
                return self._select_from_unexplored(state, node)
            else:
                return self._select_from_explored(state, node)

            choose_unepxplored = not choose_unexplored
            loops += 1

        return None

    def _default_choose_next(self, state, node, prev_sample):
        for _ in range(len(node.children)):
            self._simulate_from_unexplored(state, node, prev_sample)

        next_node = self._simulate_from_explored(state, node)
        return next_node

    def simulate(self, state):
        current_node = self.root
        path = []
        samples = []

        success = True
        cur_state = state
        prev_sample = None
        terminated = False
        opt_state = np.zeros((len(state),))
        while not self.is_terminal(cur_state) and not current_node.depth >= self.max_depth:
            next_node = self.choose_next(cur_state, current_node, prev_sample)

            samples = []
            if next_node == None:
                break

            task = self.tasks[next_node.label]
            target = self._target_f(self.agent.plans[task], task, cur_state)
            for n in range(self.num_samples):
                samples.append(self.agent.sample_task(self.rollout_policy[next_node.label], self.condition, cur_state, task, save=True, noisy=True))

            sample_costs = {}
            for sample in samples:
                # Note: the cost function needs to describe any dynamics not enforced by the sim (such as collisions in openrave)
                sample_costs[sample] = self._cost_f(sample.get_X(), task, target)
            
            lowest_cost_ind = np.argmin(samples_costs.values())
            lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]

            opt_fail = False
            
            # If the motion policy is not optimal yet, want to collect optimal trajectories from the motion planner
            if sample_costs[lowest_cost_sample] > 0:
                opt_sample = self.agent.sample_optimal_trajectory(cur_state, self.tasks[next_node.label], self.condition, target)
                if opt_sample != None:
                    opt_X, opt_U = opt_sample.get_X(), opt_sample.get_U()
                    samples.append(opt_sample)
                    for sample in samples:
                        sample.set_ref_X(opt_X)
                        sample.set_ref_U(opt_U)
                else:
                    opt_fail = True

            if sample_costs[lowest_cost_sample] == 0 or opt_fail:
                for sample in samples:
                    # When setting the reference trajectory, only refer for those samples that are not optimal already
                    if sample_costs[sample] > 0:
                        sample.set_ref_X(lowest_cost_sample.get_X().copy())
                        sample.set_ref_U(lowest_cost_sample.get_U().copy())
                    else:
                        sample.set_ref_X(sample.get_X().copy())
                        sample.set_ref_U(sample.get_U().copy())
            
            cur_state = lowest_cost_sample.get_X(t=lowest_cost_sample.T-1)
            current_node.sample_links[lowest_cost_sample] = prev_sample # Used to retrace paths
            prev_sample = lowest_cost_sample
 
            current_node = next_node
            path.append(current_node)

            # For task planning, asks "would this have been the best course of action if I'd moved optimally?"
            opt_state = lowest_cost_sample.get_ref_X(t=lowest_cost_sample.T-1)

            if self._goal_f(opt_state) == 0 or current_node.depth >= self.max_depth:
                break

        end_ts = self.agent.plans[task].horizon - 1
        path_value = self._goal_f(cur_state)
        path = []
        while current_node is not self.root:
            path.append(prev_sample)
            prev_sample = current_node.sample_links[prev_sample]
            current_node.n_explored += 1
            current_node.value -= path_value
            current_node = current_node.parent

        path.reverse()
        return path

    def simulate_from_next(self, node, state, prev_sample):
        cost, samples = self._simulate_from_next(npde.label, node.depth, state, self.prob_func, [])
        node.sample_links[samples[0]] = prev_sample
        return cost

    def _simulate_from_next(self, task_ind, depth, state, prob_func, samples):
        task = self.tasks[task_ind]
        new_samples = []
        target = self._target_f(self.agent.plans[task], task, state)

        for n in range(self.num_sample):
            sample = self.agent.sample_task(self.rollout_policy[task], self.condition, task, state, save=True, noisy=True)
            new_samples.append(sample)

        sample_costs = {}
        for sample in new_samples:
            sample_costs[sample] = self._cost_f(sample.get_X(), task, target)
        
        lowest_cost_ind = np.argmin(samples_costs.values())
        lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]

        if sample_costs[lowest_cost_sample] > 0:
            opt_X, opt_U = self.agent.get_opt_traj(cur_state, self.tasks[taks_ind], self.condition, target)
            for sample in new_samples:
                sample.set_ref_X(opt_X)
                sample.set_ref_U(opt_U)
        else:
            for sample in new_samples:
                if sample_costs[sample] > 0:
                    sample.set_ref_X(lowest_cost_sample.get_X().copy())
                    sample.set_ref_U(lowest_cost_sample.get_U().copy())
       
        self.agent.add_sample_batch(new_samples)
 
        samples.append(lowest_cost_sample)

        # Refer to what the end state would have been moving optimally
        end_state = lowest_cost_sample.get_ref_X(t=lowest_cost_sample.T-1)
        if self.is_terminal(end_state) or depth >= self.max_depth * 2:
            return self._goal_f(end_state), samples

        prob_distr = prob_func(end_state)
        if np.random.sample() > self.soft_decision:
            next_task = np.random.choice(range(len(self.tasks)), prob_distr)
        else:
            next_task = np.argmax(prob_distr)

        return self._simulate_from_next(next_task, depth+1, end_state, prob_func, samples)

