from copy import copy, deepcopy
import numpy as np

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *

class MCTSNode():
    def __init__(self, label, value, parent, num_tasks, num_objs, num_targs):
        self.label = label
        self.value = value
        self.num_tasks = num_tasks
        self.num_objs = num_objs
        self.num_targs = num_targs
        self.is_leaf = True
        self.children = [None for _ in range(num_tasks)]
        self.parent = parent
        self.n_explored = 1.0
        self.sample_links = {}
        self.sample_to_traj = {}
        self.depth = parent.depth + 1 if parent != None else 0

    def is_leaf(self):
        return self.is_leaf()

    def get_task(self):
        return self.label[0]

    def get_obj(self):
        return self.label[1]

    def get_targ(self):
        return self.label[2]

    def update_value(self, new_value):
        self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)
        self.n_explored += 1

    def get_child(self, task_ind, obj_ind, targ_ind):
        if self.children[task_ind] is None or \
           self.children[task_ind][obj_ind] is None or \
           self.children[task_ind][obj_ind][targ_ind] is None:
            return None
        return self.children[task_ind][obj_ind][targ_ind]

    def add_child(self, child):
        task_ind, obj_ind, targ_ind = child.label
        if self.children[task_ind] is None:
            self.children[task_ind] = [None for _ in range(self.num_objs)]
        if self.children[task_ind][obj_ind] is None:
            self.children[task_ind][obj_ind] = [None for _ in range(self.num_targs)]
        self.children[task_ind][obj_ind][targ_ind] = child
        self.is_leaf = False
        child.parent = self

    def get_explored_children(self):
        children = []
        for task in self.children:
            if task is None: continue
            for obj in task: 
                if obj is None: continue
                children.extend(filter(lambda n: n is not None, obj))
        return children

    def has_unexplored(self):
        for task in self.children:
            if task is None: return True
            for obj in task:
                if obj is None: return True
                for targ in obj:
                    if targ is None: return True
        return False


class MCTS:
    def __init__(self, tasks, prob_func, plan_f, cost_f, goal_f, target_f, encode_f, rollout_policy, distilled_policy, condition, agent, num_samples, num_distilled_samples, choose_next=None, soft_decision=1.0, C=2, max_depth=20):
        self.tasks = tasks
        self._prob_func = prob_func
        self.root = MCTSNode((-1, -1, -1), 0, None, len(tasks), len(agent.obj_list), len(agent.targ_list))
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
        self._encode_f = encode_f
        self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self.targets = np.zeros((self.agent.target_dim))
        for target_name in self.agent.targets[self.condition]:
            target = self.agent.plans.values()[0].params[target_name]
            self.targets[self.agent.target_inds[target.name, 'value']] = self.agent.targets[condition][target.name]

        self.Q = {}
        self.N = {}
        self.P = {}

        self._opt_cache = {}

    def prob_func(self, state):
        state = state.reshape((1, -1))
        return self._prob_func(state)

    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = np.inf
        paths = []
        if hl_plan == None:
            for n in range(num_rollouts):
                self.agent.reset_hist()
                print "MCTS Rollout {0} for condition {1}.\n".format(n, self.condition)
                next_path = self.simulate(state.copy(), use_distilled, debug=debug)
                if len(next_path):
                    end = next_path[-1]
                    new_opt_value = self._goal_f(end.get_X(t=end.T-1), self.agent.targets[self.condition], self.agent.plans.values()[0])
                    if new_opt_value == 0: paths.append(next_path)
                    opt_val = np.minimum(new_opt_value, opt_val)
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
                next_sample, _ = self.agent.sample_optimal_trajectory(cur_state, step[0], self.condition, targets)
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

    def _simulate_from_unexplored(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print 'Simulating from unexplored children.'
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))

        obj = self.agent.plans.values()[0].params[self.agent.obj_list[np.argmax(obj_distr)]]
        targ = self.agent.plans.values()[0].params[self.agent.targ_list[np.argmax(targ_distr)]]

        next_node = None
        old_traj_hist = self.agent.get_hist()
        while np.any(task_distr > 0):
            next_task_ind = np.argmax(task_distr)
            task_distr[next_task_ind] = 0
            new_obj_distr = obj_distr.copy()
            while np.any(new_obj_distr > 0):
                next_obj_ind = np.argmax(new_obj_distr)
                new_obj_distr[next_obj_ind] = 0
                new_targ_distr = targ_distr.copy()
                while np.any(new_targ_distr > 0):
                    next_targ_ind = np.argmax(new_targ_distr)
                    new_targ_distr[next_targ_ind] = 0

                    obj = self.agent.plans.values()[0].params[self.agent.obj_list[next_obj_ind]]
                    targ = self.agent.plans.values()[0].params[self.agent.targ_list[next_targ_ind]]

                    if node.get_child(next_task_ind, next_obj_ind, next_targ_ind) != None:
                        continue

                    self.agent.reset_hist(deepcopy(old_traj_hist))
                    next_node = MCTSNode((next_task_ind, next_obj_ind, next_targ_ind), 
                                         0, 
                                         node, 
                                         len(self.tasks), 
                                         len(self.agent.obj_list), 
                                         len(self.agent.targ_list))
                    cost, _ = self.simulate_from_next(next_node, state, prev_sample, num_samples=5, use_distilled=use_distilled, debug=debug)
                    next_node.update_value(-cost)
                    node.add_child(next_node)
                    while node != self.root:
                        node.update_value(-cost)
                        node = node.parent
                    return next_node


        self.agent.reset_hist(old_traj_hist)

        return None

    def _select_from_explored(self, state, node, exclude_hl=[], debug=False):
        if debug:
            print "Selecting from explored children."
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        # task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))

        # obj = self.agent.plans.values()[0].params[self.agent.obj_list[np.argmax(obj_distr)]]
        # targ = self.agent.plans.values()[0].params[self.agent.targ_list[np.argmax(targ_distr)]]
        # target = [obj, targ]
        # for i in range(len(task_distr)):
        #     if node.children[i] == None: task_distr[i] = 0
        # while np.any(task_distr > 0):
        #     next_ind = np.argmax(task_distr)
        #     task_distr[next_ind] = 0
        #     target = self._target_f(self.agent.plans.values()[0], state, self.tasks[next_ind], self.agent.targets[self.condition])
        #     if target[0] != None:
        #         return node.children[next_ind]
        #     plan = self.agent.plans[self.tasks[next_ind], obj.name]
        #     if self._cost_f(state, self.tasks[next_ind], target, self.agent.targets[self.condition], plan, active_ts=(0,0)) == 0:
        #         return node.children[next_ind]

        children = node.get_explored_children() 
        children_distr = map(self.node_check_f, children)
        # children_distr = np.array(children_distr) / len(children_distr)

        while len(children_distr) and np.any(children_distr != -np.inf):
            next_ind = np.argmax(children_distr)
            children_distr[next_ind] = -np.inf
            # target = self._target_f(self.agent.plans.values()[0], state, self.tasks[next_ind], self.agent.targets[self.condition])
            next_node = children[next_ind]
            task_ind = next_node.get_task()
            obj_ind = next_node.get_obj()
            targ_ind = next_node.get_targ()

            obj = self.agent.plans.values()[0].params[self.agent.obj_list[obj_ind]]
            targ = self.agent.plans.values()[0].params[self.agent.targ_list[targ_ind]]

            plan = self.agent.plans[self.tasks[task_ind], obj.name]
            if self._cost_f(state, self.tasks[task_ind], [obj, targ], self.agent.targets[self.condition], plan, active_ts=(0,0)) == 0:
                return children[next_ind]

        return None

    def _default_choose_next(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        if node.has_unexplored():
            new_nodes = []
            for i in range(node.num_tasks):
                for j in range(node.num_objs):
                    for k in range(node.num_targs):
                        if node.get_child(i, j, k) is None:
                            new_node = self._simulate_from_unexplored(state, node, prev_sample, exclude_hl, use_distilled, debug=debug)
                            if new_node is not None:
                                new_nodes.append(new_node)
            if len(new_nodes):
                val = max(map(lambda n: n.value, new_nodes))
                if debug:
                    print 'Chose unexplored node'
                return None, val

        next_node = self._select_from_explored(state, node, exclude_hl, debug=debug)
        if next_node is None:
            return next_node, -np.inf
        if debug:
            print 'Chose explored node.'
        return next_node, next_node.value / next_node.n_explored

    def sample(self, task, cur_state, target, plan, num_samples, use_distilled=True, node=None, save=True, debug=False):
        samples = []
        old_traj_hist = self.agent.get_hist()

        for n in range(self.num_samples):
            self.agent.reset_hist(deepcopy(old_traj_hist))
            samples.append(self.agent.sample_task(self.rollout_policy[task], self.condition, cur_state, (task, target[0].name, target[1].name), noisy=True))

        if use_distilled:
            for n in range(self.num_distilled_samples):
                self.agent.reset_hist(deepcopy(old_traj_hist))
                samples.append(self.agent.sample_task(self.distilled_policy, self.condition, cur_state, (task, target[0].name, target[1].name), use_prim_obs=True, noisy=True))

        sample_costs = {}
        for sample in samples:
            sample_costs[sample] = self._cost_f(sample.get_X(), task, target, self.agent.targets[self.condition], plan)
            sample.plan = plan

        lowest_cost_ind = np.argmin(sample_costs.values())
        lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]

        opt_fail = False

        if save:
            self.agent.add_sample_batch(samples, task)
        cur_state = lowest_cost_sample.get_X(t=lowest_cost_sample.T-1)
        self.agent.reset_hist(lowest_cost_sample.get_U()[-self.agent.hist_len:].tolist())

        return lowest_cost_sample, cur_state

    def simulate(self, state, use_distilled=True, debug=False):
        current_node = self.root
        path = []
        samples = []

        success = True
        cur_state = state.copy()
        prev_sample = None
        terminated = False
        iteration = 0
        exclude = []
        path_value = None
        while True:
            if debug:
                print "Taking simulation step"
            if self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0]) == 0 or current_node.depth >= self.max_depth:
                break

            next_node, path_value = self._choose_next(cur_state, current_node, prev_sample, use_distilled, debug=debug)

            if next_node == None:
                break

            task = self.tasks[next_node.get_task()]
            obj_name = self.agent.obj_list[next_node.get_obj()]
            targ_name = self.agent.targ_list[next_node.get_targ()]

            obj = self.agent.plans.values()[0].params[obj_name]
            targ = self.agent.plans.values()[0].params[targ_name]
            target = [obj, targ]

            plan = self._plan_f(task, target)
            if self._cost_f(cur_state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
                break

            next_sample, cur_state = self.sample(task, cur_state, target, plan, self.num_samples, use_distilled, debug=debug)

            current_node.sample_links[next_sample] = prev_sample # Used to retrace paths
            prev_sample = next_sample
 
            current_node = next_node
            path.append(current_node)

            iteration += 1


        if path_value is None:
            path_value = self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0])
        path = []
        while current_node is not self.root:
            path.append(prev_sample)
            prev_sample.task_cost = path_value
            prev_sample = current_node.parent.sample_links[prev_sample]
            current_node.update_value(-path_value)
            current_node = current_node.parent

            path.reverse()
        return path

    def simulate_from_next(self, node, state, prev_sample, num_samples=1, save=False, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print "Running simulate from next"
        task_ind = node.get_task()
        obj_ind = node.get_obj()
        targ_ind = node.get_targ()
        cost, samples = self._simulate_from_next(task_ind, obj_ind, targ_ind, node.depth, state, self.prob_func, [], num_samples, save, exclude_hl, use_distilled, [], debug=debug)
        if len(samples):
            node.sample_links[samples[0]] = prev_sample
        else:
            samples.append(None)
        return cost, samples[0]

    def _simulate_from_next(self, task_ind, obj_ind, targ_ind, depth, state, prob_func, samples, num_samples=1, save=False, exclude_hl=[], use_distilled=True, exclude=[], debug=False):
        task = self.tasks[task_ind]
        new_samples = []

        obj = self.agent.plans.values()[0].params[self.agent.obj_list[obj_ind]]
        targ = self.agent.plans.values()[0].params[self.agent.targ_list[targ_ind]]

        target = [obj, targ]
        if target[0] == None:
            return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task, obj.name]), samples 

        plan = self._plan_f(task, target)
        if self._cost_f(state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
            return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans.values()[0]), samples

        next_sample, end_state = self.sample(task, state, target, plan, num_samples=num_samples, use_distilled=use_distilled, save=save, debug=debug)

        samples.append(next_sample)

        path_value = self._goal_f(end_state, self.agent.targets[self.condition], self.agent.plans.values()[0])
        if path_value == 0 or depth >= self.max_depth:
            for sample in samples:
                sample.task_cost = path_value
            return path_value, samples

        
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, end_state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))
        if sum(task_distr) > 0:
            task_distr = task_distr / np.sum(task_distr)
        if sum(obj_distr) > 0:
            obj_distr = obj_distr / np.sum(obj_distr)
        if sum(targ_distr) > 0:
            targ_distr = targ_distr / np.sum(targ_distr)       

        
        if sum(task_distr) > 0 and np.random.sample() > self.soft_decision:
            next_task_ind = np.random.choice(range(len(task_distr)), p=task_distr)
        else:
            next_task_ind = np.argmax(task_distr)
        
        if sum(obj_distr) > 0 and np.random.sample() > self.soft_decision:
            next_obj_ind = np.random.choice(range(len(obj_distr)), p=obj_distr)
        else:
            next_obj_ind = np.argmax(obj_distr)  

        if sum(targ_distr) > 0 and np.random.sample() > self.soft_decision:
            next_targ_ind = np.random.choice(range(len(targ_distr)), p=targ_distr)
        else:
            next_targ_ind = np.argmax(targ_distr)

        return self._simulate_from_next(next_task_ind, next_obj_ind, next_targ_ind, depth+1, end_state, prob_func, samples, use_distilled=use_distilled, debug=debug)
