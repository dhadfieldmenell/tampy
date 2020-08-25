from copy import copy, deepcopy
from datetime import datetime
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
        if parent is not None:
            parent.add_child(self)

    def is_leaf(self):
        return self.is_leaf()

    def get_task(self):
        return self.label[0]

    def get_obj(self):
        return self.label[1]

    def get_targ(self):
        return self.label[2]

    def update_value(self, new_value):
        # self.value = (self.value*self.n_explored + new_value) / (self.n_explored + 1)
        # if new_value == 0:
        #     new_value = 1
        # else:
        #     new_value = 0

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
                children.extend([n for n in obj if n is not None])
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
    def __init__(self, tasks, prob_func, plan_f, cost_f, goal_f, target_f, encode_f, value_f, rollout_policy, distilled_policy, condition, agent, branch_factor, num_samples, num_distilled_samples, choose_next=None, soft_decision=1.0, C=2, max_depth=20, always_opt=False):
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
        self.branch_factor = branch_factor
        self.num_samples = 1
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = choose_next if choose_next != None else self._default_choose_next
        self._plan_f = plan_f
        self._cost_f = cost_f
        self._goal_f = goal_f
        self._target_f = target_f
        self._encode_f = encode_f
        self._value_f = value_f
        # self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self.targets = np.zeros((self.agent.target_dim))
        for target_name in self.agent.targets[self.condition]:
            target = list(self.agent.plans.values())[0].params[target_name]
            if (target.name, 'value') in self.agent.target_inds:
                self.targets[self.agent.target_inds[target.name, 'value']] = self.agent.targets[condition][target.name]

        self._opt_cache = {}
        self.always_opt = always_opt

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
                                len(self.agent.obj_list),
                                len(self.agent.targ_list))

            else:
                node.update_value(int(success))

    def node_check_f(self, task_ind, obj_ind, targ_ind, state, parent):
        child = parent.get_child(task_ind, obj_ind, targ_ind)
        sample = Sample(self.agent)
        sample.set(STATE_ENUM, state.copy(), 0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
        sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
        task_vec = np.zeros((len(self.tasks)), dtype=np.float32)
        task_vec[task_ind] = 1.
        obj_vec = np.zeros((len(self.agent.obj_list)), dtype='float32')
        targ_vec = np.zeros((len(self.agent.targ_list)), dtype='float32')
        obj_vec[obj_ind] = 1.
        targ_vec[targ_ind] = 1.
        sample.set(OBJ_ENUM, obj_vec, 0)
        sample.set(TARG_ENUM, targ_vec, 0)
        sample.set(TASK_ENUM, task_vec, 0)
        ee_pose = state[self.agent.state_inds['pr2', 'pose']]
        obj_pose = state[self.agent.state_inds[self.agent.obj_list[obj_ind], 'pose']] - ee_pose
        targ_pose = self.agent.targets[self.condition][self.agent.targ_list[targ_ind]] - ee_pose
        sample.set(OBJ_POSE_ENUM, obj_pose.copy(), 0)
        sample.set(TARG_POSE_ENUM, targ_pose.copy(), 0)
        sample.set(EE_ENUM, ee_pose.copy(), 0)

        if LIDAR_ENUM in self.agent._hyperparams['obs_include']:
            plan = list(self.agent.plans.values())[0]
            set_params_attrs(plan.params, plan.state_inds, state, 0)
            lidar = self.agent.dist_obs(plan, 0)
            sample.set(LIDAR_ENUM, lidar.flatten(), 0)

        obs = sample.get_obs(t=0)
        prim_obs = sample.get_prim_obs(t=0)
        val_obs = sample.get_val_obs(t=0)
        q_value = self.value_func(val_obs)[1] if child is None else child.value
        policy_distr = self.prob_func(prim_obs)
        prob = policy_distr[0][task_ind] * policy_distr[1][obj_ind] * policy_distr[2][targ_ind]
        child_explored = child.n_explored if child is not None else 0
        # print task_ind, obj_ind, targ_ind, q_value, self.value_func(obs)[1]
        # return self.value_func(val_obs)[1] + self.C * np.sqrt(parent.n_explored) / (1 + child_explored)
        # return q_value + self.C * self.value_func(obs)[1] / (1 + child_explored)
        return q_value + self.C * np.sqrt(np.log(parent.n_explored) / (1 + child_explored))

    def print_run(self, state, use_distilled=True):
        path = self.simulate(state.copy(), use_distilled, debug=False)
        print('Testing rollout of MCTS')
        for sample in path:
            task = self.tasks[np.argmax(sample.get(TASK_ENUM, t=0))]
            obj = self.agent.obj_list[np.argmax(sample.get(OBJ_ENUM, t=0))]
            targ = self.agent.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]
            print(task, obj, targ)
            print(sample.get_X())
        print('End of MCTS rollout.\n\n')

    def run(self, state, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        opt_val = np.inf
        paths = []
        for n in range(num_rollouts):
            self.agent.reset_hist()
            print("MCTS Rollout {0} for condition {1}.\n".format(n, self.condition))
            next_path = self.simulate(state.copy(), use_distilled, debug=debug)
            print("Finished Rollout {0} for condition {1}.\n".format(n, self.condition))
            if len(next_path):
                end = next_path[-1]
                new_opt_value = self._goal_f(end.end_state, self.agent.targets[self.condition], list(self.agent.plans.values())[0])
                if new_opt_value == 0: paths.append(next_path)
                opt_val = np.minimum(new_opt_value, opt_val)

        self.agent.add_task_paths(paths)
        return opt_val

    def _simulate_from_unexplored(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, task=None, debug=False):
        if debug:
            print('Simulating from unexplored children.')
        if task is None:
            sample = Sample(self.agent)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TARGETS_ENUM, self.agent.target_vecs[self.condition].copy(), 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
            task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))
            obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[np.argmax(obj_distr)]]
            targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[np.argmax(targ_distr)]]
        else:
            task_distr = np.zeros(len(self.tasks))
            task_distr[task[0]] = 1.
            obj_distr = np.zeros(len(self.agent.obj_list))
            obj_distr[task[1]] = 1.
            targ_distr = np.zeros(len(self.agent.targ_list))
            targ_distr[task[2]] = 1.

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

                    obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[next_obj_ind]]
                    targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[next_targ_ind]]

                    if node.get_child(next_task_ind, next_obj_ind, next_targ_ind) != None:
                        continue

                    next_encoding = self._encode_f(state, self.agent.plans[self.tasks[next_task_ind], obj.name], self.agent.targets[self.condition], (self.tasks[next_task_ind], obj.name, targ.name))
                    precond_cost = self._cost_f(state, self.tasks[next_task_ind], [obj, targ], self.agent.targets[self.condition], self.agent.plans[self.tasks[next_task_ind], obj.name], active_ts=(0,0), debug=debug)
                    if next_encoding in exclude_hl or precond_cost > 0:
                        continue

                    self.agent.reset_hist(deepcopy(old_traj_hist))
                    next_node = MCTSNode((next_task_ind, next_obj_ind, next_targ_ind),
                                         0,
                                         node,
                                         len(self.tasks),
                                         len(self.agent.obj_list),
                                         len(self.agent.targ_list))
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

    def _select_from_explored(self, state, node, exclude_hl=[], task=None, debug=False):
        if debug:
            print("Selecting from explored children.")
            print("State: ", state)

        if task is None:
            children = node.get_explored_children()
            children_distr = list(map(self.node_check_f, children))
        else:
            children = [node.get_child(*task)]
            assert children[0] is not None
            children_distr = np.ones(1)

        while np.any(children_distr > -np.inf):
            next_ind = np.argmax(children_distr)
            children_distr[next_ind] = -np.inf
            next_node = children[next_ind]
            task_ind = next_node.get_task()
            obj_ind = next_node.get_obj()
            targ_ind = next_node.get_targ()
            obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[obj_ind]]
            targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[targ_ind]]
            plan = self.agent.plans[self.tasks[task_ind], obj.name]
            if self._cost_f(state, self.tasks[task_ind], [obj, targ], self.agent.targets[self.condition], plan, active_ts=(0,0), debug=debug) == 0:
                if debug:
                    print('Chose explored child.')
                return children[next_ind]

        if debug:
            if task is None:
                print('Preconditions violated for all explored children.')
            else:
                print('Preconditions violated for {0} {1}.'.format(self.tasks[task[0]], self.agent.obj_list[task[1]]))
        return None

    def _default_choose_next(self, state, node, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        if debug:
            print('Choosing next node.')
        parameterizations, values = [], []
        for i in range(node.num_tasks):
            for j in range(node.num_objs):
                for k in range(node.num_targs):
                    parameterizations.append((i, j, k))
                    values.append(self.node_check_f(i, j, k, state, node))

        values = np.array(values)
        p = parameterizations[np.argmax(values)]
        values[np.argmax(values)] = -np.inf
        obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[p[1]]]
        targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[p[2]]]
        cost = self._cost_f(state, self.tasks[p[0]], [obj, targ], self.agent.targets[self.condition], self.agent.plans[self.tasks[p[0]], obj.name], active_ts=(0,0), debug=False)
        while cost > 0 and np.any(values > -np.inf):
            p = parameterizations[np.argmax(values)]
            values[np.argmax(values)] = -np.inf
            obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[p[1]]]
            targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[p[2]]]
            cost = self._cost_f(state, self.tasks[p[0]], [obj, targ], self.agent.targets[self.condition], self.agent.plans[self.tasks[p[0]], obj.name], active_ts=(0,0), debug=False)

        child = node.get_child(*p)
        if debug:
            print('Chose to explore ', p)
        if child is None:
            new_node = self._simulate_from_unexplored(state, node, prev_sample, exclude_hl, use_distilled, task=p, debug=debug)
        else:
            new_node = self._select_from_explored(state, node, exclude_hl, task=p, debug=debug)
        if new_node is None:
            return new_node, -np.inf
        return new_node, new_node.value

    def sample(self, task, cur_state, target, plan, num_samples, use_distilled=True, node=None, save=True, debug=False):
        samples = []
        old_traj_hist = self.agent.get_hist()

        # if self.always_opt:
        #     self.agent.reset_hist(deepcopy(old_traj_hist))
        #     sample, failed, success = self.agent.sample_optimal_trajectory(cur_state, task, self.condition, fixed_targets=target)
        #     if success:
        #         self.agent.add_sample_batch([sample], task)
        #         self.agent.reset_hist(deepcopy(old_traj_hist))
        #         return sample, sample.get_X(sample.T-1)
        #     else:
        #         return None, cur_state

        for n in range(self.num_samples):
            self.agent.reset_hist(deepcopy(old_traj_hist))
            samples.append(self.agent.sample_task(self.rollout_policy[task], self.condition, cur_state, (task, target[0].name, target[1].name), noisy=True))

        # if use_distilled and self.distilled_policy.scale is not None:
        #     for n in range(self.num_distilled_samples):
        #         self.agent.reset_hist(deepcopy(old_traj_hist))
        #         samples.append(self.agent.sample_task(self.distilled_policy, self.condition, cur_state, (task, target[0].name, target[1].name), use_prim_obs=True, noisy=True))

        # sample_costs = {}
        # for sample in samples:
        #     sample_costs[sample] = self._cost_f(sample.get_X(), task, target, self.agent.targets[self.condition], plan)
        #     sample.plan = plan

        # lowest_cost_ind = np.argmin(sample_costs.values())
        # lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]
        lowest_cost_sample = samples[0] # Too time expsenive to run cost check

        opt_fail = False

        if save:
            self.agent.add_sample_batch(samples, task)
        cur_state = lowest_cost_sample.end_state
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
            if self._goal_f(cur_state, self.agent.targets[self.condition], list(self.agent.plans.values())[0]) == 0 or current_node.depth >= self.max_depth:
                break

            next_node, _ = self._choose_next(cur_state, current_node, prev_sample, exclude_hl, use_distilled, debug=debug)

            if next_node == None:
                break

            task = self.tasks[next_node.get_task()]
            obj_name = self.agent.obj_list[next_node.get_obj()]
            targ_name = self.agent.targ_list[next_node.get_targ()]

            obj = list(self.agent.plans.values())[0].params[obj_name]
            targ = list(self.agent.plans.values())[0].params[targ_name]
            target = [obj, targ]

            plan = self._plan_f(task, target)
            if self._cost_f(cur_state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
                break

            next_sample, cur_state = self.sample(task, cur_state, target, plan, self.num_samples, use_distilled, debug=debug)

            if next_sample is None:
                break

            current_node.sample_links[next_sample] = prev_sample # Used to retrace paths
            prev_sample = next_sample

            current_node = next_node
            path.append(current_node)
            exclude_hl += [self._encode_f(cur_state, plan, self.agent.targets[self.condition])]

            iteration += 1
            if np.random.uniform() < early_stop_prob:
                break


        if path_value is None:
            path_value = self._goal_f(cur_state, self.agent.targets[self.condition], list(self.agent.plans.values())[0])
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
        # print 'Entering simulate call:', datetime.now()
        task = self.tasks[task_ind]
        new_samples = []

        obj = list(self.agent.plans.values())[0].params[self.agent.obj_list[obj_ind]]
        targ = list(self.agent.plans.values())[0].params[self.agent.targ_list[targ_ind]]

        target = [obj, targ]
        if target[0] == None or self._encode_f(state, list(self.agent.plans.values())[0], self.agent.targets[self.condition], (task, obj.name, targ.name)) in exclude_hl:
            return self._goal_f(state, self.agent.targets[self.condition], self.agent.plans[task, obj.name]), samples

        plan = self._plan_f(task, target)
        if self._cost_f(state, task, target, self.agent.targets[self.condition], plan, active_ts=(0,0)) > 0:
            return self._goal_f(state, self.agent.targets[self.condition], list(self.agent.plans.values())[0]), samples

        next_sample, end_state = self.sample(task, state, target, plan, num_samples=num_samples, use_distilled=use_distilled, save=save, debug=debug)

        if next_sample is None:
            path_value = self._goal_f(end_state, self.agent.targets[self.condition], list(self.agent.plans.values())[0])
            for sample in samples:
                sample.task_cost = path_value
                sample.success = SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples
        samples.append(next_sample)

        path_value = self._goal_f(end_state, self.agent.targets[self.condition], list(self.agent.plans.values())[0])
        hl_encoding = self._encode_f(end_state, list(self.agent.plans.values())[0], self.agent.targets[self.condition])
        if path_value == 0 or depth >= self.max_depth or hl_encoding in exclude_hl:
            for sample in samples:
                sample.task_cost = path_value
                sample.success = SUCCESS_LABEL if path_value == 0 else FAIL_LABEL
            return path_value, samples

        exclude_hl = exclude_hl + [hl_encoding]

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
            next_task_ind = np.random.choice(list(range(len(task_distr))), p=task_distr)
        else:
            next_task_ind = np.argmax(task_distr)

        if sum(obj_distr) > 0 and np.random.sample() > self.soft_decision:
            next_obj_ind = np.random.choice(list(range(len(obj_distr))), p=obj_distr)
        else:
            next_obj_ind = np.argmax(obj_distr)

        if sum(targ_distr) > 0 and np.random.sample() > self.soft_decision:
            next_targ_ind = np.random.choice(list(range(len(targ_distr))), p=targ_distr)
        else:
            next_targ_ind = np.argmax(targ_distr)

        # print 'Leaving simulate call:', datetime.now()
        return self._simulate_from_next(next_task_ind, next_obj_ind, next_targ_ind, depth+1, end_state, prob_func, samples, num_samples=num_samples, save=False, use_distilled=use_distilled, debug=debug)
