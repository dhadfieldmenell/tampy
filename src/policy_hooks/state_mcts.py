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


class StateMCTS:
    def __init__(self, tasks, prob_func, plan_f, cost_f, goal_f, target_f, encode_f, value_f, rollout_policy, distilled_policy, agent, branch_factor, num_samples, num_distilled_samples, choose_next=None, soft_decision=1.0, C=2, max_depth=20, always_opt=False):
        self.tasks = tasks
        self._prob_func = prob_func
        self.root = MCTSNode((-1, -1, -1), 0, None, len(tasks), len(agent.obj_list), len(agent.targ_list))
        self.max_depth = max_depth
        self.rollout_policy = rollout_policy
        self.distilled_policy = distilled_policy
        self.agent = agent
        self.soft_decision = soft_decision
        self.C = C
        self.branch_factor = branch_factor
        self.num_samples = num_samples
        self.num_distilled_samples = num_distilled_samples
        self._choose_next = choose_next if choose_next != None else self._default_choose_next
        self._plan_f = plan_f
        self._cost_f = cost_f 
        self._goal_f = goal_f
        self._target_f = target_f
        self._encode_f = encode_f
        self._value_f = value_f
        self.hl_state_values = {}
        self.n_trans = {}
        self.visit_count = {}
        # self.node_check_f = lambda n: n.value/n.n_explored+self.C*np.sqrt(np.log(n.parent.n_explored)/n.n_explored) if n != None else -np.inf
        self._opt_cache = {}
        self.always_opt = always_opt


    def update_hl_info(self, state, task, value):
        if (state, task) not in self.n_trans:
            self.n_trans[(state, task)] = 1.
        else:
            self.n_trans[(state, task)] += 1
        if (state, task) not in self.hl_state_values:
            self.hl_state_values[(state, task)] = value
        else:
            self.hl_state_values[(state, task)] = (self.n_trans[(state, task)] * self.hl_state_values[(state, task)] + value) / self.n_trans[(state, task)]

        if state in self.visit_count:
            self.visit_count[state] += 1
        else:
            self.visit_count[state] = 1.

    def prob_func(self, prim_obs):
        prim_obs = prim_obs.reshape((1, -1))
        return self._prob_func(prim_obs)

    def value_func(self, obs):
        obs = obs.reshape((1, -1))
        return self._value_f(obs)

    def node_check_f(self, task_ind, obj_ind, targ_ind, state):
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
        obj_name, targ_name = self.agent.obj_list[obj_ind], self.agent.targ_list[targ_ind]
        sample.set(OBJ_POSE_ENUM, state[self.agent.plans.values()[0].state_inds[obj_name, 'pose']], 0)
        sample.set(TARG_POSE_ENUM, self.agent.targets[self.condition][targ_name].copy(), 0)
        obs = sample.get_obs(t=0)
        prim_obs = sample.get_prim_obs(t=0)
        hl_state = self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition])
        q_expl_value = self.value_func(obs) if (hl_state, (task_ind, obj_ind, targ_ind)) not in self.hl_state_values else self.hl_state_values[(hl_state, (task_ind, obj_ind, targ_ind))]
        q_pred_value = self.value_func(obs)
        policy_distr = self.prob_func(prim_obs)
        prob = policy_distr[0][task_ind] * policy_distr[1][obj_ind] * policy_distr[2][targ_ind]
        n_visits = self.n_trans[(hl_state, (task_ind, obj_ind, targ_ind))] if (hl_state, (task_ind, obj_ind, targ_ind)) in self.n_trans else 0.
        visit_count = self.visit_count[hl_state] if hl_state in self.visit_count else 1
        # import ipdb; ipdb.set_trace()
        return (q_expl_value + q_pred_value) + self.C * np.sqrt(visit_count) / (1. + n_visits)

    def print_run(self, state, use_distilled=True):
        path = self.simulate(state.copy(), use_distilled, debug=False)
        print 'Testing rollout of MCTS'
        for sample in path:
            task = self.tasks[np.argmax(sample.get(TASK_ENUM, t=0))]
            obj = self.agent.obj_list[np.argmax(sample.get(OBJ_ENUM, t=0))]
            targ = self.agent.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]
            print task, obj, targ
            print sample.get_X()
        print 'End of MCTS rollout.\n\n'

    def run(self, state, condition, num_rollouts=20, use_distilled=True, hl_plan=None, new_policies=None, debug=False):
        if new_policies != None:
            self.rollout_policy = new_policies
        self.condition = condition
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

    def _default_choose_next(self, state, prev_sample, exclude_hl=[], use_distilled=True, debug=False):
        parameterizations, values = [], []
        for i in range(len(self.tasks)):
            for j in range(len(self.agent.obj_list)):
                for k in range(len(self.agent.targ_list)):
                    parameterizations.append((i, j, k))
                    values.append(self.node_check_f(i, j, k, state))

        values = np.array(values)
        p = parameterizations[np.argmax(values)]
        values[np.argmax(values)] = -np.inf
        obj = self.agent.plans.values()[0].params[self.agent.obj_list[p[1]]]
        targ = self.agent.plans.values()[0].params[self.agent.targ_list[p[2]]]
        cost = self._cost_f(state, self.tasks[p[0]], [obj, targ], self.agent.targets[self.condition], self.agent.plans[self.tasks[p[0]], obj.name], active_ts=(0,0), debug=False)
        # hl_state = self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition], (self.tasks[0], obj.name, targ.name))
        while (cost > 0 or p in exclude_hl) and np.any(values > -np.inf):
            p = parameterizations[np.argmax(values)]
            values[np.argmax(values)] = -np.inf
            obj = self.agent.plans.values()[0].params[self.agent.obj_list[p[1]]]
            targ = self.agent.plans.values()[0].params[self.agent.targ_list[p[2]]]
            cost = self._cost_f(state, self.tasks[p[0]], [obj, targ], self.agent.targets[self.condition], self.agent.plans[self.tasks[p[0]], obj.name], active_ts=(0,0), debug=False)
            # hl_state = self._encode_f(state, self.agent.plans.values()[0], self.agent.targets[self.condition], (self.tasks[0], obj.name, targ.name))

        if cost > 0:
            return None

        return p

    def sample(self, task, cur_state, target, plan, num_samples, use_distilled=True, node=None, save=True, debug=False):
        samples = []
        old_traj_hist = self.agent.get_hist()

        if self.always_opt:
            self.agent.reset_hist(deepcopy(old_traj_hist))
            sample, failed, success = self.agent.sample_optimal_trajectory(cur_state, task, self.condition, fixed_targets=target)
            if success:
                self.agent.add_sample_batch([sample], task)
                self.agent.reset_hist(deepcopy(old_traj_hist))
                return sample, sample.get_X(sample.T-1)
            else:
                return None, cur_state

        for n in range(self.num_samples):
            self.agent.reset_hist(deepcopy(old_traj_hist))
            samples.append(self.agent.sample_task(self.rollout_policy[task], self.condition, cur_state, (task, target[0].name, target[1].name), noisy=True))

        if use_distilled and self.distilled_policy.scale is not None:
            for n in range(self.num_distilled_samples):
                self.agent.reset_hist(deepcopy(old_traj_hist))
                samples.append(self.agent.sample_task(self.distilled_policy, self.condition, cur_state, (task, target[0].name, target[1].name), use_prim_obs=True, noisy=True))

        # sample_costs = {}
        # for sample in samples:
        #     sample_costs[sample] = self._cost_f(sample.get_X(), task, target, self.agent.targets[self.condition], plan)
        #     sample.plan = plan

        # lowest_cost_ind = np.argmin(sample_costs.values())
        # lowest_cost_sample = sample_costs.keys()[lowest_cost_ind]
        sample = samples[0]

        opt_fail = False

        if save:
            self.agent.add_sample_batch(samples, task)
        cur_state = sample.get_X(t=sample.T-1)
        self.agent.reset_hist(sample.get_U()[-self.agent.hist_len:].tolist())

        return sample, cur_state

    def simulate(self, state, use_distilled=True, debug=False):
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
                print "Taking simulation step"
            if self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0]) == 0:
                break


            next_task = self._choose_next(cur_state, prev_sample, exclude_hl, use_distilled, debug=debug)
            if next_task is None:
                break
                
            if np.any(np.abs(cur_state) > 1e2):
                import ipdb; ipdb.set_trace()

            task = self.tasks[next_task[0]]
            obj_name = self.agent.obj_list[next_task[1]]
            targ_name = self.agent.targ_list[next_task[2]]

            obj = self.agent.plans.values()[0].params[obj_name]
            targ = self.agent.plans.values()[0].params[targ_name]
            target = [obj, targ]

            plan = self._plan_f(task, target)

            next_sample, cur_state = self.sample(task, cur_state, target, plan, self.num_samples, use_distilled, debug=debug)
            path.append(next_sample)

            if len(path) >= self.max_depth or next_sample is None:
                break

            path.append(next_sample)
            # hl_state = self._encode_f(cur_state, plan, self.agent.targets[self.condition])
            if task not in exclude_hl:
                exclude_hl += [task]
            else:
                break

            iteration += 1

        if path_value is None:
            path_value = self._goal_f(cur_state, self.agent.targets[self.condition], self.agent.plans.values()[0])

        for sample in path:
            sample.task_cost = path_value
            hl_state = self._encode_f(sample.get_X(0), self.agent.plans.values()[0], self.agent.targets[self.condition])
            self.update_hl_info(hl_state, (sample.task_ind, sample.obj_ind, sample.targ_ind), path_value)

        return path
