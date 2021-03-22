import numpy as np

from sco.expr import EqExpr

from policy_hooks.sample import Sample
from policy_hooks.search_node import *
from policy_hooks.utils.policy_solver_utils import *


ROLL_PRIORITY = 5

class RolloutSupervisor():
    def __init__(self, agent, policy_opt, hyperparams, check_pre=False, check_mid=False, check_post=False, check_random=False, \
                 neg_precond=False, neg_postcond=False, soft=False, eta=5):
        self.agent = agent
        self.policy_opt = policy_opt
        self.hyperparams = hyperparams
        self.check_precond = check_pre
        self.check_midcond = check_mid
        self.check_postcond = check_post
        self.check_random = check_random
        self.soft = soft
        self.eta = eta
        self.neg_precond = neg_precond
        self.neg_postcond = neg_postcond
        self.reset()
        self.t_per_task = 20
        self.s_per_task = 3
        if self.agent.retime:
            self.s_per_task *= 3


    def reset(self):
        self.cur_tasks = []
        self.counts = [0]
        self.switch_pts = [(0,0)]
        self.switch_x = []
        self.task_successes = {task: [] for task in self.agent.task_list}
        self.postcond_viols = []
        self.precond_viols = []
        self.expl_precond_viols = []
        self.cur_ids = [0]
        self.neg_samples = []
        self.ll_nodes = []
        self.hl_nodes = []
        self.n_fails = 0
        self.n_suc = 0
        self.tol = 2e-3
        self.postcond_info = []
        self.fail_types = {}
        self.postcond_costs = {task: [] for task in self.agent.task_list}

    
    def task_f(self, sample, t, curtask):
        targets = sample.targets
        task = self.get_task(sample.get_X(t=t), sample.targets, curtask, self.soft)
        truetask = task
        truecurtask = curtask
        task = tuple([val for val in task if np.isscalar(val)])
        curtask = tuple([val for val in curtask if np.isscalar(val)])

        val = 1 - self.agent.goal_f(0, sample.get_X(t), targets)
        if val > 0.999 or self.counts[-1] >= self.s_per_task * self.t_per_task:
            self.cur_tasks.append(curtask)
            self.counts.append(self.counts[-1]+1)
            return truecurtask

        postcost = None
        precost = None
        if self.check_postcond:
            postcost = self.agent.postcond_cost(sample, curtask, t, x0=self.switch_x[-1])
            if task == curtask and postcost < 1e-4:
                self.postcond_viols.append((self.cur_ids[-1], t))
                if self.neg_postcond: self.neg_samples.append((sample, t, truetask))

        #    n_tries = 0
        #    postcost = self.agent.postcond_cost(sample, curtask, t)
        #    #cur_eta = self.eta
        #    #eta_scale = 0.9
        #    #while n_tries < 10 and task == curtask and postcost < 1e-3:
        #    #    task = self.get_task(sample.get_X(t=t), sample.targets, curtask, True, eta=cur_eta)
        #    #    cur_eta *= eta_scale
        #    #    n_tries += 1

        if task != curtask and self.check_postcond:
            postcost = self.agent.postcond_cost(sample, curtask, t, x0=self.switch_x[-1])
            if postcost > 1e-4:
                if self.neg_postcond: self.neg_samples.append((sample, t, truetask))
                task = curtask
            else:
                self.postcond_costs[self.agent.task_list[curtask[0]]].append(postcost)
        
        if task != curtask and self.check_precond:
            precost = self.agent.precond_cost(sample, task, t, tol=self.tol)
            if precost > 1e-4:
                self.precond_viols.append((self.cur_ids[-1], t))
            
            n_tries = 0
            cur_eta = self.eta
            eta_scale = 0.9
            while n_tries < 20 and task != curtask and precost > 1e-4:
                if self.neg_precond: self.neg_samples.append((sample, t, truetask))
                task = self.get_task(sample.get_X(t=t), sample.targets, curtask, True, eta=cur_eta)
                truetask = task
                task = tuple([val for val in task if np.isscalar(val)])
                precost = self.agent.precond_cost(sample, task, t, tol=self.tol)
                cur_eta *= eta_scale
                n_tries += 1

            if precost > 1e-4 and task != curtask:
                self.expl_precond_viols.append((self.cur_ids[-1], t))
                task = curtask

        if task == curtask:
            truetask = list(truetask)
            self.counts.append(self.counts[-1]+1)

            for ind in range(len(truetask)):
                if np.isscalar(truetask[ind]):
                    truetask[ind] = truecurtask[ind]

            truetask = tuple(truetask)

        else:
            self.counts.append(0)
            self.switch_pts.append((self.cur_ids[-1], t))
            self.switch_x.append(sample.get_X(t=t))
            self.task_successes[self.agent.task_list[curtask[0]]].append(1)

        self.cur_tasks.append(task)
        return truetask


    def rollout(self, x, targets, node, tol=4e-3):
        self.agent.target_vecs[0] = targets
        self.agent.reset_to_state(x)
        self.tol = tol

        ntask = len(self.agent.task_list)
        rlen = ntask * self.agent.num_objs
        if self.agent.retime:
            rlen *= 4

        self.adj_eta = True
        l = self.get_task(x, targets, None, self.soft)
        l = tuple([val for val in l if np.isscalar(val)])
        self.cur_tasks.append(l)

        s, t = 0, 0
        val = 0
        state = x
        path = []
        last_switch = 0
        self.switch_x.append(state)

        while val < 1 and s < rlen:
            curtask = tuple([val for val in self.cur_tasks[-1] if np.isscalar(val)])
            task_name = self.agent.task_list[self.cur_tasks[-1][0]]
            pol = self.agent.policies[task_name]
            hor = self.t_per_task
            sample = self.agent.sample_task(pol, 0, state, self.cur_tasks[-1], skip_opt=True, hor=hor, task_f=self.task_f, policies=self.agent.policies)
            path.append(sample)
            state = sample.get(STATE_ENUM, t=sample.T-1)

            s += 1
            self.cur_ids.append(s)
            val = 1 - self.agent.goal_f(0, sample.get_X(sample.T-1), targets)
            if self.counts[-1] >= self.s_per_task * self.t_per_task:
                print('TIMEOUT TERMINATING', self.counts[-1], curtask)
                break
            
            if len(self.expl_precond_viols):
                print('PRECOND TERMINATING', self.counts[-1], curtask, self.switch_pts[-1])
                break

        if len(path):
            val = 1 - self.agent.goal_f(0, path[-1].get_X(path[-1].T-1), targets)
        else:
            val = 0

        for step in path: step.source_label = 'rollout'
        self.adj_eta = False

        self.postcond_info.append(val)
        postcost = self.agent.postcond_cost(path[-1], self.cur_tasks[-1], path[-1].T-1, tol=self.tol, x0=self.switch_x[-1])
        self.postcond_costs[self.agent.task_list[self.cur_tasks[-1][0]]].append(postcost)
        if postcost > 1e-4:
            self.task_successes[self.agent.task_list[self.cur_tasks[-1][0]]].append(0)
        else:
            self.task_successes[self.agent.task_list[self.cur_tasks[-1][0]]].append(1)

        train_pts = []
        fail_type = 'successful_rollout'

        if len(self.precond_viols):
            fail_type = 'rollout_precondition_failure'
            rand_ind = np.random.choice(range(len(self.precond_viols)))
            train_pts.append(tuple(self.precond_viols[rand_ind]) + (fail_type,))

        if self.check_postcond and val < 1-1e-4:
            fail_type = 'rollout_postcondition_failure'
            bad_pt = self.switch_pts[-1]
            train_pts.append(tuple(bad_pt) + (fail_type,))

        if self.check_random:
            ind = np.random.choice(range(len(self.switch_pts)))
            train_pts.append(tuple(self.switch_pts[ind]) + ('rollout_random_switch',))

        self.parse_midcond(path)

        if val >= 0.999:
            print('Success in rollout. Pre: {} Post: {} Mid: {} Random: {}'.format(self.check_precond, \
                                                                        self.check_postcond, \
                                                                        self.check_midcond, \
                                                                        self.check_random))
            
            self.agent.add_task_paths([path])
            n_plans = self.hyperparams['policy_opt']['buffer_sizes']['n_rollout']
            with n_plans.get_lock():
                n_plans.value += 1
            n_plans = self.hyperparams['policy_opt']['buffer_sizes']['n_total']
            with n_plans.get_lock():
                n_plans.value += 1

        else:
            print('Failure in supervised rollout. {}'.format(train_pts))

        self.parse_train_pts(train_pts, path, targets, node)
        return val, path
 

    def get_task(self, state, targets, prev_task, soft=False, eta=None):
        if eta is None: eta = self.eta
        sample = Sample(self.agent)
        sample.set_X(state.copy(), t=0)
        self.agent.fill_sample(0, sample, sample.get(STATE_ENUM, 0), 0, prev_task, fill_obs=True, targets=targets)
        distrs = self.primitive_call(sample.get_prim_obs(t=0), soft, eta=eta, t=0, task=prev_task)
        for d in distrs:
            for i in range(len(d)):
                d[i] = round(d[i], 5)

        ind = []
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        enums = list(opts.keys())
        for i, d in enumerate(distrs):
            enum = enums[i]
            if not np.isscalar(opts[enum]):
                val = np.max(d)
                ind.append(np.random.choice([i for i in range(len(d)) if d[i] >= val]))
            else:
                ind.append(d)
        next_label = tuple(ind)
        return next_label


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None, adj_eta=False):
        if adj_eta: eta *= self.agent.eta_scale
        distrs = self.policy_opt.task_distr(prim_obs, eta)
        if not soft: return distrs

        out = []
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        enums = list(opts.keys())
        for ind, d in enumerate(distrs):
            enum = enums[ind]
            if not np.isscalar(opts[enum]):
                p = d / np.sum(d)
                ind = np.random.choice(list(range(len(d))), p=p)
                d[ind] += 1e2
                d /= np.sum(d)
            out.append(d)
        return out


    def parse_midcond(self, path):
        if not self.check_midcond: return
        bad_pt = self.switch_pts[-1]
        curtask = tuple([val for val in self.cur_tasks[-1] if np.isscalar(val)])
        plan = self.agent.plans[curtask]
        st = bad_pt[1]
        x0 = path[bad_pt[0]].get(STATE_ENUM, st).copy()
        targets = path[bad_pt[0]].targets
        traj, steps, _, env_states = self.agent.reverse_retime(path[bad_pt[0]:], (0, plan.horizon-1), label=True, start_t=st)

        for t in range(len(traj)-1):
            t = min(t, plan.horizon-1)
            set_params_attrs(plan.params, self.agent.state_inds, traj[t], t)

        set_params_attrs(plan.params, self.agent.state_inds, path[bad_pt[0]].get_X(t=st), 0)
        self.agent.set_symbols(plan, self.cur_tasks[-1], targets=targets)
        failed_preds = plan.get_failed_preds(tol=self.tol, active_ts=(0, plan.horizon-1))
        failed_preds = [p for p in failed_preds if (p[1]._rollout or (not p[1]._nonrollout and type(p[1].expr) is not EqExpr))]

        if len(failed_preds):
            valid_ts = np.ones(plan.horizon)
            for p in failed_preds:
                for ts in range(max(0, p[2]+p[1].active_range[0]), \
                                min(p[2]+p[1].active_range[1], len(valid_ts))+1):
                    valid_ts[ts] = 0.

            fail_t = len(valid_ts) - 1
            while fail_t > 0 and valid_ts[fail_t] == 0:
                fail_t -= 1
        else:
            fail_t = plan.horizon - 1

        fail_t = max(0, fail_t)
        fail_t = min(fail_t, plan.horizon-3)
        if len(failed_preds):
            fail_type = 'rollout_midcondition_failure'
            if fail_t < len(steps):
                fail_s = bad_pt[0] + steps[fail_t]
            else:
                fail_s = bad_pt[0]

            print('MID COND:', fail_s, fail_t, bad_pt, failed_preds)
            initial, goal = self.agent.get_hl_info(x0.copy(), targets)
            plan.start = 0
            new_node = LLSearchNode(plan.plan_str, 
                                    prob=plan.prob, 
                                    domain=plan.domain,
                                    initial=plan.prob.initial,
                                    priority=ROLL_PRIORITY,
                                    ref_plan=plan,
                                    targets=targets,
                                    x0=x0,
                                    expansions=1,
                                    label='rollout_midcondition_failure',
                                    refnode=None,
                                    freeze_ts=fail_t,
                                    hl=False,
                                    ref_traj=traj,
                                    env_state=env_states,
                                    nodetype='dagger')
            self.ll_nodes.append(new_node)


    def parse_train_pts(self, train_pts, path, targets, node):
        val = 1 - self.agent.goal_f(0, path[-1].get_X(path[-1].T-1), path[-1].targets)
        x0 = path[0].get_X(0)
        for s, t, fail_type in train_pts:
            if val < 1:
                self.n_fails += 1
                if fail_type not in self.fail_types:
                    self.fail_types[fail_type] = 1.
                else:
                    self.fail_types[fail_type] += 1.

                state = x0
                if len(path):
                    t = min(path[s].T-1, t)
                    state = path[s].get(STATE_ENUM, t)

                initial, goal = self.agent.get_hl_info(state.copy(), targets)
                concr_prob = node.concr_prob
                abs_prob = self.agent.hl_solver.translate_problem(concr_prob, initial=initial, goal=goal)
                set_params_attrs(concr_prob.init_state.params, self.agent.state_inds, state.copy(), 0)
                hlnode = HLSearchNode(abs_prob,
                                      node.domain,
                                      concr_prob,
                                      priority=ROLL_PRIORITY,
                                      prefix=None,
                                      llnode=None,
                                      expansions=node.expansions,
                                      label=fail_type,
                                      x0=state,
                                      targets=targets,
                                      nodetype='dagger')
                self.hl_nodes.append(hlnode)


