import pickle as pickle
from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time
from software_constants import *

from PIL import Image
from scipy.cluster.vq import kmeans2 as kmeans
import tensorflow as tf

from sco.expr import *
from core.internal_repr.plan import Plan
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.server import *
from policy_hooks.search_node import *


ROLL_PRIORITY = 5

class RolloutServer(Server):
    def __init__(self, hyperparams):
        super(RolloutServer, self).__init__(hyperparams)
        self.in_queue = self.rollout_queue
        self.out_queue = self.motion_queue
        self.check_precond = hyperparams['check_precond']
        self.check_postcond = hyperparams['check_postcond']
        self.neg_precond = hyperparams['neg_precond']
        self.neg_postcond = hyperparams['neg_postcond']
        self.check_midcond = hyperparams['check_midcond']
        self.check_random_switch = hyperparams['check_random_switch']
        self.fail_plan = hyperparams['train_on_fail']
        self.fail_mode = hyperparams['fail_mode']
        self.current_id = 0
        self.cur_step = 0
        self.label_type = 'rollout'
        self.adj_eta = False
        self.run_hl_test = hyperparams.get('run_hl_test', False)
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        self.explore_eta = hyperparams['explore_eta']
        self.explore_wt = hyperparams['explore_wt']
        self.explore_n = hyperparams['explore_n']
        self.explore_nmax = hyperparams['explore_nmax']
        self.explore_success = hyperparams['explore_success']
        self.soft = hyperparams['soft_eval']
        self.eta = hyperparams['eta']
        self.ll_rollout_opt = hyperparams['ll_rollout_opt']
        self.hl_rollout_opt = hyperparams['hl_rollout_opt']
        self.hl_test_log = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'hl_test_{0}{1}log.npy'
        self.fail_log = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'failure_{0}_log.txt'.format(self.id)
        self.fail_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_'+'failure_{0}_data.txt'.format(self.id)
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.hl_data = []
        self.fail_data = []
        self.postcond_info = []
        self.postcond_costs = {task: [] for task in self.task_list}
        self.fail_types = {}
        self.n_fails = 0
        self.dagger_log = LOG_DIR + hyperparams['weight_dir'] + '/RolloutInfo_{0}_log.txt'.format(self.id)
        self.last_hl_test = time.time()
        self.task_calls = {task: [] for task in self.task_list}
        self.task_successes = {task: [] for task in self.task_list}
        self.suc_trajs = [] # Each entry should be (traj, list of ts, list of prob under policy)
        self.failed_trajs = [] # Each entry should be (traj, list of ts, list of prob under policy)
        self.prev_suc = [] # Each entry is a state and a list of time, success pairs
        self.prev_fail = [] # Each entry is a state and a list of time, success pairs


    def hl_log_prob(self, path):
        log_l = 0
        for sample in path:
            for t in range(sample.T):
                hl_act = sample.get_prim_out(t=t)
                distrs = self.policy_opt.task_distr(sample.get_prim_obs(t=t), eta=1.)
                ind = 0
                p = 1.
                for d in distrs:
                    u = hl_act[ind:ind+len(d)]
                    p *= d[np.argmax(u)]
                    ind += len(d)
                log_l += np.log(p)
        return log_l


    def get_task(self, state, targets, prev_task, soft=False, eta=None):
        if eta is None:
            eta = self.eta
        sample = Sample(self.agent)
        sample.set_X(state.copy(), t=0)
        self.agent.fill_sample(0, sample, sample.get(STATE_ENUM, 0), 0, prev_task, fill_obs=True, targets=targets)
        distrs = self.primitive_call(sample.get_prim_obs(t=0), soft, eta=eta, t=0, task=prev_task)
        #labels = list(self.agent.plans.keys())
        for d in distrs:
            for i in range(len(d)):
                d[i] = round(d[i], 5)
        #distr = [np.prod([distrs[i][l[i]] for i in range(len(l))]) for l in labels]
        #distr = np.array(distr)
        ind = []
        opts = self.agent.prob.get_prim_choices(self.task_list)
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

    def compare_tasks(self, task1, task2):
        for i in range(len(task1)):
            if type(task1[i]) is int and task1[i] != task2[i]:
                return False
            elif np.sum(np.abs(task1[i]-task2[i])) > 1e-2:
                return False

        return True

    def rollout(self, x, targets, tol=4e-3):
        switch_pts = [(0,0)]
        counts = [0]
        cur_ids = [0]
        cur_tasks = []
        postcond_viols = []
        precond_viols = []
        expl_precond_viols = []
        self.agent.target_vecs[0] = targets
        self.agent.reset_to_state(x)
        neg_samples = []
        init_t = time.time()
        switch_x = [x]

        def task_f(sample, t, curtask):
            task = self.get_task(sample.get_X(t=t), sample.targets, curtask, self.soft)
            truetask = task
            truecurtask = curtask
            task = tuple([val for val in task if np.isscalar(val)])
            curtask = tuple([val for val in curtask if np.isscalar(val)])
            #onehot_task = tuple([val for val in task if np.isscalar(val)])
            #cur_onehot_task = tuple([val for val in curtask if np.isscalar(val)])
            val = 1 - self.agent.goal_f(0, sample.get_X(t), targets)
            if val > 0.999:
                cur_tasks.append(curtask)
                counts.append(counts[-1]+1)
                return truecurtask

            postcost = None
            precost = None
            if self.check_postcond:
                postcost = self.agent.postcond_cost(sample, curtask, t, x0=switch_x[-1])
                if task == curtask and postcost < 1e-4:
                    postcond_viols.append((cur_ids[-1], t))
                    if self.neg_postcond: neg_samples.append((sample, t, truetask))

            #    n_tries = 0
            #    postcost = self.agent.postcond_cost(sample, curtask, t)
            #    #cur_eta = self.eta
            #    #eta_scale = 0.9
            #    #while n_tries < 10 and task == curtask and postcost < 1e-3:
            #    #    task = self.get_task(sample.get_X(t=t), sample.targets, curtask, True, eta=cur_eta)
            #    #    cur_eta *= eta_scale
            #    #    n_tries += 1

            if task != curtask and self.check_postcond:
                postcost = self.agent.postcond_cost(sample, curtask, t, x0=switch_x[-1])
                if postcost > 1e-4:
                    if self.neg_postcond: neg_samples.append((sample, t, truetask))
                    task = curtask
                else:
                    self.postcond_costs[self.task_list[curtask[0]]].append(postcost)
            
            if task != curtask and self.check_precond:
                precost = self.agent.precond_cost(sample, task, t, tol=tol)
                if precost > 1e-4:
                    #if self.neg_precond: neg_samples.append((sample, t, truetask))
                    precond_viols.append((cur_ids[-1], t))
                
                n_tries = 0
                cur_eta = self.eta
                eta_scale = 0.9
                while n_tries < 20 and task != curtask and precost > 1e-4:
                    if self.neg_precond: neg_samples.append((sample, t, truetask))
                    task = self.get_task(sample.get_X(t=t), sample.targets, curtask, True, eta=cur_eta)
                    truetask = task
                    task = tuple([val for val in task if np.isscalar(val)])
                    precost = self.agent.precond_cost(sample, task, t, tol=tol)
                    cur_eta *= eta_scale
                    n_tries += 1

                if precost > 1e-4 and task != curtask:
                    expl_precond_viols.append((cur_ids[-1], t))
                    task = curtask

            if task == curtask:
                truetask = list(truetask)
                counts.append(counts[-1]+1)
                for ind in range(len(truetask)):
                    if np.isscalar(truetask[ind]):
                        truetask[ind] = truecurtask[ind]
                truetask = tuple(truetask)
            else:
                counts.append(0)
                switch_pts.append((cur_ids[-1], t))
                switch_x.append(sample.get_X(t=t))
                self.task_successes[self.task_list[curtask[0]]].append(1)
            cur_tasks.append(task)
            return truetask

        ntask = len(self.agent.task_list)
        rlen = ntask * self.agent.num_objs # if not self.agent.retime else (3*ntask) * self.agent.num_objs
        if self.agent.retime:
            rlen *= 4

        t_per_task = 20 # self.agent.plans[curtask].horizon 
        s_per_task = 3
        if self.agent.retime:
            s_per_task *= 3

        self.adj_eta = True
        l = self.get_task(x, targets, None, self.soft)
        l = tuple([val for val in l if np.isscalar(val)])
        cur_tasks.append(l)
        s, t = 0, 0
        val = 0
        state = x
        path = []
        last_switch = 0
        while val < 1 and s < rlen:
            #if self.check_precond and len(precond_viols): break
            
            curtask = tuple([val for val in cur_tasks[-1] if np.isscalar(val)])
            task_name = self.task_list[cur_tasks[-1][0]]
            pol = self.agent.policies[task_name]
            #hor = int(3 * self.agent.plans[curtask].horizon)
            #if self.agent.retime:
            #    hor *= 2
            hor = t_per_task

            sample = self.agent.sample_task(pol, 0, state, cur_tasks[-1], skip_opt=True, hor=hor, task_f=task_f, policies=self.agent.policies)
            path.append(sample)
            state = sample.get(STATE_ENUM, t=sample.T-1)
            s += 1
            cur_ids.append(s)
            val = 1 - self.agent.goal_f(0, sample.get_X(sample.T-1), targets)
            if counts[-1] >= s_per_task * t_per_task:
                print('TIMEOUT TERMINATING', counts[-1], curtask)
                break
            
            if len(expl_precond_viols):
                print('PRECOND TERMINATING', counts[-1], curtask, switch_pts[-1])
                break

        if len(path):
            val = 1 - self.agent.goal_f(0, path[-1].get_X(path[-1].T-1), targets)
        else:
            val = 0

        self.adj_eta = False
        self.postcond_info.append(val)
        for step in path: step.source_label = 'rollout'
        postcost = self.agent.postcond_cost(path[-1], cur_tasks[-1], path[-1].T-1, tol=tol, x0=switch_x[-1])
        self.postcond_costs[self.task_list[cur_tasks[-1][0]]].append(postcost)
        if postcost > 1e-4:
            self.task_successes[self.task_list[cur_tasks[-1][0]]].append(0)
        else:
            self.task_successes[self.task_list[cur_tasks[-1][0]]].append(1)

        self.log_path(path, -20)
        #return val, path, max(0, s-last_switch), 0
        train_pts = []
        fail_type = 'successful_rollout'
        if len(precond_viols):
            fail_type = 'rollout_precondition_failure'
            bad_pt = precond_viols[0]
            #train_pts.extend([tuple(pt) + (fail_type,) for pt in precond_viols])
            rand_ind = np.random.choice(range(len(precond_viols)))
            train_pts.append(tuple(precond_viols[rand_ind]) + (fail_type,))

        if self.check_postcond or val < 1-1e-4:
            fail_type = 'rollout_postcondition_failure'
            bad_pt = switch_pts[-1]
            train_pts.append(tuple(bad_pt) + (fail_type,))
            #self.failed_trajs.append((path[bad_pt[0]:], [], []))

        if self.check_random_switch:
            ind = np.random.choice(range(len(switch_pts)))
            train_pts.append(tuple(switch_pts[ind]) + ('rollout_random_switch',))

        if self.check_midcond:
            bad_pt = switch_pts[-1]
            curtask = tuple([val for val in cur_tasks[-1] if np.isscalar(val)])
            #self.save_video(path[bad_pt[0]:], 0, lab='odd_midfail', st=bad_pt[1])
            plan = self.agent.plans[curtask]
            st = bad_pt[1]
            x0 = path[bad_pt[0]].get(STATE_ENUM, st).copy()

            traj, steps, _, env_states = self.agent.reverse_retime(path[bad_pt[0]:], (0, plan.horizon-1), label=True, start_t=st)
            for t in range(len(traj)-1):
                t = min(t, plan.horizon-1)
                set_params_attrs(plan.params, self.agent.state_inds, traj[t], t)
            set_params_attrs(plan.params, self.agent.state_inds, path[bad_pt[0]].get_X(t=st), 0)
            self.agent.set_symbols(plan, cur_tasks[-1], targets=targets)
            failed_preds = plan.get_failed_preds(tol=tol, active_ts=(0, plan.horizon-1))
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

                #fail_t = min([p[2]+p[1].active_range[0] for p in failed_preds]) - 1
            else:
                fail_t = plan.horizon - 1

            #if not len(failed_preds):
            #    self.suc_trajs.append((path[bad_pt[0]:], [], []))

            fail_t = max(0, fail_t)
            fail_t = min(fail_t, plan.horizon-3)
            if len(failed_preds):
                fail_type = 'rollout_midcondition_failure'
                if fail_t < len(steps):
                    fail_s = bad_pt[0] + steps[fail_t]
                else:
                    fail_s = bad_pt[0]
                #self.failed_trajs.append((path[fail_s:], [], []))

                print('MID COND:', fail_s, fail_t, bad_pt, failed_preds, self.id)
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
                self.push_queue(new_node, self.motion_queue)

        if val >= 0.999:
            print('Success in rollout. Pre: {} Post: {} Mid: {} Random: {}'.format(self.check_precond, \
                                                                        self.check_postcond, \
                                                                        self.check_midcond, \
                                                                        self.check_random_switch))
            self.agent.add_task_paths([path])
            n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_rollout']
            with n_plans.get_lock():
                n_plans.value += 1
            n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_total']
            with n_plans.get_lock():
                n_plans.value += 1
        else:
            print('Failure in supervised rollout. {}'.format(train_pts))


        if np.random.uniform() < 1:#0.25:
            lab = '' if val > 0.999 else fail_type
            self.save_video(path, val, lab=fail_type)
        return val, path, train_pts, neg_samples #bad_pt[0], bad_pt[1], fail_type, neg_samples
 
        
    def plan_from_fail(self, augment=False, mode='start'):
        self.cur_step += 1
        val = 1.
        i = 0
        while val >= 1. and i < 10:
            self.agent.replace_cond(0)
            self.agent.reset(0)
            val, path = self.test_hl(eta=self.explore_eta)
            i += 1

        if val < 1:
            if mode == 'start':
                s, t = 0, 0
            elif mode == 'end':
                s, t = -1, -1
            elif mode == 'random':
                s = np.random.randint(len(path))
                t = np.random.randint(path[s].T)
            elif mode == 'collision':
                opts = []
                for cur_s, sample in enumerate(path):
                    if 1 in sample.col_ts:
                        s = cur_s
                        t = list(sample.col_ts).index(1) - 5
                        if t < 0:
                            if s == 0:
                                s, t = 0, 0
                            else:
                                s -= 1
                                t = path[s].T + t
                        opts.append((s,t))
                if len(opts):
                    ind = np.random.randint(len(opts))
                    s, t = opts[0]#opts[ind]
                else:
                    s = np.random.randint(len(path))
                    t = np.random.randint(path[s].T)
            elif mode == 'tail':
                wts = np.exp(np.arange(len(path)) / 5.)
                wts /= np.sum(wts)
                s = np.random.choice(list(range(len(path))), p=wts)
                t = np.random.randint(path[s].T)
            elif mode == 'task':
                breaks = find_task_breaks(path)
                cost_f = lambda x, task: self.agent.cost_f(x, task, condition=0, targets=path[0].targets, active_ts=(-1,-1))
                fail_pt = first_failed_state(cost_f, breaks, path)
                if fail_pt is None:
                    (s, t) = len(path)-1, path[-1].T-1
                else:
                    (s, t) = fail_pt
            elif mode == 'switch':
                cur_task = path[0].get(FACTOREDTASK_ENUM, t=0)
                s, t = 0, 0
                cur_s, cur_t = 0, 0
                delta = 2
                post_cost = 0
                for s, sample in enumerate(path):
                    for t in range(sample.T):
                        if post_cost > 0: break
                        if not self.agent.compare_tasks(cur_task, sample.get(FACTOREDTASK_ENUM, t=t)):
                            post_cost = min([self.agent.postcond_cost(sample, 
                                                                      task=tuple(cur_task.astype(int)),
                                                                      t=i) 
                                                for i in range(max(0,t-delta), min(sample.T-1,t+delta))])
                            cur_task = sample.get(FACTOREDTASK_ENUM, t=t)
                            if post_cost == 0:
                                cur_s, cur_t = s, t
                s, t = cur_s, cur_t
            else:
                raise NotImplementedError

            x0 = path[s].get_X(t=t) # self.agent.x0[0]
            targets = path[s].targets # self.agent.target_vecs[0]

            initial, goal = self.agent.get_hl_info(x0, targets)
            plan = self.agent.plans[tuple(path[s].get(FACTOREDTASK_ENUM, t=t))]
            prob = plan.prob
            domain = plan.domain
            abs_prob = self.agent.hl_solver.translate_problem(prob, goal=goal, initial=initial)
            set_params_attrs(plan.params, self.agent.state_inds, x0, 0)
            hlnode = HLSearchNode(abs_prob,
                                 domain,
                                 prob,
                                 priority=ROLL_PRIORITY,
                                 x0=x0,
                                 targets=targets,
                                 expansions=0,
                                 label='failed_rollout',
                                 nodetype='dagger')
            self.push_queue(hlnode, self.task_queue)


    def check_hl_statistics(self, xvar=None, thresh=0):
        from tabulate import tabulate
        inds = {
                    'success at end': 0,
                    'path length': 1,
                    'optimal rollout success': 9,
                    'time': 3,
                    'n data': 6,
                    'n plans': 10,
                    'subgoals anywhere': 11,
                    'subgoals closest dist': 12,
                    'collision': 8,
                    'any target': 13,
                    'smallest tol': 14,
                    'success anywhere': 7,
                 }
        data = np.array(self.hl_data)
        mean_data = np.mean(data, axis=0)[0]
        info = []
        headers = ['Statistic', 'Value']
        for key in inds:
            info.append((key, mean_data[inds[key]]))
        print(self._hyperparams['weight_dir'])
        print(tabulate(info))


    def test_hl(self, rlen=None, save=True, ckpt_ind=None,
                restore=False, debug=False, save_fail=False,
                save_video=False, eta=None):
        if ckpt_ind is not None:
            print(('Rolling out for index', ckpt_ind))

        self.set_policies()
        if restore:
            self.policy_opt.restore_ckpts(ckpt_ind)
        elif self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()

        self.agent.debug = False
        prim_opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        n_targs = list(range(len(prim_opts[OBJ_ENUM])))
        res = []
        ns = [self.config['num_targs']]
        if self.config['curric_thresh'] > 0:
            ns = list(range(1, self.config['num_targs']+1))
        n = np.random.choice(ns)
        s = []
        x0 = self.agent.x0[0]
        targets = self.agent.target_vecs[0].copy()
        for t in range(n, n_targs[-1]):
            obj_name = prim_opts[OBJ_ENUM][t]
            targets[self.agent.target_inds['{0}_end_target'.format(obj_name), 'value']] = x0[self.agent.state_inds[obj_name, 'pose']]
        if rlen is None:
            rlen = 2 + n * len(self.agent.task_list)
            if self.agent.retime:
                rlen *= 2
        self.agent.T = 40 # self.config['task_durations'][self.task_list[0]]
        val, path = self.test_run(x0, targets, rlen, hl=True, soft=self.config['soft_eval'], eta=eta, lab=-5)
        adj_val = val
        #if not self.adj_eta:
        #    self.adj_eta = True
        #    adj_val, adj_path = self.test_run(x0, targets, rlen, hl=True, soft=True, eta=eta, lab=-10)
        #    self.adj_eta = False
        true_disp = np.min(np.min([[self.agent.goal_f(0, step.get(STATE_ENUM, t), targets, cont=True) for t in range(step.T)] for step in path]))
        true_val = np.max(np.max([[1-self.agent.goal_f(0, step.get(STATE_ENUM, t), targets) for t in range(step.T)] for step in path]))
        smallest_tol = 2.
        for tol in range(1, 20):
            next_val = self.agent.goal_f(0, path[-1].get(STATE_ENUM, path[-1].T-1), path[-1].targets, tol=tol/10.)
            if next_val < 0.1:
                smallest_tol = tol/10.
                break
        subgoal_suc = 1-self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets)
        anygoal_suc = 1-self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets, anywhere=True)
        subgoal_dist = self.agent.goal_f(0, np.concatenate([s.get(STATE_ENUM) for s in path]), targets, cont=True)
        ncols = 1. if len(path) >1 and any([len(np.where(sample.col_ts > 0.99)[0]) > 3 for sample in path[:-1]]) else 0. # np.max([np.max(sample.col_ts) for sample in path])
        plan_suc_rate = np.nan if self.agent.n_plans_run == 0 else float(self.agent.n_plans_suc_run) / float(self.agent.n_plans_run)
        n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans'].value
        s.append((val,
                  len(path), \
                  true_disp, \
                  time.time()-self.start_t, \
                  self.config['num_objs'], \
                  n, \
                  self.policy_opt.buf_sizes['n_data'].value, \
                  true_val, \
                  ncols, \
                  plan_suc_rate, \
                  n_plans,
                  subgoal_suc,
                  subgoal_dist,
                  anygoal_suc,
                  smallest_tol,
                  n_plans/(time.time()-self.start_t)))
        if len(self.postcond_info):
            s[0] = s[0] + (np.mean(self.postcond_info[-5:]),)
        else:
            s[0] = s[0] + (0,)
        s[0] = s[0] + (adj_val,)
        if ckpt_ind is not None:
            s[0] = s[0] + (ckpt_ind,)
        res.append(s[0])
        if save:
            if all([s.opt_strength == 0 for s in path]): self.hl_data.append(res)
            if val > 1-1e-2:
                print('-----> SUCCESS! Rollout succeeded in test!', self.id)
            # if self.use_qfunc: self.log_td_error(path)
            np.save(self.hl_test_log.format('', 'rerun_' if ckpt_ind is not None else ''), np.array(self.hl_data))

        if val < 1:
            fail_pt = {'time': time.time() - self.start_t,
                        'no': self.config['num_objs'],
                        'nt': self.config['num_targs'],
                        'N': self.policy_opt.N,
                        'x0': list(x0),
                        'targets': list(targets),
                        'goal': list(path[0].get(ONEHOT_GOAL_ENUM, t=0))}
            with open(self.fail_data_file, 'a+') as f:
                f.write(str(fail_pt))
                f.write('\n')

        opt_path = None
        if self.render and save_video:
            print('Saving video...', val)
            self.save_video(path, val > 0, lab='_{0}'.format(n_plans))
            if opt_path is not None: self.save_video(opt_path, val > 0, lab='_mp')
            print('Saved video. Rollout success was: ', val > 0)
        self.last_hl_test = time.time()
        self.agent.debug = True
        #if not self.run_hl_test and self.explore_wt > 0:
        #    if val > 0.999:
        #        for s in path: s.source_label = 'rollout'
        #        self.agent.add_task_paths([path])
        # print('TESTED HL')
        return val, path


    def run(self):
        step = 0
        ff_iters = self._hyperparams['warmup_iters']
        self.agent.hl_pol = False
        while not self.stopped:
            step += 1
            if self._n_plans <= ff_iters:
                n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans']
                self._n_plans = n_plans.value

            if self.run_hl_test or time.time() - self.last_hl_test > 120:
                self.agent.replace_cond(0)
                self.agent.reset(0)
                n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans'].value
                save_video = self.id.find('test') >= 0 and n_plans > 10
                self.test_hl(save_video=save_video)

            if self.run_hl_test: continue
            if self._n_plans < ff_iters: continue

            self.set_policies()
            node = self.pop_queue(self.rollout_queue)
            #if node is None:
            #    new_node = self.spawn_problem()
            #    self.push_queue(new_node, self.rollout_queue)
            #else:
            #    self.send_rollout(node)
            if node is None:
                node = self.spawn_problem()
            self.send_rollout(node)

            if self.fail_plan:
                self.plan_from_fail(mode=self.fail_mode)

            for task in self.alg_map:
                data = self.agent.get_opt_samples(task, clear=True)
                if len(data) and self.ll_rollout_opt:
                    inv_cov = self.agent.get_inv_cov()
                    self.alg_map[task]._update_policy_no_cost(data, label='rollout')

            if self.hl_rollout_opt:
                self.run_hl_update(label='rollout')

            self.write_log()
        self.policy_opt.sess.close()


    def send_rollout(self, node):
        x0 = node.x0
        targets = node.targets
        #val, path, s, t, fail_type, neg_samples = self.rollout(x0, targets)
        val, path, train_pts, neg_samples = self.rollout(x0, targets)
        print('Rolled out success from server {0}: {1}'.format(self.id, val))
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
                self.push_queue(hlnode, self.task_queue)

        if len(neg_samples) and self.use_neg:
            self.update_negative_primitive(neg_samples)
       

    def test_run(self, state, targets, max_t=20, hl=False, soft=False, check_cost=True, eta=None, lab=0):
        def task_f(sample, t, curtask):
            return self.get_task(sample.get_X(t=t), sample.targets, curtask, soft)
        self.agent.reset_to_state(state)
        path = []
        val = 0
        nout = len(self.agent._hyperparams['prim_out_include'])
        l = None
        t = 0
        if eta is not None: self.eta = eta 
        old_eta = self.eta
        debug = np.random.uniform() < 0.1
        while t < max_t and val < 1-1e-2:
            l = self.get_task(state, targets, l, soft)
            if l is None: break
            task_name = self.task_list[l[0]]
            pol = self.agent.policies[task_name]
            s = self.agent.sample_task(pol, 0, state, l, noisy=False, task_f=task_f, skip_opt=True, hor=30)
            val = 1 - self.agent.goal_f(0, s.get_X(s.T-1), targets)
            t += 1
            state = s.end_state # s.get_X(s.T-1)
            path.append(s)
        self.eta = old_eta
        self.log_path(path, lab)
        return val, path

    def check_failed_likelihoods(self):
        cur_t = time.time()
        for buf in [self.suc_trajs, self.failed_trajs]:
            for traj, ts, lls in buf:
                cur_lls = []
                for sample in traj:
                    for t in range(sample.T):
                        task = tuple(sample.get(FACTOREDTASK_ENUM, t=t).astype(int))
                        task_name = self.task_list[task[0]]
                        pol = self.agent.policies[task_name]
                        mu = pol.act(sample.get_X(t=t), sample.get_obs(t=t), t)
                        act = sample.get(ACTION_ENUM, t=t)
                        cur_lls.append(np.sum((mu-act)**2))
                ts.append(cur_t)
                lls.append(np.mean(cur_lls))

    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                }

        wind = 10
        self.check_failed_likelihoods()
        info['dagger_success'] = np.mean(self.postcond_info[-wind:]) if len(self.postcond_info) else 0.
        info['failed_trajs'] = [(ts, ll) for _, ts, ll in self.failed_trajs]
        info['suc_trajs'] = [(ts, ll) for _, ts, ll in self.suc_trajs]
        for key in self.fail_types:
            info[key] = self.fail_types[key] / self.n_fails

        for key in self.postcond_costs:
            if len(self.postcond_costs[key]):
                info[key+'_costs'] = np.mean(self.postcond_costs[key][-wind:])

        for key in self.task_successes:
            if len(self.task_successes[key]):
                info['{0}_successes'.format(key)] = np.mean(self.task_successes[key][-wind:])
                #info['{0}_success_rate'.format(key)] =  info['{0}_calls'.format(key)] / info['{0}_successes'.format(key)]
            else:
                #info['{0}_calls'.format(key)] = 0. 
                info['{0}_successes'.format(key)] = 0. 
                #info['{0}_success_rate'.format(key)] = 0. 

        return info #self.log_infos


    def write_log(self):
        with open(self.dagger_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')

