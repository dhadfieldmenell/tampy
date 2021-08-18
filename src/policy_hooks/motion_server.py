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

from core.internal_repr.plan import Plan
import core.util_classes.transform_utils as T
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.server import Server
from policy_hooks.search_node import *


LOG_DIR = 'experiment_logs/'

class MotionServer(Server):
    def __init__(self, hyperparams):
        super(MotionServer, self).__init__(hyperparams)
        self.in_queue = self.motion_queue
        self.out_queue = self.task_queue
        self.label_type = 'optimal'
        self.opt_wt = hyperparams['opt_wt']
        self.motion_log = LOG_DIR + hyperparams['weight_dir'] + '/MotionInfo_{0}_log.txt'.format(self.id)
        self.log_infos = []
        self.infos = {'n_ff': 0, 'n_postcond': 0, 'n_precond': 0, 'n_midcond': 0, 'n_explore': 0, 'n_plans': 0}
        self.avgs = {key: [] for key in self.infos}
        self.fail_infos = {'n_fail_ff': 0, 'n_fail_postcond': 0, 'n_fail_precond': 0, 'n_fail_midcond': 0, 'n_fail_explore': 0, 'n_fail_plans': 0}
        self.fail_avgs = {key: [] for key in self.fail_infos}
        self.fail_rollout_infos = {'n_fail_rollout_ff': 0, 'n_fail_rollout_postcond': 0, 'n_fail_rollout_precond': 0, 'n_fail_rollout_midcond': 0, 'n_fail_rollout_explore': 0}
        self.init_costs = []
        self.rolled_costs = []
        self.final_costs = []
        self.plan_times = []
        self.plan_horizons = []
        self.opt_rollout_info = {'{}_opt_rollout_success'.format(taskname): [] for taskname in self.task_list}
        with open(self.motion_log, 'w+') as f:
            f.write('')


    def gen_plan(self, node):
        node.gen_plan(self.agent.hl_solver, self.agent.openrave_bodies, self.agent.ll_solver)
        plan = node.curr_plan
        if type(plan) is str: return plan
        if not len(plan.actions):
            return plan

        for a in range(min(len(plan.actions), plan.start+1)):
            task = self.agent.encode_action(plan.actions[a])
            self.agent.set_symbols(plan, task, a, targets=node.targets)
        
        plan.start = min(plan.start, len(plan.actions)-1)
        ts = (0, plan.actions[plan.start].active_timesteps[0])
        try:
            failed_prefix = plan.get_failed_preds(active_ts=ts, tol=1e-3)
        except Exception as e:
            failed_prefix = ['ERROR IN FAIL CHECK', e]

        if len(failed_prefix) and node.hl:
            #print('BAD PREFIX! -->', plan.actions[:plan.start], 'FAILED', failed_prefix, node._trace)
            plan.start = 0

        ts = (0, plan.actions[plan.start].active_timesteps[0])
        if node.freeze_ts <= 0:
            set_params_attrs(plan.params, self.agent.state_inds, node.x0, ts[1])
        plan.freeze_actions(plan.start)
        cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0
        return plan


    def refine_plan(self, node):
        start_t = time.time()
        if node is None: return

        plan = self.gen_plan(node)
        if type(plan) is str or not len(plan.actions): return

        cur_t = node.freeze_ts if node.freeze_ts >= 0 else 0
        cur_step = 2
        self.n_plans += 1

        while cur_t >= 0:
            path, success, opt_suc = self.collect_trajectory(plan, node, cur_t)
            self.log_node_info(node, success, path)
            prev_t = cur_t
            cur_t -= cur_step
            if success and len(path) and path[-1].success: continue

            if not opt_suc: self.parse_failed(plan, node, prev_t)
            while len(plan.get_failed_preds((cur_t, cur_t))) and cur_t > 0:
                cur_t -= 1

            node.freeze_ts = cur_t
            plan = self.gen_plan(node)


    def collect_trajectory(self, plan, node, cur_t):
        x0 = None
        if cur_t < len(node.ref_traj): x0 = node.ref_traj[cur_t]
        if cur_t == 0: x0 = node.x0

        wt = self.explore_wt if node.label.lower().find('rollout') >= 0 or node.nodetype.find('dagger') >= 0 else 1.
        verbose = self.verbose and (self.id.find('r0') >= 0 or np.random.uniform() < 0.05)
        self.agent.store_hist_info(node.info)
        
        init_t = time.time()
        success, opt_suc, path, info = self.agent.backtrack_solve(plan, 
                                                                 anum=plan.start, 
                                                                 x0=x0,
                                                                 targets=node.targets,
                                                                 n_resamples=self._hyperparams['n_resample'], 
                                                                 rollout=self.rollout_opt, 
                                                                 traj=node.ref_traj, 
                                                                 st=cur_t, 
                                                                 permute=self.permute_hl,
                                                                 label=node.nodetype,
                                                                 backup=self.backup,
                                                                 verbose=verbose,
                                                                 hist_info=node.info)
        end_t = time.time()
        for step in path:
            step.wt = wt

        if success:
            self.plan_horizons.append(plan.horizon)
            self.plan_horizons = self.plan_horizons[-5:]
            self.plan_times.append(end_t-init_t)
            self.plan_times = self.plan_times[-5:]

        self.n_failed += 0. if success else 1.
        n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_plans']
        with n_plans.get_lock():
            n_plans.value += 1

        if self.verbose and len(path):
            if node.nodetype.find('dagger') >= 0 and np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_dgr'.format(success))
            elif np.random.uniform() < 0.05:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt'.format(success), annotate=True)
            elif not success and np.random.uniform() < 0.5:
                self.save_video(path, path[-1]._postsuc, lab='_suc_{}_opt_fail'.format(success), annotate=True)

        if self.verbose and self.render:
            for ind, batch in enumerate(info['to_render']):
                for next_path in batch:
                    if len(next_path):
                        print('BACKUP VIDEO:', next_path[-1].task)
                        self.save_video(next_path, next_path[-1]._postsuc, lab='_{}_backup_solve'.format(ind))

        self.log_path(path, 10)
        for step in path: step.source_label = node.nodetype
        if success and len(path):
            print(self.id, 'succ. refine', node.label, plan.actions[0].name, 'rollout succ:', path[-1]._postsuc, path[-1].success, 'goal:', self.agent.goal(0, path[-1].targets))

        if len(path) and path[-1].success:
            n_plans = self._hyperparams['policy_opt']['buffer_sizes']['n_total']
            with n_plans.get_lock():
                n_plans.value += 1

        n_plan = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}'.format(node.nodetype)]
        with n_plan.get_lock():
            n_plan.value += 1

        if not success:
            print('Opt failure from', node.label, node.nodetype)
            n_fail = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}_failed'.format(node.nodetype)]
            with n_fail.get_lock():
                n_fail.value += 1

        if np.random.uniform() < 0.05 and success:
            self.send_to_label(path, success)
        return path, success, opt_suc


    def parse_failed(self, plan, node, prev_t):
        try:
            fail_step, fail_pred, fail_negated = node.get_failed_pred(st=prev_t)
        except:
            fail_pred = None

        if fail_pred is None:
            print('Failure without failed constr?')
            return

        failed_preds = plan.get_failed_preds((prev_t, fail_step+fail_pred.active_range[1]), priority=-1)
        if len(failed_preds):
            print('Refine failed with linear constr. viol.', node._trace, plan.actions, failed_preds, len(node.ref_traj), node.label)
            return

        print('Refine failed:', plan.get_failed_preds((0, fail_step+fail_pred.active_range[1])), fail_pred, fail_step, plan.actions, node.label, node._trace, prev_t)
        if not node.hl: return
        if not node.gen_child(): return
        n_problem = node.get_problem(fail_step, fail_pred, fail_negated)
        abs_prob = self.agent.hl_solver.translate_problem(n_problem, goal=node.concr_prob.goal)
        prefix = node.curr_plan.prefix(fail_step)
        hlnode = HLSearchNode(abs_prob,
                             node.domain,
                             n_problem,
                             priority=node.priority+1,
                             prefix=prefix,
                             llnode=node,
                             x0=node.x0,
                             targets=node.targets,
                             expansions=node.expansions+1,
                             label=self.id,
                             nodetype=node.nodetype,
                             info=node.info)
        self.push_queue(hlnode, self.task_queue)
        print(self.id, 'Failed to refine, pushing to task node.')


    def run(self):
        step = 0
        while not self.stopped:
            node = self.pop_queue(self.in_queue)
            if node is None:
                time.sleep(0.01)
                continue

            self.set_policies()
            self.write_log()
            self.refine_plan(node)

            inv_cov = self.agent.get_inv_cov()
            for task in self.alg_map:
                data = self.agent.get_opt_samples(task, clear=True)
                opt_samples = [sample for sample in data if not len(sample.source_label) or sample.source_label.find('opt') >= 0]
                expl_samples = [sample for sample in data if len(sample.source_label) and sample.source_label.find('opt') < 0]
                if len(opt_samples):
                    self.alg_map[task]._update_policy_no_cost(opt_samples, label='optimal', inv_cov=inv_cov)
                if len(expl_samples):
                    self.alg_map[task]._update_policy_no_cost(expl_samples, label='dagger', inv_cov=inv_cov)

            self.run_hl_update()

            cont_samples = self.agent.get_cont_samples()
            if len(cont_samples):
                self.update_cont_network(cont_samples)

            step += 1

        self.policy_opt.sess.close()


    def update_expert_demos(self, demos):
        for path in demos:
            for key in self.expert_demos:
                self.expert_demos[key].append([])
            for s in path:
                for t in range(s.T):
                    if not s.use_ts[t]: continue
                    self.expert_demos['acs'][-1].append(s.get(ACTION_ENUM, t=t))
                    self.expert_demos['obs'][-1].append(s.get_prim_obs(t=t))
                    self.expert_demos['ep_rets'][-1].append(1)
                    self.expert_demos['rews'][-1].append(1)
                    self.expert_demos['tasks'][-1].append(s.get(FACTOREDTASK_ENUM, t=t))
                    self.expert_demos['use_mask'][-1].append(s.use_ts[t])
        if self.cur_step % 5:
            np.save(self.expert_data_file, self.expert_demos)

    def log_node_info(self, node, success, path):
        key = 'n_ff'
        if node.label.find('post') >= 0:
            key = 'n_postcond'
        elif node.label.find('pre') >= 0:
            key = 'n_precond'
        elif node.label.find('mid') >= 0:
            key = 'n_midcond'
        elif node.label.find('rollout') >= 0:
            key = 'n_explore'

        self.infos[key] += 1
        self.infos['n_plans'] += 1
        for altkey in self.avgs:
            if altkey != key:
                self.avgs[altkey].append(0)
            else:
                self.avgs[altkey].append(1)

        failkey = key.replace('n_', 'n_fail_')
        if not success:
            self.fail_infos[failkey] += 1
            self.fail_infos['n_fail_plans'] += 1
            self.fail_avgs[failkey].append(0)
        else:
            self.fail_avgs[failkey].append(1)

        with self.policy_opt.buf_sizes[key].get_lock():
            self.policy_opt.buf_sizes[key].value += 1


    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                'optimization time': np.mean(self.plan_times),
                'plan length': np.mean(self.plan_horizons),
                'opt duration per ts': np.mean(self.plan_times) / np.mean(self.plan_horizons),
                }

        for key in self.infos:
            info[key] = self.infos[key]

        for key in self.fail_infos:
            info[key] = self.fail_infos[key]

        for key in self.fail_rollout_infos:
            info[key] = self.fail_rollout_infos[key]

        wind = 10
        for key in self.avgs:
            if len(self.avgs[key]):
                info[key+'_avg'] = np.mean(self.avgs[key][-wind:])

        for key in self.fail_avgs:
            if len(self.fail_avgs[key]):
                info[key+'_avg'] = np.mean(self.fail_avgs[key][-wind:])

        for key in self.opt_rollout_info:
            if len(self.opt_rollout_info[key]):
                info[key] = np.mean(self.opt_rollout_info[key][-wind:])

        if len(self.init_costs): info['mp initial costs'] = np.mean(self.init_costs[-10:])
        if len(self.rolled_costs): info['mp rolled out costs'] = np.mean(self.rolled_costs[-10:])
        if len(self.final_costs): info['mp optimized costs'] = np.mean(self.final_costs[-10:])
        return info #self.log_infos


    def write_log(self):
        with open(self.motion_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')

