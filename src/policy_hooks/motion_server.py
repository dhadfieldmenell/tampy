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
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.server import Server
from policy_hooks.search_node import *


class MotionServer(Server):
    def __init__(self, hyperparams):
        super(MotionServer, self).__init__(hyperparams)
        self.in_queue = self.motion_queue
        self.out_queue = self.task_queue


    def refine_plan(self):
        start_t = time.time()
        node = self.pop_queue(self.in_queue)
        size = self.in_queue.qsize()
        if node is None: return

        node.gen_plan(self.agent.hl_solver, self.agent.openrave_bodies)
        plan = node.curr_plan
        if type(plan) is str: return

        for a in range(plan.start+1):
            task = self.agent.encode_action(plan.actions[a])
            self.agent.set_symbols(plan, task, a, targets=node.targets)

        ts = (0, plan.actions[plan.start].active_timesteps[0])
        failed_prefix = plan.get_failed_preds(active_ts=ts, tol=1e-3)
        if len(failed_prefix):
            print('BAD PREFIX! -->', plan.actions[:plan.start], 'FAILED', failed_prefix, node.label)
            plan.start = 0

        plan.freeze_actions(plan.start)
        init_t = time.time()
        success = self.agent.backtrack_solve(plan, anum=plan.start, n_resamples=self._hyperparams['n_resample'], rollout=True)
        if success:
            path = self.agent.run_plan(plan, node.targets)
            for step in path: step.source_label = 'n_plans'
            print(self.id, 'Successful refine.', path[-1].success)
        if not success and node.gen_child():
            fail_step, fail_pred, fail_negated = node.get_failed_pred()
            print('Refine failed:', plan.get_failed_preds((0, fail_step)), fail_step, plan.actions)
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
                                 label=self.id)
            self.push_queue(hlnode, self.task_queue)
            print(self.id, 'Failed to refine, pushing to task node.')


    def run(self):
        step = 0
        while not self.stopped:
            node = self.spawn_problem()
            self.push_queue(node, self.task_queue)
            self.refine_plan()
            for task in self.alg_map:
                data = self.agent.get_opt_samples(task, clear=True)
                if len(data): self.alg_map[task]._update_policy_no_cost(data)
            self.run_hl_update()
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

