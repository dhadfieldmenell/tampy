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

from core.internal_repr.plan import Plan
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList

from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.server import Server
from policy_hooks.search_node import *


EXPAND_LIMIT = 10

class TaskServer(Server):
    def __init__(self, hyperparams):
        os.nice(1)

        super(TaskServer, self).__init__(hyperparams)
        self.in_queue = self.task_queue
        self.out_queue = self.motion_queue
        self.prob_queue = []
        self.labelled_dir = self._hyperparams.get('reference_dir', None) + '/'
        self.max_labels = self._hyperparams.get('max_label', -1)


    def run(self):
        if self.labelled_dir is not None:
            self.load_labelled_state()
        while not self.stopped:
            self.find_task_plan()
            time.sleep(0.01)


    def load_labelled_state(self):
        probs = []
        if self.labelled_dir is None or not len(self.labelled_dir): return
        fnames = os.listdir(self.labelled_dir)
        for fname in fnames:
            if fname.find('npy') < 0: continue
            data = np.load(self.labelled_dir+fname, allow_pickle=True)
            for pt in data:
                label = pt[0]
                x, targets = pt[1], pt[2]
                inds = pt[3]
                suc = pt[4]
                ts = pt[-1]
                if label in ['after', 'during']:
                    probs.append((x, targets))
        if self.max_labels > 0 and len(probs) > self.max_labels:
            inds = np.random.choice(len(probs), self.max_labels)
            probs = [probs[i] for i in inds]
        ind = int(self.id[-1])
        ntask = self._hyperparams['num_task']
        nper = int(len(probs) / ntask)
        probs = probs[ind*nper:(ind+1)*nper]
        self.prob_queue.extend(probs)
        print('\n\n\n {} loaded {} labels\n\n'.format(self.id, len(probs)))


    def find_task_plan(self):
        node = self.pop_queue(self.task_queue)
        if node is None or node.expansions > EXPAND_LIMIT:
            if len(self.prob_queue):
                x, targets = self.prob_queue.pop()
                node = self.spawn_problem(x, targets)
                node.nodetype = 'human'
                node.label = 'offline human'
            else:
                node = self.spawn_problem()

        try:
            plan_str = self.agent.hl_solver.run_planner(node.abs_prob, node.domain, node.prefix, label='{}_{}'.format(self.id, self.exp_id))
        except OSError as e:
            print('OSError in hl solve:', e)
            plan_str = Plan.IMPOSSIBLE

        if plan_str == Plan.IMPOSSIBLE:
            n_plan = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}'.format(node.nodetype)]
            with n_plan.get_lock():
                n_plan.value += 1

            n_fail = self._hyperparams['policy_opt']['buffer_sizes']['n_plan_{}_failed'.format(node.nodetype)]
            with n_fail.get_lock():
                n_fail.value += 1

            with open(self.log_file, 'a+') as f:
                state_info = {(pname, aname): node.x0[self.agent.state_inds[pname, aname]] for (pname, aname) in self.agent.state_inds}
                info = '\n\n{} Task server could not plan for: {}\n{}\nExpansions: {}\n\n'.format(node.label, node.abs_prob, state_info, node.expansions)
                f.write(str(info))
            return

        new_node = LLSearchNode(plan_str, 
                                prob=node.concr_prob, 
                                domain=node.domain,
                                initial=node.concr_prob.initial,
                                priority=node.priority,
                                ref_plan=node.ref_plan,
                                targets=node.targets,
                                x0=node.x0,
                                expansions=node.expansions+1,
                                label=node.label,
                                refnode=node,
                                nodetype=node.nodetype,
                                info=node.info)

        if self.config['seq']:
            import pma.backtrack_ll_solver as bt_ll
            visual = len(os.environ.get('DISPLAY', '')) > 0
            if visual: self.agent.add_viewer()
            bt_ll.DEBUG = True
            plan = new_node.gen_plan(self.agent.hl_solver, self.agent.openrave_bodies, self.agent.ll_solver)
            success, opt_suc, path, info = self.agent.backtrack_solve(plan, anum=0, x0=node.x0, targets=node.targets, permute=False, label='seq')
            #if not success or not opt_suc:
            #    import ipdb; ipdb.set_trace()
            new_init = self.agent.hl_solver.apply_action(plan.prob.initial, plan.actions[0])
            import ipdb; ipdb.set_trace()
        self.push_queue(new_node, self.motion_queue)


