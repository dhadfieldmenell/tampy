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
        self.labelled_dir = self._hyperparams.get('labelled_dir', None)


    def run(self):
        while not self.stopped:
            self.find_task_plan()
            time.sleep(0.01)


    def load_labelled_state(self):
        probs = []
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
        ind = int(self.id[-1])
        ntask = self._hyperparams['num_task']
        nper = len(probs) / ntask
        probs = probs[ind*nper:(ind+1)*nper]
        self.prob_queue.extend(probs)


    def find_task_plan(self):
        node = self.pop_queue(self.task_queue)
        if node is None or node.expansions > EXPAND_LIMIT:
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
                info = '\n\n{} Task server could not plan for: {}\n{}\n\n'.format(node.label, node.abs_prob, state_info)
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
                                nodetype=node.nodetype)
        self.push_queue(new_node, self.motion_queue)


