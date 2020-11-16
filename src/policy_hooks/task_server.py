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


class TaskServer(Server):
    def __init__(self, hyperparams):
        super(TaskServer, self).__init__(hyperparams)
        self.in_queue = self.task_queue
        self.out_queue = self.motion_queue


    def run(self):
        while not self.stopped:
            self.find_task_plan()
            time.sleep(0.01)

    def find_task_plan(self):
        node = self.pop_queue(self.task_queue)
        if node is None:
            return

        node.expansions += 1
        plan_str = self.agent.hl_solver.run_planner(node.abs_prob, node.domain, node.prefix, label=self.id)
        if plan_str == Plan.IMPOSSIBLE: return
        new_node = LLSearchNode(plan_str, 
                                prob=node.concr_prob, 
                                domain=node.domain,
                                initial=node.concr_prob.initial,
                                priority=node.priority,
                                ref_plan=node.ref_plan,
                                targets=node.targets,
                                x0 = node.x0,
                                expansions=node.expansions+1)
        self.push_queue(new_node, self.motion_queue)


