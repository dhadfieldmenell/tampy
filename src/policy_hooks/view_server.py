from datetime import datetime
import sys
from threading import Thread
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

from numba import cuda
import rospy
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


class DummyPolicy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call

    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class ViewServer(object):
    def __init__(self, hyperparams):
        rospy.init_node('view_server')
        self.agent = hyperparams['agent']
        if self.agent.viewer is None:
            self.agent.add_viewer()
        self.task_list = self.agent.task_list
        self.stopped = False
        self.weight_subscriber = rospy.Subscriber('tf_weights', String, self.store_weights, queue_size=1, buff_size=2**20)
        self.stop = rospy.Subscriber('terminate', String, self.end)

        self.policy_proxies = {task: rospy.ServiceProxy(task+'_policy_act', PolicyAct, persistent=True) for task in self.task_list}
        self.value_proxy = rospy.ServiceProxy('qvalue', QValue, persistent=True)
        self.primitive_proxy = rospy.ServiceProxy('primitive', Primitive, persistent=True)
        self.prob_proxies = {task: rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True) for task in self.task_list}
        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] + '_trained'
            hyperparams['policy_opt']['scope'] = None
            self.policy_opt = hyperparams['policy_opt']['type'](
                hyperparams['policy_opt'], 
                hyperparams['dO'],
                hyperparams['dU'],
                hyperparams['dObj'],
                hyperparams['dTarg'],
                hyperparams['dPrimObs']
            )
            self.rollout_policies = {task: self.policy_opt.task_map[task]['policy'] for task in self.task_list}

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']


    def gen_controls(self):
        plt.figure(figsize=(8,4))
        self.output_box = Rectangle((0,0), 1, 0.3, facecolor='black')
        self.full_axes = plt.axes([0, 0, 1, 1])
        self.full_axes.add_patch(self.output_box)
        self.output_label = plt.text(0, 0.25, 'Output:', color='white')
        self.output_text = plt.text(0, 0.2, '', color='white', wrap=True)
        self.axes1  = plt.axes([0.1, 0.5, 0.25, 0.05])
        self.cur_condition = 0
        self.cur_condition_text = self.full_axes.text(0.1, 0.4, 'Current condition: {0}'.format(self.cur_condition))
        self.b_run_cond = Button(self.axes1, "Run current condition.", color='Green')
        self.b_run_cond.on_clicked(self.run_condition)
        self.axes2  = plt.axes([0.2, 0.6, 0.075, 0.05])
        self.t_choose_cond = TextBox(self.axes2, 'Condition: ')
        self.t_choose_cond.on_text_change(self.update_condition)
        plt.show()


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received signal to terminate.')


    def store_weights(self, msg):
        if self.use_local:
            self.policy_opt.deserialize_weights(msg.data)


    def policy_call(self, x, obs, t, noise, task):
        # print 'Entering policy call:', datetime.now()
        if self.use_local:
            if self.policy_opt.task_map[task]['policy'].scale is None:
                return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
            return self.policy_opt.task_map[task]['policy'].act(x.copy(), obs.copy(), t, noise)

        rospy.wait_for_service(task+'_policy_act', timeout=10)
        req = PolicyActRequest()
        req.obs = obs
        req.noise = noise
        req.task = task
        resp = self.policy_proxies[task](req)
        # print 'Leaving policy call:', datetime.now()
        return np.array(resp.act)


    def value_call(self, obs):
        # print 'Entering value call:', datetime.now()
        if self.use_local:
            return self.policy_opt.value(obs)

        rospy.wait_for_service('qvalue', timeout=10)
        req = QValueRequest()
        req.obs = obs
        resp = self.value_proxy(req)
        # print 'Leaving value call:', datetime.now()
        return np.array(resp.value)


    def primitive_call(self, prim_obs):
        # print 'Entering primitive call:', datetime.now()
        if self.use_local:
            return self.policy_opt.task_distr(prim_obs)

        rospy.wait_for_service('primitive', timeout=10)
        req = PrimitiveRequest()
        req.prim_obs = prim_obs
        resp = self.primitive_proxy(req)
        # print 'Leaving primitive call:', datetime.now()
        return np.array(resp.task_distr), np.array(resp.obj_distr), np.array(resp.targ_distr)


    def prob(self, obs, task):
        # print 'Entering prob call:', datetime.now()
        if self.use_local:
            return self.policy_opt.prob(obs, task)

        rospy.wait_for_service(task+'_policy_prob', timeout=10)
        req_obs = []
        for i in range(len(obs)):
            next_line = Float32MultiArray()
            next_line.data = obs[i]
            req_obs.append(next_line)
        req = PolicyProbRequest()
        req.obs = req_obs
        req.task = task
        resp = self.prob_proxies[task](req)
        # print 'Leaving prob call:', datetime.now()
        return np.array([resp.mu[i].data for i in range(len(resp.mu))]), np.array([resp.sigma[i].data for i in range(len(resp.sigma))]), [], []


    def update_condition(self, txt):
        try:
            cond = int(txt)
            if cond in range(len(self.agent.x0)):
                self.cur_condition = cond
                self.cur_condition_text.set_text('Current condition: {0}'.format(self.cur_condition))
                self.output_text.set_text('')
            else:
                print 'Not a valid condition number.'
                self.output_text.set_text('Not a valid condition.')
        except ValueError:
            print 'Not a valid integer.'
            self.output_text.set_text('Not a valid integer.')
        plt.draw()


    def sample_current_policies(self, state, cond=0, task=None):
        if task is None:
            sample = Sample(self.agent)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TARGETS_ENUM, self.agent.target_vecs[cond].copy(), 0)
            obs = sample.get_prim_obs(t=0)
            task_distr, obj_distr, targ_distr = self.policy_opt.task_distr(obs)
            task_ind, obj_ind, targ_ind = np.argmax(task_distr), np.argmax(obj_distr), np.argmax(targ_distr)
        else:
            task_ind, obj_ind, targ_ind = task

        task = self.agent.task_list[task_ind]
        obj = self.agent.obj_list[obj_ind]
        targ = self.agent.targ_list[targ_ind]
        print 'Executing {0} on {1} to {2}'.format(task, obj, targ)

        policy = self.rollout_policies[task]
        sample = self.agent.sample_task(policy, cond, state, (task, obj, targ), noisy=False)
        return sample


    def run_condition(self, event):
        steps = 5 
        cond = self.cur_condition
        if cond not in range(len(self.agent.x0)):
            print 'Condition {0} does not exist.'.format(cond)
            return

        state = self.agent.x0[cond]
        for _ in range(steps):
            sample = self.sample_current_policies(state, cond)
            self.agent.animate_sample(sample)
            state = sample.end_state

 
    def run(self):
        ctrl_thread = Thread(target=self.gen_controls)
        ctrl_thread.daemon = True
        ctrl_thread.start()
        rospy.spin()
