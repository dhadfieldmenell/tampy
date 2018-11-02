from abc import ABCMeta, abstractmethod
import copy
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

import rospy
from std_msgs.msg import *


from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.tamp_eval_funcs import *

from tamp_ros.msg import *
from tamp_ros.srv import *


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class AbstractMotionPlanServer(object):
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        self.id =  hyperparams['id']
        self.config = hyperparams
        rospy.init_node(hyperparams['domain']+'_mp_solver_'+str(self.id))
        self.task_list = hyperparams['task_list']
        plans = {}
        env = None
        openrave_bodies = {}
        prob = hyperparams['prob']
        for task in self.task_list:
            for c in range(hyperparams['num_objs']):
                plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)] = prob.get_plan_for_task(task, ['{0}{1}'.format(hyperparams['obj_type'], c), '{0}{1}_end_target'.format(hyperparams['obj_type'], c)], hyperparams['num_objs'], env, openrave_bodies)
                if env is None:
                    env = plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)].env
                    for param in plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body
        # self.policy_opt = hyperparams['policy_opt']
        # self.solver.policy_opt = self.policy_opt
        self.solver.policy_opt = DummyPolicyOpt(self.prob)
        self.agent = hyperparams['agent']
        self.solver.agent = self.agent
        self.agent.solver = self.solver
        self.weight_dir = hyperparams['weight_dir']
        self.solver.policy_inf_fs = {}
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            for j in range(len(self.agent.obj_list)):
                for k in range(len(self.agent.targ_list)):
                    self.solver.policy_inf_fs[(i,j,k)] = lambda o, s: self.prob(o, s, task)

        self.mp_service = rospy.Service('motion_planner_'+str(self.id), MotionPlan, self.serve_motion_plan)
        self.stopped = False
        self.mp_publishers = {i: rospy.Publisher('motion_plan_result_'+str(i), MotionPlanResult, queue_size=50) for i in range(hyperparams['n_rollout_servers'])}
        self.hl_publishers = {i: rospy.Publisher('hl_result_'+str(i), HLPlanResult, queue_size=50) for i in range(hyperparams['n_rollout_servers'])}
        self.async_planner = rospy.Subscriber('motion_plan_prob', MotionPlanProblem, self.publish_motion_plan, queue_size=10)
        self.async_hl_planner = rospy.Subscriber('hl_prob', HLProblem, self.publish_hl_plan, queue_size=10)
        self.weight_subscriber = rospy.Subscriber('tf_weights', String, self.store_weights, queue_size=1, buff_size=2**20)
        self.targets_subscriber = rospy.Subscriber('targets', String, self.update_targets)
        self.stop = rospy.Subscriber('terminate', String, self.end)

        self.prob_proxy = rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True)
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
                hyperparams['dPrimObs'],
                hyperparams['dValObs']
            )
        self.perturb_steps = hyperparams['perturb_steps']

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']
        self.n_time_samples_per_log = 10 if 'n_time_samples_per_log' not in hyperparams else hyperparams['n_time_samples_per_log']
        self.time_samples = []


    def run(self):
        rospy.spin()
        # while not self.stopped:
        #     rospy.sleep(0.01)


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def store_weights(self, msg):
        if self.use_local:
            self.policy_opt.deserialize_weights(msg.data)


    def update_targets(self, msg):
        raise NotImplementedError()


    def prob(self, obs, init_state, task):
        if self.use_local:
            mu, sig, prec, det_sig = self.policy_opt.traj_prob(obs, task)
            for p_name, a_name in self.solver.action_inds:
                mu[0, :, self.solver.action_inds[p_name, a_name]] += init_state[self.solver.state_inds[p_name, a_name]].reshape(-1,1)
            return mu, sig, prec, det_sig

        raise NotImplementedError()
        # rospy.wait_for_service(task+'_policy_prob', timeout=10)
        # req_obs = []
        # for i in range(len(obs)):
        #     next_line = Float32MultiArray()
        #     next_line.data = obs[i]
        #     req_obs.append(next_line)
        # req = PolicyProbRequest()
        # req.obs = req_obs
        # req.task = task
        # resp = self.prob_proxies[task](req)
        # return np.array([resp.mu[i].data for i in range(len(resp.mu))]), np.array([resp.sigma[i].data for i in range(len(resp.sigma))]), [], []


    def update_timing_info(self, time):
        if self.log_timing:
            self.time_samples.append(time)
            if len(self.time_samples) >= self.n_time_samples_per_log:
                with open(self.time_log, 'a+') as f:
                    f.write('Average time to motion plan for {0} problems: {1}\n\n'.format(len(self.time_samples), np.mean(self.time_samples)))
                self.time_samples = []


    def publish_motion_plan(self, msg):
        if msg.solver_id != self.id: return
        print 'Server {0} solving motion plan for rollout server {1}.'.format(self.id, msg.server_id)
        state = np.array(msg.state)
        task = msg.task
        task_tuple = (self.task_list.index(task), self.agent.obj_list.index(msg.obj), self.agent.targ_list.index(msg.targ))
        cond = msg.cond
        mean = np.array([msg.traj_mean[i].data for i in range(len(msg.traj_mean))])
        targets = [msg.obj, msg.targ]
        targets = [self.agent.plans.values()[0].params[p_name] for p_name in targets]
        sample, failed, success = self.agent.solve_sample_opt_traj(state, task, cond, mean, targets)
        failed = str(failed)
        resp = MotionPlanResult()
        resp.traj = []
        out = sample.get(STATE_ENUM)
        for t in range(len(out)):
            next_line = Float32MultiArray()
            next_line.data = out[t]
            resp.traj.append(next_line)
        resp.failed = failed
        resp.success = success
        resp.plan_id = msg.prob_id
        resp.cond = msg.cond
        resp.task = msg.task
        resp.obj = msg.obj
        resp.targ = msg.targ
        self.mp_publishers[msg.server_id].publish(resp)

        if success:
            for _ in range(self.perturb_steps):
                out, failed, success = self.agent.perturb_solve(sample)
                failed = str(failed)
                resp = MotionPlanResult()
                resp.traj = []
                out = sample.get(STATE_ENUM)
                for t in range(len(out)):
                    next_line = Float32MultiArray()
                    next_line.data = out[t]
                    resp.traj.append(next_line)
                resp.failed = failed
                resp.success = success
                resp.plan_id = msg.prob_id
                resp.cond = msg.cond
                resp.task = msg.task
                resp.obj = msg.obj
                resp.targ = msg.targ
                self.mp_publishers[msg.server_id].publish(resp)


    def publish_hl_plan(self, msg):
        if msg.solver_id != self.id: return
        paths = []
        failed = []
        new_failed = []
        stop = False
        attempt = 0
        cur_sample = None
        cond = msg.cond
        opt_hl_plan = []
        cur_path = []
        cur_state = np.array(msg.init_state)

        try:
            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
        except:
            hl_plan = []

        last_reset = 0
        while not stop and attempt < 4 * len(self.agent.obj_list):
            last_reset += 1
            for step in hl_plan:
                targets = [self.agent.plans.values()[0].params[p_name] for p_name in step[1]]
                plan = self.config['plan_f'](step[0], targets)
                if len(targets) < 2:
                    targets.append(plan.params['{0}_end_target'.format(targets[0].name)])
                next_sample, new_failed, success = self.agent.solve_sample_opt_traj(cur_state, step[0], cond, fixed_targets=targets)
                next_sample.success = FAIL_LABEL
                if not success:
                    if last_reset > 5:
                        failed = []
                        last_reset = 0
                    next_sample.success = FAIL_LABEL
                    if not len(new_failed):
                        stop = True
                    else:
                        failed.extend(new_failed)
                        try:
                            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
                        except:
                            hl_plan = []
                        # attempt += 1
                    break

                cur_path.append(next_sample)
                cur_sample = next_sample
                cur_state = cur_sample.end_state.copy()
                opt_hl_plan.append(step)

            if self.config['goal_f'](cur_state, self.agent.targets[cond], self.agent.plans.values()[0]) == 0:
                for sample in cur_path:
                    sample.success = SUCCESS_LABEL
                break

            attempt += 1

        resp = HLPlanResult()
        steps = []
        for sample in cur_path:
            mp_step = MotionPlanResult()

            mp_step.traj = []
            out = sample.get(STATE_ENUM)
            for t in range(len(out)):
                next_line = Float32MultiArray()
                next_line.data = out[t]
                mp_step.traj.append(next_line)
            mp_step.failed = ''
            mp_step.success = True
            mp_step.plan_id = -1
            mp_step.cond = msg.cond
            mp_step.task = sample.task
            mp_step.obj = sample.obj
            mp_step.targ = sample.targ

            steps.append(mp_step)
        resp.steps = steps
        resp.path_to = msg.path_to
        resp.success = len(cur_path) and cur_path[0].success == SUCCESS_LABEL
        resp.cond = msg.cond
        self.hl_publishers[msg.server_id].publish(resp)


    def serve_motion_plan(self, req):
        state = req.state
        task = req.task
        cond = req.condition
        mean = np.array([req.traj_mean[i].data for i in range(len(req.traj_mean))])
        targets = [reg.obj, req.targ]
        out, failed, success = self.sample_optimal_trajectory(state, task, cond, mean, targets)
        failed = str(failed)
        resp = MotionPlanResponse()
        resp.traj = out
        resp.failed = failed
        resp.success = success
        return resp


    def update_weight(self, msg):
        scope = msg.scope
        weight_dir = self.weight_dir
        variables = tf.get_colleciton(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(variables)
        saver.restore(self.policy_opt.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')


    @abstractmethod
    def sample_optimal_trajectory(self, state, task_tuple, condition, traj_mean=[], fixed_targets=[]):
        pass
