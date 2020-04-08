from abc import ABCMeta, abstractmethod
import copy
import random
import sys
import time
import traceback
from software_constants import *
import queue

import numpy as np
import tensorflow as tf

if USE_ROS:
    import rospy
    from std_msgs.msg import *
    from tamp_ros.msg import *
    from tamp_ros.srv import *

from gps.utility.gmm import GMM

from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.msg_classes import *

PLAN_COST_THRESHOLD = 1e-2


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class AbstractMotionPlanServer(object):
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        self.id =  hyperparams['id']
        self.task_name =  hyperparams['task_name']
        self.group_id = hyperparams['group_id']
        self.seed = int((1e2*time.time()) % 1000.)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.start_t = hyperparams['start_t']
        self.last_opt_t = time.time()

        self.config = hyperparams
        if USE_ROS: rospy.init_node(hyperparams['domain']+'_mp_solver_{0}_{1}'.format(self.id, self.group_id))
        self.task_list = hyperparams['task_list']
        self.on_policy = hyperparams['on_policy']
        self.problem = hyperparams['prob']
        plans, openrave_bodies, env = self.problem.get_plans()
        
        # self.policy_opt = hyperparams['policy_opt']
        # self.solver.policy_opt = self.policy_opt
        self.solver.policy_opt = DummyPolicyOpt(self.problem)
        self.solver.policy_priors = {task: GMM() for task in self.task_list}
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.solver.agent = self.agent
        self.agent.solver = self.solver
        self.weight_dir = hyperparams['weight_dir']
        if USE_ROS: self.test_publisher = rospy.Publisher('is_alive', String, queue_size=2)
        if not USE_ROS:
            self.queues = hyperparams['queues']

        # self.mp_service = rospy.Service('motion_planner_'+str(self.id), MotionPlan, self.serve_motion_plan)
        self.stopped = False
        self.busy = False
        # self.mp_publishers = {i: rospy.Publisher('motion_plan_result_'+str(i), MotionPlanResult, queue_size=5) for i in range(hyperparams['n_rollout_servers'])}
        # self.hl_publishers = {i: rospy.Publisher('hl_result_'+str(i), HLPlanResult, queue_size=5) for i in range(hyperparams['n_rollout_servers'])}
        self.mp_publishers = {}
        self.hl_publishers = {}
        if USE_ROS:
            # self.policy_prior_subscriber = rospy.Subscriber('policy_prior', PolicyPriorUpdate, self.update_policy_prior)
            self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)
            self.opt_count_publisher = rospy.Publisher('optimization_counter', String, queue_size=1)

            # self.prob_proxy = rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True)
        
        self.use_local = False
        '''
        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] + '_trained'
            hyperparams['policy_opt']['scope'] = None
            hyperparams['policy_opt']['gpu_fraction'] = 1./32
            hyperparams['policy_opt']['use_gpu'] = 1.
            hyperparams['policy_opt']['allow_growth'] = True
            self.policy_opt = hyperparams['policy_opt']['type'](
                hyperparams['policy_opt'], 
                hyperparams['dO'],
                hyperparams['dU'],
                hyperparams['dPrimObs'],
                hyperparams['dValObs'],
                hyperparams['prim_bounds']
            )
        '''
        self.perturb_steps = hyperparams['perturb_steps']

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.traj_init_log = 'tf_saved/'+hyperparams['weight_dir']+'/traj_init_log.txt'
        self.log_timing = hyperparams['log_timing']
        self.n_time_samples_per_log = 10 if 'n_time_samples_per_log' not in hyperparams else hyperparams['n_time_samples_per_log']
        self.time_samples = []
        # self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)
        self.mp_plans_since_log = {task: 0 for task in self.task_list}
        self.hl_plans_since_log = 0
        self.total_traj_init_since_log = {task: 0 for task in self.task_list}
        self.total_planning_time_since_log = {task: 0 for task in self.task_list}
        self.n_successful_trajs_since_log = {task: 0 for task in self.task_list}
        self.failed_per_task = {}

        self.mp_queue = []
        if USE_ROS:
            self.async_publisher = rospy.Publisher('motion_plan_result_{0}'.format(self.group_id), MotionPlanResult, queue_size=1)
            self.async_planner = rospy.Subscriber('motion_plan_prob_{0}'.format(self.group_id), MotionPlanProblem, self.publish_motion_plan, queue_size=1, buff_size=2**19)
            # self.async_hl_planner = rospy.Subscriber('hl_prob_{0}'.format(self.group_id), HLProblem, self.publish_hl_plan, queue_size=2, buff_size=2**20)
            self.weight_subscriber = rospy.Subscriber('tf_weights_{0}'.format(self.group_id), UpdateTF, self.store_weights, queue_size=1, buff_size=2**22)
            # self.targets_subscriber = rospy.Subscriber('targets', String, self.update_targets, queue_size=1)
            rospy.sleep(1)


    def parse_queue(self):
        q = self.queues['optimizer{0}'.format(self.id)]
        if q.empty(): return
        i = 0
        while i < q._maxsize and not q.empty():
            try:
                prob = q.get_nowait()
                self.publish_motion_plan(prob)
            except queue.Empty:
                break


    def run(self):
        # rospy.spin()
        i = 0
        while not self.stopped:
            if not USE_ROS: self.parse_queue()
            if len(self.mp_queue):
                self.solve_motion_plan(self.mp_queue.pop())
            i += 1
            td = time.time() - self.last_opt_t
            if time.time() - self.start_t > self.config['time_limit'] or (self.id > 5 and td > 300 and time.time() - self.start_t > 900): #TODO: variables for when to kill process
                self.stopped = True
                break
            # if not i % 10:
            #     self.test_publisher.publish('MP {0} alive.'.format(self.id))

        # self.policy_opt.sess.close()

    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received notice to terminate.')


    def publish_log_info(self):
        info = {}
        info['avg_init_cost'] = {task: self.total_traj_init_since_log[task] / float(self.mp_plans_since_log[task]) for task in self.task_list if self.mp_plans_since_log[task] > 0}
        info['avg_plan_time'] = {task: self.total_planning_time_since_log[task] / float(self.mp_plans_since_log[task]) for task in self.task_list if self.mp_plans_since_log[task] > 0}
        info['failed_preds'] = str({self.failed_per_task})
        info['successes'] = str(self.n_successful_trajs_since_log)
        info['num_mp'] = self.mp_plans_since_log
        self.mp_plans_since_log = {task: 0 for task in self.task_list}
        self.total_planning_time_since_log = {task: 0 for task in self.task_list}
        self.total_traj_init_since_log = {task: 0 for task in self.task_list}
        self.n_successful_trajs_since_log = {task: 0 for task in self.task_list}
        self.failed_per_task = {}
        # self.log_publisher.publish(str({'motion_plan': info}))


    def store_weights(self, msg):
        if self.use_local:
            save = self.id == 0
            self.policy_opt.deserialize_weights(msg.data, save=save)


    def publish_mp(self, msg, server_id):
        self.async_publisher.publish(msg)
        '''
        if server_id not in self.mp_publishers:
            self.mp_publishers[server_id] = rospy.Publisher('motion_plan_result_'+str(server_id), MotionPlanResult, queue_size=5)
            time.sleep(1)
        self.mp_publishers[server_id].publish(msg)
        '''


    def publish_hl(self, msg, server_id):
        if server_id not in self.hl_publishers:
            self.hl_publishers[server_id] = rospy.Publisher('hl_result_'+str(server_id), HLPlanResultResult, queue_size=5)
        self.hl_publishers[server_id].publish(msg)


    # def update_policy_prior(self, msg):
    #     task = msg.task
    #     sigma = np.array(msg.sigma).reshape(msg.K, msg.Do, msg.Do)
    #     mu = np.array(msg.mu).reshape(msg.K, msg.Do)
    #     logmass = np.array(msg.logmass).reshape(msg.K, 1)
    #     mass = np.array(msg.mass).reshape(msg.K, 1)
    #     self.solver.policy_prior[task].sigma = sigma
    #     self.solver.policy_prior[task].mu = mu
    #     self.solver.policy_prior[task].logmass = logmass
    #     self.solver.policy_prior[task].mass = mass
    #     self.solver.policy_prior[task].N = msg.N


    def gen_gmm(self, msg):
        gmm = GMM()
        sigma = np.array(msg.sigma).reshape((msg.K, msg.Do, msg.Do))
        mu = np.array(msg.mu).reshape((msg.K, msg.Do))
        logmass = np.array(msg.logmass).reshape((msg.K, 1))
        mass = np.array(msg.mass).reshape((msg.K, 1))
        gmm.sigma = sigma
        gmm.mu = mu
        gmm.logmass = logmass
        gmm.mass = mass
        gmm.N = msg.N
        return gmm


    def gen_all_gmm(self, msg):
        gmms = {}
        info = eval(msg.gmms)
        for task in info:
            gmms[task] = GMM()
            N = info[task]['N']
            K = info[task]['K']
            Do = info[task]['Do']
            sigma = np.array(info[task]['sigma']).reshape((K, Do, Do))
            mu = np.array(info[task]['mu']).reshape((K, Do))
            logmass = np.array(info[task]['logmass']).reshape((K, 1))
            mass = np.array(info[task]['mass']).reshape((K, 1))
            gmm.sigma = sigma
            gmm.mu = mu
            gmm.logmass = logmass
            gmm.mass = mass
            gmm.N = N
        return gmm


    def update_targets(self, msg):
        raise NotImplementedError()


    def prob(self, sample):
        mu, sig, _, _ = self.policy_opt.prob(sample.get_obs(), task=sample.task)
        return mu[0], sig[0]


    def update_timing_info(self, time):
        if self.log_timing:
            self.time_samples.append(time)
            if len(self.time_samples) >= self.n_time_samples_per_log:
                with open(self.time_log, 'a+') as f:
                    f.write('Average time to motion plan for {0} problems: {1}\n\n'.format(len(self.time_samples), np.mean(self.time_samples)))
                self.time_samples = []


    def publish_motion_plan(self, msg, check=True):
        task_name = self.task_list[eval(msg.task)[0]]
        if check and msg.solver_id != self.id: return
        self.mp_queue.append(msg)
        while len(self.mp_queue) > 10:
            self.mp_queue.pop(0)


    def solve_motion_plan(self, msg):
        # print(self.id, 'got message for', msg.solver_id)
        # if self.busy:
        #     print 'Server', self.id, 'busy, rejecting request for motion plan.'
        #     return
        if msg.solver_id != self.id: return
        # self.busy = True
        print('Server {0} solving motion plan on task{1} for rollout server {2}{3}.'.format(self.id, msg.task, msg.server_id, ' using prior' if msg.use_prior else ' no prior'))
        state = np.array(msg.state)
        targets = np.array(msg.targets)
        task = eval(msg.task)
        task_tuple = eval(msg.task)
        task_name = self.task_list[task[0]]
        cond = msg.cond
        if USE_ROS:
            mean = np.array([msg.traj_mean[i].data for i in range(len(msg.traj_mean))])
        else:
            mean = np.array(msg.traj_mean)

        if msg.use_prior:
            T, dU, dX = msg.T, msg.dU, msg.dX
            K = np.array(msg.pol_K).reshape((T, dU, dX))
            k = np.array(msg.pol_k).reshape((T, dU))
            chol_pol_S = np.array(msg.chol).reshape((T, dU, dU))
            # gmm = self.gen_gmm(msg)
            # inf_f = lambda s: self.gmm_inf(gmm, s)
            inf_f = (K, k, chol_pol_S)
        else:
            K, k, chol_pol_S = None, None, None
            inf_f = None

        plan = self.agent.plans[task]

        '''
        init_cost, failed_preds = self.log_init_traj_cost(mean, task)
        self.total_traj_init_since_log[task_name] += init_cost
        failed_info = self.failed_per_task.get(task_name, [])
        failed_info.append(([str(pred) for pred in failed_preds], state.tolist()))
        self.failed_per_task[task_name] = failed_info
        '''

        start_t = time.clock()
        if True or (np.isnan(init_cost) or init_cost > PLAN_COST_THRESHOLD and len(failed_preds) > 0):
            mean = []
            start_t = time.time()
            sample, failed, success = self.agent.solve_sample_opt_traj(state, task, cond, mean, inf_f, targets=targets, x_only=True)
            # print('Time in full solve:', time.time() - start_t)
            out = sample.get(STATE_ENUM)
        else:
            out = mean
            failed =  []
            success = True
            self.n_successful_trajs_since_log[task_name] += 1

        self.total_planning_time_since_log[task_name] += time.clock() - start_t
        self.mp_plans_since_log[task_name] += 1

        failed = str(failed)
        resp = MotionPlanResult() if USE_ROS else DummyMSG
        if USE_ROS:
            resp.traj = []
            for t in range(len(out)):
                next_line = Float32MultiArray()
                next_line.data = out[t]
                resp.traj.append(next_line)
        else:
            resp = out

        resp.failed = failed
        resp.success = success
        resp.plan_id = msg.prob_id
        resp.cond = msg.cond
        resp.task = msg.task
        resp.state = state.tolist()
        resp.server_id = msg.server_id
        resp.alg_id = msg.alg_id
        resp.targets = targets
        if USE_ROS:
            # self.mp_publishers[msg.server_id].publish(resp)
            self.publish_mp(resp, msg.server_id)
        else:
            alg_q = self.queues['alg_opt_rec{0}'.format(msg.algorithm_id)]
            rol_q = self.queues['rollout_opt_rec{0}'.format(msg.rollout_id)]
            for q in [alg_q, rol_q]:
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    q.put_nowait(resp)
                except queue.Full:
                    pass

        if success:
            print('Succeeded:', success, failed)
            print('\n')

        if False and success and self.agent.prob.USE_PERTURB:
            for _ in range(self.perturb_steps):
                out, failed, success = self.agent.perturb_solve(sample, inf_f=inf_f)
                if success:
                    failed = str(failed)
                    resp = MotionPlanResult()
                    resp.traj = []
                    state = out.get_X(t=0)
                    out_traj = out.get(STATE_ENUM)
                    for t in range(len(out_traj)):
                        next_line = Float32MultiArray()
                        next_line.data = out_traj[t]
                        resp.traj.append(next_line)
                    resp.failed = failed
                    resp.targets = targets
                    resp.success = success
                    resp.plan_id = 'no_id'
                    resp.cond = msg.cond
                    resp.task = msg.task
                    resp.state = state.tolist()
                    # self.mp_publishers[msg.server_id].publish(resp)
                    resp.server_id = msg.server_id
                    resp.alg_id = msg.alg_id
                    self.publish_mp(resp, msg.server_id)
        self.busy = False
        print('Server {0} free.'.format(self.id, msg.server_id))
        self.last_opt_t = time.time()


    def publish_hl_plan(self, msg):
        if self.busy:
            print('Server', self.id, 'busy, rejecting request for motion plan.')
        if msg.solver_id != self.id or self.busy: return
        self.busy = True
        self.hl_plans_since_log += 1
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

        gmms = self.gen_all_gmm(msg)
        inf_fs = {}
        for task in gmms:
            inf_f[task] = lambda s: self.gmm_inf(gmms[task], s)
        for task in self.task_list:
            if task not in gmms:
                inf_f[task] = None

        try:
            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
        except:
            hl_plan = []

        last_reset = 0
        while not stop and attempt < 4 * len(self.agent.obj_list):
            last_reset += 1
            for step in hl_plan:
                task = []
                task.append(self.task_list.index(step[0]))
                prim_options = self.problem.get_prim_choices().values()
                for i in range(len(step[1])):
                    # First value of prim_options is the task list
                    task.append(prim_options[i+1].index(step[1][i]))
                task = tuple(task)
                plan = self.agent.plans[task]
                next_sample, new_failed, success = self.agent.solve_sample_opt_traj(cur_state, task, cond, inf_f=inf_f[step[0]])
                self.optimization_counter.publish("Solved motion plan.")
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
                cur_state = cur_sample.end_state # get_X(t=cur_sample.T-1)
                opt_hl_plan.append(step)

            if self.config['goal_f'](cur_state, self.agent.targets[cond], self.agent.plans_list()[0]) == 0:
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
            mp_step.task = str(sample.task)
            mp_step.state = sample.get_X(t=0).tolist()
            steps.append(mp_step)

        resp.steps = steps
        resp.path_to = msg.path_to
        resp.success = len(cur_path) and cur_path[0].success == SUCCESS_LABEL
        resp.cond = msg.cond
        # self.hl_publishers[msg.server_id].publish(resp)
        self.publish_hl(resp, msg.server_id)
        self.busy = False


    def check_traj_cost(self, traj, task):
        if np.any(np.isnan(np.array(traj))):
           raise Exception('Nans in trajectory passed')
        plan = self.agent.plans[task]
        old_free_attrs = plan.get_free_attrs()
        for t in range(0, len(traj)):
            for param_name, attr in plan.state_inds:
                param = plan.params[param_name]
                if param.is_symbol(): continue
                if hasattr(param, attr):
                    getattr(param, attr)[:, t] = traj[t, plan.state_inds[param_name, attr]]
                    param.fix_attr(attr, (t,t))
                    if attr == 'pose' and (param_name, 'rotation') not in plan.state_inds and hasattr(param, 'rotation'):
                        param.rotation[:, t] = 0
                    
                    if attr == 'pose':
                        init_target = '{0}_init_target'.format(param_name)
                        if init_target in plan.params:
                            plan.params[init_target].value[:, 0] = param.pose[:, 0]

        # Take a quick solve to resolve undetermined symbolic values
        self.solver._backtrack_solve(plan, time_limit=10, max_priority=-1, task=task)
        plan.store_free_attrs(old_free_attrs)
        
        plan_total_violation = np.sum(plan.check_total_cnt_violation(active_ts=(1,plan.horizon-1)))
        plan_failed_constrs = plan.get_failed_preds_by_type()
        return plan_total_violation, plan_failed_constrs


    def log_init_traj_cost(self, mean, task):
        plan = self.agent.plans[task]
        '''
        for t in range(0, len(mean)):
            for param_name, attr in plan.state_inds:
                param = plan.params[param_name]
                if param.is_symbol(): continue
                if hasattr(param, attr):
                    getattr(param, attr)[:, t] = mean[t, plan.state_inds[param_name, attr]]
        '''
        plan_actions = str(plan.actions)
        plan_total_violation, plan_failed_constrs = self.check_traj_cost(mean, task)
        with open(self.traj_init_log, 'a+') as f:
            f.write(str((plan_actions, plan_total_violation, plan_failed_constrs)))
            f.write('\n\n\n')
        return plan_total_violation, plan_failed_constrs
        

    def update_weight(self, msg):
        if not self.use_local(): return
        scope = msg.scope
        weight_dir = self.weight_dir
        variables = tf.get_colleciton(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(variables)
        saver.restore(self.policy_opt.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')


    # def gmm_inf(self, gmm, sample):
    #     mu, sig = gmm.inference(np.concatenate([sample.get(STATE_ENUM)[:, :-1], sample.get(STATE_ENUM)[:,1:]]))
    #     return mu, sig, True, False
   
    def gmm_inf(self, gmm, sample):
        dux = self.agent.symbolic_bound + self.agent.dU
        mu, sig = np.zeros((sample.T, dux)), np.zeros((sample.T, dux, dux))
        for t in range(sample.T):
            mu0, sig0, _, _ = gmm.inference(np.concatenate([sample.get(STATE_ENUM, t=t), sample.get_U(t=t)], axis=-1).reshape((1, -1)))
            mu[t] = mu0
            sig[t] = sig0
        return mu, sig, False, True

