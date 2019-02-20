from datetime import datetime
import numpy as np
import os
import random
import sys
import time

from numba import cuda
import rospy
from scipy.cluster.vq import kmeans2 as kmeans
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.utils.policy_solver_utils import *


class DummyPolicy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call

    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


class RolloutServer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        np.random.seed(self.id*1234)
        rospy.init_node('rollout_server_'+str(self.id))
        self.mcts = hyperparams['mcts']
        self.prim_dims = hyperparams['prim_dims']
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        for m in self.mcts:
            m.value_func = self.value_call
            m.prob_func = self.primitive_call
            m.agent = self.agent
        self.alg_map = hyperparams['alg_map']
        self.task_list = self.agent.task_list
        self.pol_list = tuple(hyperparams['policy_list']) + ('value', 'primitive')
        self.traj_opt_steps = hyperparams['traj_opt_steps']
        self.num_samples = hyperparams['num_samples']
        for mcts in self.mcts:
            mcts.num_samples = self.num_samples
        self.num_rollouts = hyperparams['num_rollouts']
        self.stopped = False

        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=2) for task in self.pol_list}
        self.updaters['value'] = rospy.Publisher('value_update', PolicyUpdate, queue_size=5)
        self.updaters['primitive'] = rospy.Publisher('primitive_update', PolicyUpdate, queue_size=5)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob', MotionPlanProblem, queue_size=1)
        self.hl_publisher = rospy.Publisher('hl_prob', HLProblem, queue_size=2)
        self.test_publisher = rospy.Publisher('is_alive', String, queue_size=2)

        for alg in self.alg_map.values():
            alg.policy_opt = DummyPolicyOpt(self.update, self.prob)
        self.n_optimizers = hyperparams['n_optimizers']
        self.waiting_for_opt = {}
        self.sample_queue = []
        self.current_id = 0
        self.cur_step = 0
        self.opt_samples = {task: [] for task in self.task_list}
        self.max_sample_queue = hyperparams['max_sample_queue']
        self.max_opt_sample_queue = hyperparams['max_opt_sample_queue']
        self.early_stop_prob = hyperparams['mcts_early_stop_prob']
        self.run_hl_prob = hyperparams['run_hl_prob'] if 'run_hl_prob' in hyperparams else 0
        self.opt_prob = hyperparams['opt_prob'] if 'opt_prob' in hyperparams else 0.05

        # self.policy_proxies = {task: rospy.ServiceProxy(task+'_policy_act', PolicyAct, persistent=True) for task in self.task_list}
        # self.value_proxy = rospy.ServiceProxy('qvalue', QValue, persistent=True)
        # self.primitive_proxy = rospy.ServiceProxy('primitive', Primitive, persistent=True)
        # self.prob_proxies = {task: rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True) for task in self.task_list}
        # self.mp_proxies = {mp_id: rospy.ServiceProxy('motion_planner_'+str(mp_id), MotionPlan, persistent=True) for mp_id in range(self.n_optimizers)}

        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] + '_trained'
            hyperparams['policy_opt']['scope'] = None
            hyperparams['policy_opt']['gpu_fraction'] = 1./16
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
            for alg in self.alg_map.values():
                alg.local_policy_opt = self.policy_opt

        self.traj_centers = hyperparams['n_traj_centers']
        self.opt_queue = []
        self.hl_opt_queue = []

        self.rollout_log = 'tf_saved/'+hyperparams['weight_dir']+'/rollout_log_{0}.txt'.format(self.id)
        state_info = []
        params = self.agent.plans.values()[0].params
        for x in self.agent.x0:
            info = []
            for param_name, attr in self.agent.state_inds:
                if params[param_name].is_symbol(): continue
                value = x[self.agent._x_data_idx[STATE_ENUM]][self.agent.state_inds[param_name, attr]]
                info.append((param_name, attr, value))
            state_info.append(info)
        with open(self.rollout_log, 'w+') as f:
            for i in range(len(state_info)):
                f.write(str(i)+': '+str(state_info)+'\n')
            f.write('\n\n\n')

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']

        self.mp_subcriber = rospy.Subscriber('motion_plan_result_'+str(self.id), MotionPlanResult, self.sample_mp, queue_size=3, buff_size=2**19)
        self.hl_subscriber = rospy.Subscriber('hl_result_'+str(self.id), HLPlanResult, self.update_hl, queue_size=1)
        self.weight_subscriber = rospy.Subscriber('tf_weights', String, self.store_weights, queue_size=1, buff_size=2**22)
        self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)



    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def renew_publisher(self):
        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=2) for task in self.pol_list}
        self.updaters['value'] = rospy.Publisher('value_update', PolicyUpdate, queue_size=5)
        self.updaters['primitive'] = rospy.Publisher('primitive_update', PolicyUpdate, queue_size=5)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob', MotionPlanProblem, queue_size=1)
        self.hl_publisher = rospy.Publisher('hl_prob', HLProblem, queue_size=2)
        self.test_publisher = rospy.Publisher('is_alive', String, queue_size=2)


    def update(self, obs, mu, prc, wt, task, rollout_len=0):
        msg = PolicyUpdate()
        msg.obs = obs.flatten()
        msg.mu = mu.flatten()
        msg.prc = prc.flatten()
        msg.wt = wt.flatten()
        msg.dO = self.agent.dO
        msg.dPrimObs = self.agent.dPrim
        msg.dValObs = self.agent.dVal
        msg.dU = mu.shape[-1]
        msg.n = len(mu)
        msg.rollout_len = mu.shape[1] if rollout_len < 1 else rollout_len
        if task in self.pol_list:
            print 'Sent update to', task, 'policy'
            self.updaters[task].publish(msg)
        else:
            # Assume that if we don't have a policy for this task we're using a single control policy
            print 'Sent update to control policy'
            self.updaters['control'].publish(msg)


    def store_weights(self, msg):
        if self.use_local:
            self.policy_opt.deserialize_weights(msg.data)


    def policy_call(self, x, obs, t, noise, task):
        # print 'Entering policy call:', datetime.now()
        if self.use_local:
            if 'control' in self.policy_opt.task_map:
                if self.policy_opt.task_map['control']['policy'].scale is None:
                    return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
                return self.policy_opt.task_map['control']['policy'].act(x.copy(), obs.copy(), t, noise)
            else:
                if self.policy_opt.task_map[task]['policy'].scale is None:
                    return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
                return self.policy_opt.task_map[task]['policy'].act(x.copy(), obs.copy(), t, noise)
        raise NotImplementedError
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
        raise NotImplementedError
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
        raise NotImplementedError
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
        raise NotImplementedError
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


    def update_hl(self, msg):
        cond = msg.cond
        mcts = self.mcts[cond]
        path_to = eval(msg.path_to)
        samples = []
        opt_queue = []
        for step in msg.steps:
            plan_id = -1 # step.plan_id
            traj = np.array([step.traj[i].data for i in range(len(step.traj))])
            state = np.array(step.state)
            success = step.success
            task = eval(step.task)
            path_to.append((task))
            condition = step.cond
            # opt_sample = self.agent.sample_optimal_trajectory(traj[0], task, cond, traj, traj_mean=[])
            # samples.append(opt_sample)
            # self.store_opt_sample(opt_sample, -1)
            if success:
                waiters = []
                if plan_id in self.waiting_for_opt:
                    waiters = self.waiting_for_opt[plan_id]
                    del self.waiting_for_opt[plan_id]

                opt_queue.append((plan_id, state, task, condition, traj, waiters))

        self.hl_opt_queue.append((path_to, msg.success, opt_queue))
        # mcts.update_vals(path_to, msg.success)
        # self.update_qvalue(samples)
        # self.update_primitive(samples)
        if msg.success:
            self.early_stop_prob *= 0.975
        else:
            self.early_stop_prob *= 1.025


    def motion_plan(self, x, task, condition, traj_mean, targets):
        mp_id = np.random.randint(0, self.n_optimizers)
        mean = []
        for i in range(len(mean)):
            next_line = Float32MultiArray()
            next_line.data = traj_mean[i].tolist()
            mean.append(next_line)

        req = MotionPlanRequest()
        req.state = x.flatten()
        req.task = task
        req.obj = targets[0].name
        req.targ = targets[1].name
        req.condition = condition
        req.mean = mean

        resp = self.mp_proxies[mp_id](req)
        failed = eval(resp.failed)
        success = resp.success
        traj = np.array([resp.traj[i].data for i in range(len(resp.traj))])


    def store_for_opt(self, samples):
        self.waiting_for_opt[self.current_id] = samples
        self.sample_queue.append(self.current_id)
        self.current_id += 1
        while len(self.sample_queue) > self.max_sample_queue:
            if self.sample_queue[0] in self.waiting_for_opt:
                del self.waiting_for_opt[self.sample_queue[0]]
            del self.sample_queue[0]


    def store_opt_sample(self, sample, plan_id, waiters=[]):
        samples = waiters
        if plan_id in self.waiting_for_opt:
            if len(self.waiting_for_opt[plan_id]):
                samples = self.waiting_for_opt[plan_id]
            del self.waiting_for_opt[plan_id]

        for s in samples:
            s.set_ref_X(sample.get_ref_X())

        task = self.task_list[sample.task[0]]
        self.opt_samples[task].append((sample, samples))
        while len(self.opt_samples[task]) > self.max_opt_sample_queue:
            del self.opt_samples[task][0]


    def sample_mp(self, msg):
        plan_id = msg.plan_id
        traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        state = np.array(msg.state)
        success = msg.success
        task = eval(msg.task)
        condition = msg.cond
        print "Received trajectory, success:", success
        if success:
            waiters = []
            if plan_id in self.waiting_for_opt:
                waiters = self.waiting_for_opt[plan_id]
                # del self.waiting_for_opt[plan_id]

            self.opt_queue.append((plan_id, state, task, condition, traj, waiters))
            # print 'Sampling optimal trajectory on rollout server {0}.'.format(self.id)
            # opt_sample = self.agent.sample_optimal_trajectory(state, task, condition, opt_traj=traj, traj_mean=[])
            # self.store_opt_sample(opt_sample, plan_id)


    def choose_mp_problems(self, samples):
        Xs = samples.get_X()[:,:,self.agent._x_data_idx[STATE_ENUM]]
        if self.traj_centers <= 1:
            return [[np.mean(Xs, axis=0), samples]]
        flat_Xs = Xs.reshape((Xs.shape[0], np.prod(Xs.shape[1:])))
        centroids, labels = kmeans(flat_Xs, k=self.traj_centers, minit='points')
        probs = []
        for c in range(len(centroids)):
            centroid = centroids[c]
            traj_mean = centroid.reshape(Xs.shape[1:])
            probs.append([traj_mean, []])

        for i in range(len(samples)):
            probs[labels[i]][1].append(samples[i])

        probs = filter(lambda p: len(p[1]), probs)

        for p in probs:
            p[1] = SampleList(p[1])


        return probs


    def send_mp_problem(self, centroid, s_list):
        next_sample = s_list[0]
        state = next_sample.get_X(t=0)
        task = next_sample.task
        cond = next_sample.condition
        traj_mean = []
        for t in range(next_sample.T):
            next_line = Float32MultiArray()
            next_line.data = centroid[t]
            traj_mean.append(next_line)
        prob = MotionPlanProblem()
        prob.state = state
        prob.task = str(task)
        prob.cond = cond
        prob.traj_mean = traj_mean
        prob.prob_id = self.current_id
        prob.solver_id = np.random.randint(0, self.n_optimizers)
        prob.server_id = self.id
        self.store_for_opt(s_list)

        if self.alg_map[next_sample.task_name].policy_prior.gmm.sigma is None:
            prob.use_prior = False
        else:
            gmm = self.alg_map[next_sample.task_name].mp_policy_prior.gmm
            prob.use_prior = True
            prob.mu = gmm.mu.flatten()
            prob.sigma = gmm.sigma.flatten()
            prob.logmass = gmm.logmass.flatten()
            prob.mass = gmm.mass.flatten()
            prob.N = len(gmm.mu)
            prob.K = len(gmm.mass)
            prob.Do = gmm.sigma.shape[1]

        print '\n\nSending motion plan problem to server {0}.\n\n'.format(prob.solver_id)
        self.async_plan_publisher.publish(prob)
        self.test_publisher.publish('MCTS sent motion plan.')


    def run_opt_queue(self):
        if len(self.opt_queue):
            print 'Running optimal trajectory.'
            plan_id, state, task, condition, traj, waiters = self.opt_queue.pop()
            opt_sample = self.agent.sample_optimal_trajectory(state, task, condition, opt_traj=traj, traj_mean=[])
            if not len(waiters):
                print "\n\n\n\n\n\n\nNO WAITERS ON PLAN"
            else:
                print "\n\n\n\n\n\n\nFOUND WAITERS FOR PLAN"
            self.store_opt_sample(opt_sample, plan_id, waiters)


    def run_hl_opt_queue(self, mcts):
        if len(self.hl_opt_queue):
            samples = []
            path_to, success, opt_queue = self.hl_opt_queue.pop()
            mcts.update_vals(path_to, success)
            for step in opt_queue:
                plan_id, state, task, condition, traj, waiters = step
                opt_sample = self.agent.sample_optimal_trajectory(state, task, cond, traj, traj_mean=[])
                samples.append(opt_sample)
                self.store_opt_sample(opt_sample, -1)
            self.update_qvalue(samples)
            self.update_primitive(samples)


    def step(self):
        print '\n\nTaking tree search step.\n\n'
        self.cur_step += 1
        rollout_policies = {task: DummyPolicy(task, self.policy_call) for task in self.agent.task_list}

        start_time = time.time()
        all_samples = []
        if 'rollout_server_'+str(self.id) not in os.popen("rosnode list").read():
            print "\n\nRestarting dead ros node: rollout server\n\n", self.id
            rospy.init_node('rollout_server_'+str(self.id))
        
        self.renew_publisher()
        random.shuffle(self.mcts) # If rospy hangs, don't want it to always be for the same trees
        for mcts in self.mcts:
            print os.popen("rosnode list").read(), "\n", self.id, "\n\n"
            val = mcts.run(self.agent.x0[mcts.condition], 1, use_distilled=False, new_policies=rollout_policies, debug=False)
            self.run_opt_queue()
            self.run_hl_opt_queue(mcts)
            self.test_publisher.publish('MCTS Step')
            if np.random.uniform() < self.run_hl_prob and val > 0:
                init_state = self.agent.x0[mcts.condition]
                prob = HLProblem()
                prob.server_id = self.id
                prob.solver_id = np.random.randint(0, self.n_optimizers)
                prob.init_state = init_state.tolist()
                prob.cond = mcts.condition
                gmms = {}
                for task in self.task_list:
                    gmm = self.alg_map[task].mp_policy_prior.gmm
                    if gmm.sigma is None: continue
                    gmms[task] = {}
                    gmms[task]['mu'] = gmm.mu.flatten()
                    gmms[task]['sigma'] = gmm.sigma.flatten()
                    gmms[task]['logmass'] = gmm.logmass.flatten()
                    gmms[task]['mass'] = gmm.mass.flatten()
                    gmms[task]['N'] = len(gmm.mu)
                    gmms[task]['K'] = len(gmm.mass)
                    gmms[task]['Do'] = gmm.sigma.shape[1]
                prob.gmms = str(gmms)

                path = mcts.simulate(init_state, early_stop_prob=self.early_stop_prob)
                path_tuples = []
                for step in path:
                    path_tuples.append(step.task)
                prob.path_to = str(path_tuples)
                self.hl_publisher.publish(prob)

            sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}
            self.agent.clear_samples(keep_prob=0.0, keep_opt_prob=0.0)
            n_probs = 0

            if 'rollout_server_'+str(self.id) not in os.popen("rosnode list").read():
                print "\n\nRestarting dead ros node:", 'rollout_server_'+str(self.id), '\n\n'
                rospy.init_node('rollout_server_'+str(self.id))
            else:
                print "Rosnode alive", self.id

            for task in sample_lists:
                for s_list in sample_lists[task]:
                    mp_prob = np.random.uniform()
                    # print 'Checking for sample list for', task
                    # print mp_prob < self.opt_prob, mp_prob, self.opt_prob
                    if mp_prob < self.opt_prob:
                        # print 'Choosing mp problems.'
                        all_samples.extend(s_list._samples)
                        probs = self.choose_mp_problems(s_list)
                        n_probs += len(probs)
                        for p in probs:
                            self.send_mp_problem(*p)
            for task in self.agent.task_list:
                if len(self.opt_samples[task]) > 0:
                    sample_lists[task] = self.alg_map[task].iteration(self.opt_samples[task], reset=False)
            self.renew_publisher()
        end_time = time.time()


        start_time_2 = time.time()
        end_time_2 = time.time()

        # if self.log_timing:
        #     with open(self.time_log, 'a+') as f:
        #         f.write('Generated {0} problems from {1} conditions with {2} rollouts per condition.\n'.format(n_probs, len(self.mcts), self.num_rollouts))
        #         f.write('Time to complete: {0}\n'.format(end_time-start_time))
        #         f.write('Time to select problems through kmeans and send to supervised learner: {0}\n\n'.format(end_time_2-start_time_2))

        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)

        self.update_primitive(path_samples)
        self.update_qvalue(all_samples)

        start_time = time.time()
        for task in self.agent.task_list:
            if len(self.opt_samples[task]):
                sample_lists[task] = self.alg_map[task].iteration(self.opt_samples[task], reset=False)

        # for step in range(self.traj_opt_steps-1):
        #     for task in self.agent.task_list:
        #         try:
        #             sample_lists[task] = self.alg_map[task].iteration(sample_lists[task], self.opt_samples[task], reset=True)
        #             if len(sample_lists[task]):
        #                 sample_lists[task] = self.agent.resample(sample_lists[task], rollout_policies[task], self.n_samples)
        #             else:
        #                 continue
        #             self.agent._samples[task] = sample_lists[task]
        #         except:
        #             traceback.print_exception(*sys.exc_info())
        # self.agent.reset_sample_refs()
        end_time = time.time()
        if self.log_timing:
            with open(self.time_log, 'a+') as f:
                f.write('Time to update algorithms for {0} iterations on data: {1}\n\n'.format(self.traj_opt_steps, end_time-start_time))

        print '\n\nFinished tree search step.\n\n'


    def run(self):
        while not self.stopped:
            self.step()
            rospy.sleep(0.1)


    def update_qvalue(self, samples, first_ts_only=False):
        dV, dO = 2, self.agent.dVal

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            for t in range(sample.T):
                obs = [sample.get_val_obs(t=t)]
                mu = [sample.success]
                prc = [np.eye(dV)]
                wt = 1. # [10. / (t+1)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))
                if first_ts_only: break

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'value', 1)


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        for sample in samples:
            for t in range(sample.T):
                obs = [sample.get_prim_obs(t=t)]
                mu = [np.concatenate([sample.get(enum, t=t) for enum in self.prim_dims])]
                prc = [np.eye(dP)]
                wt = [1.] # [np.exp(-sample.task_cost)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 1)
