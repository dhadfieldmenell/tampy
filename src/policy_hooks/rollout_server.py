from datetime import datetime
import numpy as np
import os
import pprint
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
        U = self.policy_call(x, obs, t, noise, self.task)
        if np.any(np.isnan(x)):
            raise Exception('Nans in policy call state.')
        if np.any(np.isnan(obs)):
            raise Exception('Nans in policy call obs.')
        if np.any(np.isnan(noise)):
            raise Exception('Nans in policy call noise.')
        if np.any(np.isnan(U)):
            raise Exception('Nans in policy call action.')
        return U


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


class RolloutServer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        self.num_conds = hyperparams['num_conds']
        np.random.seed(int(time.time()/100000))
        rospy.init_node('rollout_server_'+str(self.id))
        self.mcts = hyperparams['mcts']
        self.run_mcts_rollouts = hyperparams.get('run_mcts_rollouts', True)
        self.run_alg_updates = hyperparams.get('run_alg_updates', True)
        self.prim_dims = hyperparams['prim_dims']
        self.solver = hyperparams['solver_type'](hyperparams)
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.solver = self.solver
        self.solver.agent = self.agent
        for m in self.mcts:
            m.value_func = self.value_call
            m.prob_func = self.primitive_call
            m.agent = self.agent
            # m.log_file = 'tf_saved/'+hyperparams['weight_dir']+'/mcts_log_{0}_cond{1}.txt'.format(self.id, m.condition)
            # with open(m.log_file, 'w+') as f: f.write('')
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
        self.n_opt_calls = 0
        self.rollout_opt_pairs = {task: [] for task in self.task_list}
        self.max_sample_queue = int(hyperparams['max_sample_queue'])
        self.max_opt_sample_queue = int(hyperparams['max_opt_sample_queue'])
        self.early_stop_prob = hyperparams['mcts_early_stop_prob']
        self.run_hl_prob = hyperparams['run_hl_prob'] if 'run_hl_prob' in hyperparams else 0
        self.opt_prob = hyperparams['opt_prob'] if 'opt_prob' in hyperparams else 0.05

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
            self.weights_to_store = {}

        self.traj_centers = hyperparams['n_traj_centers']
        self.opt_queue = []
        self.hl_opt_queue = []

        self.rollout_log = 'tf_saved/'+hyperparams['weight_dir']+'/rollout_log_{0}.txt'.format(self.id)
        state_info = []
        params = self.agent.plans.values()[0].params
        # for x in self.agent.x0:
        #     info = []
        #     for param_name, attr in self.agent.state_inds:
        #         if params[param_name].is_symbol(): continue
        #         value = x[self.agent._x_data_idx[STATE_ENUM]][self.agent.state_inds[param_name, attr]]
        #         info.append((param_name, attr, value))
        #     state_info.append(info)
        # with open(self.rollout_log, 'w+') as f:
        #     for i in range(len(state_info)):
        #         f.write(str(i)+': '+str(state_info)+'\n')
        #     f.write('\n\n\n')

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']

        self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)

        self.mp_subcriber = rospy.Subscriber('motion_plan_result_'+str(self.id), MotionPlanResult, self.sample_mp, queue_size=3, buff_size=2**19)
        self.hl_subscriber = rospy.Subscriber('hl_result_'+str(self.id), HLPlanResult, self.update_hl, queue_size=1)
        self.weight_subscriber = rospy.Subscriber('tf_weights', UpdateTF, self.store_weights, queue_size=1, buff_size=2**22)
        self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)
        self.node_ref = {}


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
        assert(len(mu) == len(obs))

        prc[np.where(prc > 1e10)] = 1e10
        wt[np.where(wt > 1e10)] = 1e10
        prc[np.where(prc < -1e10)] = -1e10
        wt[np.where(wt < -1e10)] = -1e10
        mu[np.where(np.abs(mu) > 1e10)] = 0
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        obs[np.where(np.abs(obs) > 1e10)] = 0
        msg = PolicyUpdate()
        msg.obs = obs.flatten().tolist()
        msg.mu = mu.flatten().tolist()
        msg.prc = prc.flatten().tolist()
        msg.wt = wt.flatten().tolist()
        msg.dO = self.agent.dO
        msg.dPrimObs = self.agent.dPrim
        msg.dValObs = self.agent.dVal
        msg.dU = mu.shape[-1]
        msg.n = len(mu)
        msg.rollout_len = mu.shape[1] if rollout_len < 1 else rollout_len
        # if task != 'value':
        #     print('Sending update on', task)

        if task in self.updaters:
            # print 'Sent update to', task, 'policy'
            self.updaters[task].publish(msg)
        else:
            # Assume that if we don't have a policy for this task we're using a single control policy
            # print 'Sent update to control policy'
            self.updaters['control'].publish(msg)


    def store_weights(self, msg):
        self.weights_to_store[msg.scope] = msg.data


    def update_weights(self):
        if self.use_local:
            scopes = self.weights_to_store.keys()
            for scope in scopes:
                save = self.id.endswith('0')
                data = self.weights_to_store[scope]
                self.weights_to_store[scope] = None
                if data is not None:
                    self.policy_opt.deserialize_weights(data, save=save)


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


    # def store_for_opt(self, samples):
    #     self.waiting_for_opt[self.current_id] = samples
    #     self.sample_queue.append(self.current_id)
    #     self.current_id += 1
    #     while len(self.sample_queue) > self.max_sample_queue:
    #         if self.sample_queue[0] in self.waiting_for_opt:
    #             del self.waiting_for_opt[self.sample_queue[0]]
    #         del self.sample_queue[0]


    def store_opt_sample(self, opt_sample, plan_id, samples=[]):
        if plan_id in self.waiting_for_opt:
            if not len(samples) and len(self.waiting_for_opt[plan_id]):
                samples = self.waiting_for_opt[plan_id]
            del self.waiting_for_opt[plan_id]

        assert not np.all(np.abs(opt_sample.get_U() < 1e-4))
        for s in samples:
            s.set_ref_X(opt_sample.get_ref_X())
            s.set_ref_U(opt_sample.get_ref_U())
            assert not np.all(np.abs(s.get_U() < 1e-4))

        task_name = self.task_list[opt_sample.task[0]]
        self.rollout_opt_pairs[task_name].append((opt_sample, samples))
        self.rollout_opt_pairs[task_name] = self.rollout_opt_pairs[task_name][-self.max_opt_sample_queue:]


    def sample_mp(self, msg):
        ### If this server isn't running algorithm updates, it doesn't need to store the mp results
        # print('Received motion plan', self.id)
        if not self.run_alg_updates:
            return

        plan_id = msg.plan_id
        traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        state = np.array(msg.state)
        success = msg.success
        task = eval(msg.task)
        condition = msg.cond
        node = self.node_ref.get(plan_id, None)
        if success:
            waiters = []
            if plan_id in self.waiting_for_opt:
                waiters = self.waiting_for_opt[plan_id]

            # print('Stored motion plan', self.id)
            self.opt_queue.append((plan_id, state, task, condition, traj, waiters, node))


    def choose_mp_problems(self, samples):
        Xs = samples.get_X()[:,:,self.agent._x_data_idx[STATE_ENUM]]
        if self.traj_centers <= 1 or len(samples) == 1:
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
        self.node_ref[self.current_id] = next_sample.node if hasattr(next_sample, 'node') else None
        self.current_id += 1
        prob.solver_id = np.random.randint(0, self.n_optimizers)
        prob.server_id = self.id
        # self.store_for_opt(s_list)

        if self.alg_map[next_sample.task_name].mp_policy_prior.gmm.sigma is None:
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

        # print '\n\nSending motion plan problem to server {0}.\n\n'.format(prob.solver_id)
        self.async_plan_publisher.publish(prob)
        self.test_publisher.publish('MCTS sent motion plan.')
        self.n_opt_calls += 1

    def parse_state(self, sample):
        state_info = {}
        params = self.agent.plans.values()[0].params
        state = sample.get(STATE_ENUM)
        for param_name, attr in self.agent.state_inds:
            if params[param_name].is_symbol(): continue
            value = state[:, self.agent.state_inds[param_name, attr]]
            state_info[param_name, attr] = value
        return state_info


    def run_opt_queue(self):
        ### Guard against accidentally storing optimized results
        if not self.run_alg_updates:
            self.opt_queue = []

        while len(self.opt_queue):
            # print('Server', self.id, 'running opt queue')
            plan_id, state, task, condition, traj, waiters, node = self.opt_queue.pop()
            opt_sample = self.agent.sample_optimal_trajectory(state, task, condition, opt_traj=traj, traj_mean=[])
            samples = self.get_rollouts([opt_sample], self.task_list[task[0]])
            if node is not None:
                path = node.tree.get_path_info(opt_sample.get_X(t=0), node, opt_sample.task, opt_sample.get_X())
                val = node.tree.simulate(state, fixed_paths=[path], debug=False)

            assert len(samples) == 1
            self.store_opt_sample(opt_sample, plan_id, samples[0])
            info = self.parse_state(opt_sample)
            '''
            with open(self.rollout_log, 'a+') as f:
                f.write("Optimal rollout for {0} {1}:\n".format(self.task_list[opt_sample.task[0]], opt_sample.task))
                pp_info = pprint.pformat(info, width=50)
                f.write(pp_info)
                f.write('\n\n')
            '''

    def run_hl_opt_queue(self, mcts):
        ### Guard against accidentally storing optimized results
        if not self.run_alg_updates:
            self.hl_opt_queue = []

        while len(self.hl_opt_queue):
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


    def get_rollouts(self, init_samples, task_name):
        policy = DummyPolicy(task_name, self.policy_call)
        return self.agent.resample(init_samples, policy, self.num_samples)


    # def run_gps(self):
    #     rollout_policies = {task: DummyPolicy(task, self.policy_call) for task in self.agent.task_list}
    #     for task in self.task_list:
    #         inds = np.random.choice(range(len(self.rollout_opt_pairs[task])), min(self.num_conds, len(self.rollout_opt_pairs[task])), replace=False)
    #         opt_samples = []
    #         for i in inds:
    #             opt_sample, old_samples = self.rollout_opt_pairs[task][i]
    #             opt_samples.append(opt_sample)

    #         for i in range(self.traj_opt_steps):
    #             samples = self.agent.resample(opt_samples, rollout_policies[task], self.num_samples)
    #             self.alg_map[task].iteration([(opt_samples[j], samples[j]) for j in range(len(samples))], reset=i==0)


    def step(self):
        # print '\n\nTaking tree search step.\n\n'
        self.cur_step += 1
        rollout_policies = {task: DummyPolicy(task, self.policy_call) for task in self.agent.task_list}
        start_time = time.time()
        all_samples = []
        tree_data = []

        # if 'rollout_server_'+str(self.id) not in os.popen("rosnode list").read():
        #     print "\n\nRestarting dead ros node: rollout server\n\n", self.id
        #     rospy.init_node('rollout_server_'+str(self.id))
        
        # self.renew_publisher()


        if self.run_mcts_rollouts:
            ### If rospy hangs, don't want it to always be for the same trees
            random.shuffle(self.mcts)

            # print('Running MCTS for server', self.id)
            for mcts in self.mcts:
                self.update_weights()
                val = mcts.run(self.agent.x0[mcts.condition], 1, use_distilled=False, new_policies=rollout_policies, debug=False)
                self.run_opt_queue()
                # self.run_hl_opt_queue(mcts)
                # self.test_publisher.publish('MCTS Step')

                ### Used for DAgger
                # if np.random.uniform() < self.run_hl_prob and val > 0:
                #     init_state = self.agent.x0[mcts.condition]
                #     prob = HLProblem()
                #     prob.server_id = self.id
                #     prob.solver_id = np.random.randint(0, self.n_optimizers)
                #     prob.init_state = init_state.tolist()
                #     prob.cond = mcts.condition
                #     gmms = {}
                #     for task in self.task_list:
                #         gmm = self.alg_map[task].mp_policy_prior.gmm
                #         if gmm.sigma is None: continue
                #         gmms[task] = {}
                #         gmms[task]['mu'] = gmm.mu.flatten()
                #         gmms[task]['sigma'] = gmm.sigma.flatten()
                #         gmms[task]['logmass'] = gmm.logmass.flatten()
                #         gmms[task]['mass'] = gmm.mass.flatten()
                #         gmms[task]['N'] = len(gmm.mu)
                #         gmms[task]['K'] = len(gmm.mass)
                #         gmms[task]['Do'] = gmm.sigma.shape[1]
                #     prob.gmms = str(gmms)
                #     path = mcts.simulate(init_state, early_stop_prob=self.early_stop_prob)
                #     path_tuples = []
                #     for step in path:
                #         path_tuples.append(step.task)
                #     prob.path_to = str(path_tuples)
                #     self.hl_publisher.publish(prob)


                ### Collect observed samples from MCTS
                sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}
                self.agent.clear_samples(keep_prob=0.0, keep_opt_prob=0.0)
                n_probs = 0

                ### Check to see if ros node is dead; restart if so
                # if 'rollout_server_'+str(self.id) not in os.popen("rosnode list").read():
                #     print "\n\nRestarting dead ros node:", 'rollout_server_'+str(self.id), '\n\n'
                #     rospy.init_node('rollout_server_'+str(self.id))

                ### Stochastically choose subset of observed init states/problems to solve the motion planning problem for
                for task in sample_lists:
                    for ind, s_list in enumerate(sample_lists[task]):
                        mp_prob = np.random.uniform()
                        if mp_prob < self.opt_prob:
                            all_samples.extend(s_list._samples)

                            ### If multiple rollouts were taken per state, can extract multiple mean-trajectories
                            probs = self.choose_mp_problems(s_list)
                            n_probs += len(probs)

                            ### Send each chosen problem to the MP server
                            for p in probs:
                                self.send_mp_problem(*p)

                ### Run an update step of the algorithm if there are any waiting rollouts that already have an optimized result from the MP server
                for task in self.agent.task_list:
                    if len(self.rollout_opt_pairs[task]) > 1:
                        # sample_lists[task] = self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=False)
                        sample_inds = np.random.choice(range(len(self.rollout_opt_pairs[task])), min(self.num_conds, len(self.rollout_opt_pairs[task])), replace=False)
                        samples = [self.rollout_opt_pairs[task][ind] for ind in sample_inds]
                        self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=False)

                ### Save information on current performance
                tree_data.append(mcts.get_data)

                ### Publisher might be dead
                self.renew_publisher()
            # print('Ran through MCTS, for server', self.id)

        end_time = time.time()
        start_time_2 = time.time()
        end_time_2 = time.time()

        # if self.log_timing:
        #     with open(self.time_log, 'a+') as f:
        #         f.write('Generated {0} problems from {1} conditions with {2} rollouts per condition.\n'.format(n_probs, len(self.mcts), self.num_rollouts))
        #         f.write('Time to complete: {0}\n'.format(end_time-start_time))
        #         f.write('Time to select problems through kmeans and send to supervised learner: {0}\n\n'.format(end_time_2-start_time_2))


        ### Look for saved successful HL rollout paths and send them to update the HL options policy
        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)

        self.update_primitive(path_samples)
        self.update_qvalue(all_samples)

        start_time = time.time()

        ### Run an update step of the algorithm if there are any waiting rollouts that already have an optimized result from the MP server
        for task in self.agent.task_list:
            if len(self.rollout_opt_pairs[task]) > 1:
                sample_inds = np.random.choice(range(len(self.rollout_opt_pairs[task])), min(self.num_conds, len(self.rollout_opt_pairs[task])), replace=False)
                samples = [self.rollout_opt_pairs[task][ind] for ind in sample_inds]


                self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=False)

        end_time = time.time()
        if False: # self.log_timing:
            with open(self.time_log, 'a+') as f:
                f.write('Time to update algorithms for {0} iterations on data: {1}\n\n'.format(self.traj_opt_steps, end_time-start_time))

        # print '\n\nFinished tree search step.\n\n'


    def run(self):
        while not self.stopped:
            self.step()
            rospy.sleep(0.1)


    def update_qvalue(self, samples, first_ts_only=False):
        dV, dO = 1, self.agent.dVal

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            mu = sample.success * np.ones((sample.T, dV))
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = np.ones((sample.T,))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_val_obs()
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dV), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'value', 1)


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        for sample in samples:
            mu = sample.get_prim_out() # np.concatenate([sample.get(enum) for enum in self.prim_dims], axis=-1)
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = np.ones((sample.T,))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dP), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 1)
            
