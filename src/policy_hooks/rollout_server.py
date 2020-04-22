from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time
from software_constants import *

from numba import cuda
from scipy.cluster.vq import kmeans2 as kmeans
import tensorflow as tf

from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList

if USE_ROS:
    import rospy
    from std_msgs.msg import Float32MultiArray, String
    from tamp_ros.msg import *
    from tamp_ros.srv import *

from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *

MAX_SAVED_NODES = 1000
MAX_BUFFER = 10


class DummyPolicy:
    def __init__(self, task, policy_call, opt_sample=None, chol_pol_covar=None, scale=None):
        self.task = task
        self.policy_call = policy_call
        self.opt_sample = opt_sample
        self.chol_pol_covar = chol_pol_covar
        self.scale = scale

    def act(self, x, obs, t, noise):
        U = self.policy_call(x, obs, t, noise, self.task, None)
        if np.any(np.isnan(x)):
            #raise Exception('Nans in policy call state.')
            print('Nans in policy call state.')
            U = np.zeros_like(U)
        if np.any(np.isnan(obs)):
            # raise Exception('Nans in policy call obs.')
            print('Nans in policy call obs.')
            U = np.zeros_like(U)
        if np.any(np.isnan(U)):
            # raise Exception('Nans in policy call action.')
            print('Nans in policy call action.')
            U = np.zeros_like(U)
        return U


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


class RolloutServer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        self._hyperparams = hyperparams
        self.config = hyperparams
        self.group_id = hyperparams['group_id']
        self.num_conds = hyperparams['num_conds']
        self.start_t = hyperparams['start_t']
        self.seed = int((1e2*time.time()) % 1000.)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.mcts = hyperparams['mcts']
        self.run_mcts_rollouts = hyperparams.get('run_mcts_rollouts', True)
        self.run_alg_updates = hyperparams.get('run_alg_updates', True)
        if not USE_ROS:
            self.queues = hyperparams['queues']

        self.run_hl_test = hyperparams.get('run_hl_test', False)
        if USE_ROS: rospy.init_node('rollout_server_{0}_{1}_{2}'.format(self.id, self.group_id, self.run_alg_updates))
        self.prim_dims = hyperparams['prim_dims']
        self.solver = hyperparams['solver_type'](hyperparams)
        self.opt_smooth = hyperparams.get('opt_smooth', False)
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.solver = self.solver
        #self.agent.replace_conditions(len(self.agent.x0))
        for mcts in self.mcts:
            mcts.agent = self.agent
            mcts.reset()
        # for c in range(len(self.agent.x0)):
        #     self.agent.replace_cond(c, curric_step=(1 if hyperparams.get('curric_thresh', 0) > 0 else 0))

        self.solver.agent = self.agent
        for i in range(len(self.mcts)):
            m = self.mcts[i]
            m.use_q = self.config.get('use_qfunc', False)
            m.discrete_prim = self.config.get('discrete_prim', True)
            m.value_func = self.value_call
            m.prob_func = self.primitive_call
            m.add_log_file('tf_saved/'+hyperparams['weight_dir']+'/mcts_{0}_{1}'.format(i, self.id))
            # m.log_file = 'tf_saved/'+hyperparams['weight_dir']+'/mcts_log_{0}_cond{1}.txt'.format(self.id, m.condition)
            # with open(m.log_file, 'w+') as f: f.write('')
        self.steps_to_replace = hyperparams.get('steps_to_replace', 1000)
        self.success_to_replace = hyperparams.get('success_to_replace', 1)
        self.alg_map = hyperparams['alg_map']
        for alg in self.alg_map.values():
            alg.set_conditions(len(self.agent.x0))
        self.task_list = self.agent.task_list
        self.label_options = self.mcts[0].label_options
        self.pol_list = tuple(hyperparams['policy_list'])
        self.traj_opt_steps = hyperparams['traj_opt_steps']
        self.num_samples = hyperparams['num_samples']
        for mcts in self.mcts:
            mcts.num_samples = self.num_samples
        self.num_rollouts = hyperparams['num_rollouts']
        self.stopped = False

        self.renew_publisher()
        self.last_log_t = time.time()

        for alg in self.alg_map.values():
            alg.policy_opt = DummyPolicyOpt(self.update, self.prob)
        self.n_optimizers = hyperparams['n_optimizers']
        self.waiting_for_opt = {}
        self.sample_queue = []
        self.current_id = 0
        self.cur_step = 0
        self.n_opt_calls = 0
        self.n_steps = 0
        self.n_success = 0
        self.n_sent_probs = 0
        self.n_received_probs = 0
        self.traj_costs = []
        self.latest_traj_costs = {}
        self.rollout_opt_pairs = {task: [] for task in self.task_list}
        self.max_sample_queue = int(hyperparams['max_sample_queue'])
        self.max_opt_sample_queue = int(hyperparams['max_opt_sample_queue'])
        self.early_stop_prob = hyperparams['mcts_early_stop_prob']
        self.run_hl_prob = hyperparams.get('run_hl_prob', 0)
        self.opt_prob = hyperparams.get('opt_prob', 0.1)
        self.opt_buffer = []
        self.add_negative = hyperparams['negative']
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.check_prim_t = hyperparams.get('check_prim_t', 1.)

        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] # + '_trained'
            hyperparams['policy_opt']['scope'] = None
            hyperparams['policy_opt']['gpu_fraction'] = 1./32.
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
        self.sampled_probs = []

        if not USE_OPENRAVE:
            self.agent.plans, self.agent.openrave_bodies, self.agent.env = self.agent.prob.get_plans()
            for plan in self.agent.plans.values():
                plan.state_inds = self.agent.state_inds
                plan.action_inds = self.agent.action_inds
                plan.dX = self.agent.dX
                plan.dU = self.agent.dU
                plan.symbolic_bound = self.agent.symbolic_bound
                plan.target_dim = self.agent.target_dim
                plan.target_inds = self.agent.target_inds

        self.rollout_log = 'tf_saved/'+hyperparams['weight_dir']+'/rollout_log_{0}_{1}.txt'.format(self.id, self.run_alg_updates)
        self.hl_test_log = 'tf_saved/'+hyperparams['weight_dir']+'/hl_test_log.npy'
        self.log_updates = []
        self.hl_data = []
        self.last_hl_test = time.time()
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

        if USE_ROS:
            #self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)

            # self.mp_subcriber = rospy.Subscriber('motion_plan_result_{0}_{1}'.format(self.id, self.group_id), MotionPlanResult, self.sample_mp, queue_size=3, buff_size=2**16)
            self.mp_subcriber = rospy.Subscriber('motion_plan_result_{0}'.format(self.group_id), MotionPlanResult, self.sample_mp, queue_size=10, buff_size=2**16)
            # self.prob_subcriber = rospy.Subscriber('prob_{0}_{1}'.format(self.id, self.group_id), MotionPlanProblem, self.store_prob, queue_size=3, buff_size=2**16)
            self.prob_subcriber = rospy.Subscriber('prob_{1}'.format(self.id, self.group_id), MotionPlanProblem, self.store_prob, queue_size=10, buff_size=2**16)
            #self.hl_subscriber = rospy.Subscriber('hl_result_{0}_{1}'.format(self.id, self.group_id), HLPlanResult, self.update_hl, queue_size=1)
            # self.weight_subscriber = rospy.Subscriber('tf_weights_{0}'.format(self.group_id), UpdateTF, self.store_weights, queue_size=1, buff_size=2**22)
            self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)
        self.node_ref = {}
        self.node_dict = {}


    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def renew_publisher(self):
        if not USE_ROS: return
        self.updaters = {task: rospy.Publisher(task+'_update_{0}'.format(self.group_id), PolicyUpdate, queue_size=2) for task in self.pol_list}
        self.updaters['value'] = rospy.Publisher('value_update_{0}'.format(self.group_id), PolicyUpdate, queue_size=5)
        self.updaters['primitive'] = rospy.Publisher('primitive_update_{0}'.format(self.group_id), PolicyUpdate, queue_size=5)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob_{0}'.format(self.group_id), MotionPlanProblem, queue_size=1)
        #self.prob_publisher = rospy.Publisher('prob_{0}_{1}'.format(self.id, self.group_id), MotionPlanProblem, queue_size=1)
        self.prob_publisher = rospy.Publisher('prob_{0}'.format(self.group_id), MotionPlanProblem, queue_size=1)
        #self.hl_publisher = rospy.Publisher('hl_prob_{0}'.format(self.group_id), HLProblem, queue_size=2)
        #self.test_publisher = rospy.Publisher('is_alive', String, queue_size=2)


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

        msg = PolicyUpdate() if USE_ROS else DummyMSG()
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
        msg.task = str(task)
        # if task != 'value':
        #     print('Sending update on', task)

        if USE_ROS:
            if task in self.updaters:
                self.updaters[task].publish(msg)
            else:
                self.updaters['control'].publish(msg)
        else:
            if '{0}_pol'.format(task) in self.queues:
                q = self.queues['{0}_pol'.format(task)]
            else:
                q = self.queues['control_pol']
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


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


    def policy_call(self, x, obs, t, noise, task, opt_s=None):
        # print 'Entering policy call:', datetime.now()
        if noise is None: noise = np.zeros(self.dU)
        if self.use_local:
            if 'control' in self.policy_opt.task_map:
                if self.policy_opt.task_map['control']['policy'].scale is None:
                    if opt_s is not None:
                        return opt_s.get_U(t) + self.alg_map[task].cur[0].traj_distr.chol_pol_covar[t].T.dot(noise)
                    return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
                return self.policy_opt.task_map['control']['policy'].act(x.copy(), obs.copy(), t, noise)
            else:
                if self.policy_opt.task_map[task]['policy'].scale is None:
                    if opt_s is not None:
                        return opt_s.get_U(t) + self.alg_map[task].cur[0].traj_distr.chol_pol_covar[t].T.dot(noise)
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


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None):
        if t > 0 and task is not None and t % self.check_prim_t: return task
        # print 'Entering primitive call:', datetime.now()
        if self.use_local:
            distrs = self.policy_opt.task_distr(prim_obs, eta)
            if not soft: return distrs
            out = []
            for d in distrs:
                p = d / np.sum(d)
                ind = np.random.choice(range(len(d)), p=p)
                d[ind] += 1.
                d /= np.sum(d)
                out.append(d)
            return out
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

        for s in samples:
            s.set_ref_X(opt_sample.get_ref_X())
            s.set_ref_U(opt_sample.get_ref_U())

        task_name = self.task_list[opt_sample.task[0]]
        self.rollout_opt_pairs[task_name].append((opt_sample, samples))
        self.rollout_opt_pairs[task_name] = self.rollout_opt_pairs[task_name][-self.max_opt_sample_queue:]
        # self.opt_buffer.append(opt_sample)
        # self.opt_buffer = self.opt_buffer[-MAX_BUFFER:]

    
    def store_prob(self, msg, check=True):
        task_name = self.task_list[eval(msg.task)[0]]
        if check and (not self.run_alg_updates or msg.alg_id != '{0}_{1}'.format(self.id, self.group_id)): return

        # print('storing ll prob for server {0} in group {1}'.format(self.id, self.group_id))
        if USE_ROS:
            traj_mean = []
            for t in range(len(msg.traj_mean)):
                traj_mean.append(msg.traj_mean[t].data)
        else:
            traj_mean = msg.traj_mean

        self.sampled_probs.append((msg, np.array(traj_mean)))
        self.sampled_probs = self.sampled_probs[-MAX_BUFFER:]


    def sample_mp(self, msg, check=True):
        ### If this server isn't running algorithm updates, it doesn't need to store the mp results
        # print('Received motion plan', self.group_id, self.id, msg.alg_id, msg.server_id)
        if (check and not self.run_alg_updates and msg.server_id != '{0}_{1}'.format(self.id, self.group_id)) or \
           (check and self.run_alg_updates and msg.alg_id != '{0}_{1}'.format(self.id, self.group_id)):
            return

        print('Server {0} in group {1} received solved plan: alg server? {2}'.format(self.id, self.group_id, self.run_alg_updates))
        self.n_opt_calls += 1
        self.n_received_probs += 1
        plan_id = msg.plan_id
        if USE_ROS:
            traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        else:
            traj = np.array(msg.traj)
        state = np.array(msg.state)
        success = msg.success
        task = eval(msg.task)
        condition = msg.cond
        node = self.node_ref.get(plan_id, None)
        if plan_id in self.node_ref: del self.node_ref[plan_id]
        if success:
            waiters = self.waiting_for_opt.pop(plan_id, [])

            # print('Stored motion plan', self.id)
            self.opt_queue.append((plan_id, state, task, condition, traj, waiters, node, np.array(msg.targets)))
        elif node is not None:
            node.tree.mark_failure(node, task)
        # else:
        #     print('Failure for {0} on server {1} and sate {2}'.format(task, self.id, state))


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


    def add_to_queue(self, prob, q):
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(prob)
        except queue.Full:
            pass


    def send_prob(self, centroid, s_list):
        # print('sending ll prob for server {0} in group {1}'.format(self.id, self.group_id))
        if self._hyperparams['n_alg_servers'] == 0:
            return

        next_sample = s_list[0]
        if not np.any(np.array(next_sample.use_ts) > 0): return
        state = next_sample.get_X(t=0)
        task = next_sample.task
        cond = next_sample.condition

        if USE_ROS:
            traj_mean = []
            for t in range(next_sample.T):
                next_line = Float32MultiArray()
                next_line.data = next_sample.get_X(t=t).tolist()
                traj_mean.append(next_line)
            obs = []
            for t in range(next_sample.T):
                next_line = Float32MultiArray()
                next_line.data = next_sample.get_obs(t=t).tolist()
                obs.append(next_line)

            U = []
            for t in range(next_sample.T):
                next_line = Float32MultiArray()
                next_line.data = next_sample.get(ACTION_ENUM, t=t).tolist()
                U.append(next_line)

        else:
            traj_mean = next_sample.get_X()
            obs = next_sample.get_obs()
            U = next_sample.get(ACTION_ENUM)

        prob = MotionPlanProblem() if USE_ROS else DummyMSG()
        prob.state = state
        prob.targets = self.agent.target_vecs[cond].tolist()
        prob.task = str(task)
        prob.cond = cond
        prob.traj_mean = traj_mean
        prob.obs = obs
        prob.U = U
        alg_id = np.random.randint(self._hyperparams['n_alg_servers'])
        prob.alg_id = '{0}_{1}'.format(alg_id, self.group_id)
        prob.server_id = str('{0}_{1}'.format(self.id, self.group_id))
        prob.prob_id = self.current_id
        self.node_dict[self.current_id] = next_sample.node if hasattr(next_sample, 'node') else None
        if USE_ROS:
            self.prob_publisher.publish(prob)
        elif 'rollout_opt_rec{0}'.format(self.id) in self.queues:
            prob.rollout_id = self.id
            prob.algorithm_id = alg_id
            q = self.queues['alg_prob_rec{0}'.format(alg_id)]
            self.add_to_queue(prob, q)
        self.current_id += 1
        self.n_sent_probs += 1


    def send_all_mp_problems(self):
        i = len(self.sampled_probs)
        while i > 0:
            prob, traj = self.sampled_probs.pop()

            cond = prob.cond
            task_name = self.agent.task_list[eval(prob.task)[0]]
            self.agent.T = len(traj)
            s = Sample(self.agent)
            s.targets = np.array(prob.targets)

            for t in range(len(traj)):
                s.set_X(traj[t], t=t)
                if USE_ROS:
                    s.set_obs(np.array(prob.obs[t].data), t=t)
                    s.set(ACTION_ENUM, np.array(prob.U[t].data), t=t)
                else:
                    s.set_obs(np.array(prob.obs[t]), t=t)
                    s.set(ACTION_ENUM, np.array(prob.U[t]), t=t)

            s.condition = cond
            s.task = eval(prob.task)
            s_list = SampleList([s])
            if not self.alg_map[task_name].mp_opt:
                s_list = SampleList(self.get_rollouts(s_list, task_name)[0])

            if USE_ROS:
                for t in range(len(traj)):
                    prob.traj_mean[t].data = traj[t][self.agent._x_data_idx[STATE_ENUM]].tolist()
            else:
                prob.traj_mean = traj[:,self.agent._x_data_idx[STATE_ENUM]]

            # prob.prob_id = self.current_id
            self.waiting_for_opt[self.current_id] = s_list
            prob.use_prior = False
            if len(s_list) > 1:
                self.waiting_for_opt[self.current_id] = s_list
                self.alg_map[task_name].set_conditions(len(self.agent.x0))
                self.alg_map[task_name].cur[cond].sample_list = s_list
                self.alg_map[task_name]._update_policy_fit(cond)
                prob.use_prior = True

                pol_info = self.alg_map[task_name].cur[cond].pol_info
                K, k, S = pol_info.pol_K, pol_info.pol_k, pol_info.pol_S
                prob.T = len(prob.traj_mean)
                prob.dU = self.agent.dU
                prob.dX = self.agent.dX
                prob.pol_K, prob.pol_k, prob.pol_S = K.flatten().tolist(), k.flatten().tolist(), S.flatten().tolist()

                try:
                    prob.chol = np.linalg.inv(pol_info.pol_S + np.tile(np.eye(s.dU), (s.T, 1, 1))).flatten().tolist() # pol_info.chol_pol_S.flatten().tolist()
                except Exception:
                    prob.use_prior = False

            #prob.prob_id = self.current_id
            #self.current_id += 1
            prob.solver_id = np.random.randint(0, self.n_optimizers)
            # prob.server_id = str('{0}_{1}'.format(self.id, self.group_id))
            if np.random.uniform() < 1.: # self.opt_prob:
                self.n_sent_probs += 1
                if USE_ROS:
                    self.async_plan_publisher.publish(prob)
                elif 'rollout_opt_rec{0}'.format(self.id) in self.queues:
                    q = self.queues['optimizer{0}'.format(prob.solver_id)]
                    self.add_to_queue(prob, q)

                '''
                prob.server_id = 'no_id'
                prob.prob_id = -1
                prob.state = traj[-1]
                prob.traj_mean = []
                self.async_plan_publisher.publish(prob)
                '''
            if self.alg_map[task_name].mp_opt:
                self.rollout_opt_pairs[task_name].append((None, [s]))
                self.rollout_opt_pairs[task_name] = self.rollout_opt_pairs[task_name][-self.max_opt_sample_queue:]
            i -= 1


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

        task_name = next_sample.task_name
        if len(s_list) == 1:
            s_list = self.get_rollouts(s_list, task_name)

        if len(s_list) > 1:
            self.waiting_for_opt[self.current_id] = s_list
            self.alg_map[task_name].set_conditions(len(self.agent.x0))
            self.alg_map[task_name].cur[cond].sample_list = s_list
            self.alg_map[task_name]._update_policy_fit(cond)
        
        prob.use_prior = True
        pol_info = self.alg_map[task_name].cur[cond].pol_info
        K, k, S = pol_info.pol_K, pol_info.pol_k, pol_info.pol_S
        prob.T = next_sample.T
        prob.dU = self.agent.dU
        prob.dX = self.agent.dX
        prob.pol_K, prob.pol_k, prob.pol_S = K.flatten().tolist(), k.flatten().tolist(), S.flatten().tolist()

        try:
            prob.chol = np.linalg.inv(pol_info.pol_S + np.tile(np.eye(next_sample.dU), (next_sample.T, 1, 1))).flatten().tolist() # pol_info.chol_pol_S.flatten().tolist()
        except Exception:
            prob.use_prior = False

        prob.traj_mean = traj_mean
        prob.prob_id = self.current_id
        self.node_ref[self.current_id] = next_sample.node if hasattr(next_sample, 'node') else None
        self.current_id += 1
        prob.solver_id = np.random.randint(0, self.n_optimizers)
        prob.server_id = str('{0}_{1}'.format(self.id, self.group_id))
        '''
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
        '''

        # print '\n\nSending motion plan problem to server {0}.\n\n'.format(prob.solver_id)
        self.async_plan_publisher.publish(prob)

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
        while len(self.opt_queue):
            # print('RUNNING OPT QUEUE {0} IS ALG: {1}'.format(self.id, self.run_alg_updates))
            # print('Server', self.id, 'running opt queue')
            plan_id, state, task, condition, traj, waiters, node, targets = self.opt_queue.pop()
            if self.run_mcts_rollouts:
                if node is None: node = self.node_dict.pop(plan_id, None)
                if node is not None and self.agent.prob.OPT_MCTS_FEEDBACK and node.valid:
                    # print('Sampling opt sample in tree')
                    # old_targets - self.agent.target_vecs[condition]
                    # self.agent.target_vecs[condition] = targets
                    path = node.tree.get_path_info(state, node, task, traj)
                    val = node.tree.simulate(state, fixed_paths=[path], debug=False)
                    # self.agent.target_vecs[condition] = old_targets

            if self.run_alg_updates:
                opt_sample = self.agent.sample_optimal_trajectory(state, task, condition, opt_traj=traj, traj_mean=[], targets=targets)
                # if node is not None and self.agent.prob.OPT_MCTS_FEEDBACK:
                #     path = node.tree.get_path_info(opt_sample.get_X(t=0), node, opt_sample.task, opt_sample.get_X())
                #     val = node.tree.simulate(state, fixed_paths=[path], debug=False)

                '''
                if self.opt_smooth:
                    s = []

                    for sample in samples[0]:
                        task = sample.task_name
                        info = self.alg_map[sample.task].cur[sample.condition].pol_info
                        try:
                            inf_f = (info.pol_K, info.pol_k, np.linalg.inv(pol_inf.pol_S))
                        except:
                            inf_f = None
                        gmm = self.alg_map[task].mp_policy_prior.gmm
                        if gmm.sigma is not None:
                            inf_f = None # lambda x: self.gmm_inf(gmm, x)
                        else:
                            inf_f = None
                        out, failed, success = self.agent.solve_sample_opt_traj(sample.get_X(t=0), sample.task, sample.condition, sample.get_X(), inf_f, t_limit=5, n_resamples=1, out_coeff=1e3, smoothing=True)
                        s.append(out)
                        T = sample.T
                        out.set(ACTION_ENUM, sample.get_U(T-1), T-1)
                    samples[0] = s
                '''
                self.store_opt_sample(opt_sample, plan_id, waiters)
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
        policy = DummyPolicy(task_name, self.policy_call)#, opt_sample=init_samples[0])
        samples = self.agent.resample(init_samples, policy, self.num_samples)
        return samples


    def run_gps(self):
        rollout_policies = {task: DummyPolicy(task, self.policy_call) for task in self.agent.task_list}
        for task in self.task_list:
            inds = np.random.choice(range(len(self.rollout_opt_pairs[task])), min(self.num_conds, len(self.rollout_opt_pairs[task])), replace=False)
            opt_samples = []
            for i in inds:
                opt_sample, old_samples = self.rollout_opt_pairs[task][i]
                opt_samples.append(opt_sample)

            for i in range(self.traj_opt_steps):
                samples = self.agent.resample(opt_samples, rollout_policies[task], self.num_samples)
                self.alg_map[task].iteration([(opt_samples[j], samples[j]) for j in range(len(samples))], reset=i==0)


    def step(self):
        # print '\n\nTaking tree search step.\n\n'
        if self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()

        self.cur_step += 1
        chol_pol_covar = {}
        for task in self.agent.task_list:
            if task not in self.policy_opt.valid_scopes:
                task_name = 'control'
            else:
                task_name = task

            if self.policy_opt.task_map[task_name]['policy'].scale is None:
                chol_pol_covar[task] = np.eye(self.agent.dU) # self.alg_map[task].cur[0].traj_distr.chol_pol_covar
            else:
                chol_pol_covar[task] = self.policy_opt.task_map[task_name]['policy'].chol_pol_covar
          
        rollout_policies = {task: DummyPolicy(task, 
                                              self.policy_call, 
                                              chol_pol_covar=chol_pol_covar[task],
                                              scale=self.policy_opt.task_map[task if task in self.policy_opt.valid_scopes else 'control']['policy'].scale) 
                                  for task in self.agent.task_list}
        start_time = time.time()
        all_samples = []
        tree_data = []

        # if 'rollout_server_'+str(self.id) not in os.popen("rosnode list").read():
        #     print "\n\nRestarting dead ros node: rollout server\n\n", self.id
        #     rospy.init_node('rollout_server_'+str(self.id))
        
        # self.renew_publisher()

        task_costs = {}
        if self.run_mcts_rollouts:
            ### If rospy hangs, don't want it to always be for the same trees
            # random.shuffle(self.mcts)

            # print('Running MCTS for server', self.id)
            start_t = time.time()
            self.update_weights()
            for mcts in self.mcts:
                if self.policy_opt.share_buffers: #  and self.cur_step > 1:
                    self.policy_opt.read_shared_weights()

                start_t = time.time()
                mcts.opt_strength = 0.0 # if np.all([self.policy_opt.task_map[task]['policy'].scale is not None for task in self.policy_opt.valid_scopes]) else 1.0
                val = mcts.run(self.agent.x0[mcts.condition], 1, use_distilled=False, new_policies=rollout_policies, debug=False)
                tree_data.append(mcts.get_data)

                self.n_steps += 1
                self.n_success += 1 if val > 1 - 1e-2 else 0

                # if mcts.n_runs > self.steps_to_replace or mcts.n_success > self.success_to_replace:
                #     mcts.reset()
                # self.run_hl_opt_queue(mcts)
                # self.test_publisher.publish('MCTS Step')

                ### Used for DAgger
                # if np.random.uniform() < self.run_hl_prob and val > 0:
                #     init_state = self.agent.x0[mcts.condition]
                #     prob = HLProblem()
                #     # prob.server_id = self.id
                #     prob.server_id = str('{0}_{1}'.format(self.id, self.group_id))
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

                # print('MCTS step time:', time.time() - start_t)
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
                    task_costs[task] = []
                    for ind, s_list in enumerate(sample_lists[task]):
                        all_samples.extend(s_list._samples)
                        cost = 1.
                        if len(s_list):
                            s = s_list[0]
                            Xs = s.get_X(t=s.T-1)
                            t = self.agent.plans[s.task].horizon-1
                            cost = s.post_cost
                            # if s.task_cost < 1e-3: cost = 0.
                            if s.opt_strength > 0.5: cost = 1.
                        if cost < 1e-3: continue
                        # viol, failed = self.check_traj_cost(s_list[0].get(STATE_ENUM), s_list[0].task)
                        # task_costs[task].append(viol)

                        mp_prob = np.random.uniform()
                        if mp_prob < self.opt_prob:
                            ### If multiple rollouts were taken per state, can extract multiple mean-trajectories
                            probs = self.choose_mp_problems(s_list)
                            n_probs += len(probs)

                            ### Send each chosen problem to the MP server
                            for p in probs:
                                self.send_prob(*p)
                            #     self.send_mp_problem(*p)

                if mcts.n_runs >= self.steps_to_replace or mcts.n_success >= self.success_to_replace:
                    mcts.reset()
                
            ### Run an update step of the algorithm if there are any waiting rollouts that already have an optimized result from the MP server
            # self.run_opt_queue()
            # for task in self.agent.task_list:
            #     if len(self.rollout_opt_pairs[task]) > 1:
            #         # sample_lists[task] = self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=False)
            #         sample_inds = np.random.choice(range(len(self.rollout_opt_pairs[task])), min(self.num_conds, len(self.rollout_opt_pairs[task])), replace=False)
            #         samples = [self.rollout_opt_pairs[task][ind] for ind in sample_inds]
            #         self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=False)

            ### Publisher might be dead
            # self.renew_publisher()
            # print('Ran through MCTS, for server', self.id)

            ### Look for saved successful HL rollout paths and send them to update the HL options policy
            path_samples = []
            for path in self.agent.get_task_paths():
                path_samples.extend(path)
            self.agent.clear_task_paths()

            self.update_primitive(path_samples)
            if self.config.get('use_qfunc', False):
                self.update_qvalue(all_samples)
            # print('Time to finish all MCTS step:', time.time() - start_t)

        self.run_opt_queue()
        costs = {}
        if self.run_alg_updates:
            ### Run an update step of the algorithm if there are any waiting rollouts that already have an optimized result from the MP server
            self.send_all_mp_problems()
            # print('Time to finish opt queue', time.time() - start_t)
            start_t = time.time()

            for task in self.agent.task_list:
                if len(self.rollout_opt_pairs[task]) > 1:
                    '''
                    for opt, s in self.rollout_opt_pairs[task]:
                        if s is None or not len(s): continue
                        viol, failed = self.check_traj_cost(s[0].get(STATE_ENUM), s[0].task, s[0].targets, active_ts=(s[0].T-1, s[0].T-1))
                        if task not in costs:
                            costs[task] = []
                        costs[task].append(viol)
                    '''
                    # task_costs[task].append(viol)
                    # print('Server {0} in group {1} updating policy'.format(self.id, self.group_id))
                    self.alg_map[task].iteration(self.rollout_opt_pairs[task], reset=True)

            # if time.time() - start_t > 1:
            #     print('Time to finish alg iteration', time.time() - start_t)

            if len(costs.keys()):
                self.latest_traj_costs.clear()
                self.latest_traj_costs.update(costs)

            if (time.time() - self.last_log_t > 120) and self.rollout_log is not None:
                info = {'id': self.id, 
                        'n_opt_calls': self.n_opt_calls,
                        'n_steps': self.n_steps,
                        'cur_step': self.cur_step,
                        # 'x0': [list(x) for x in self.agent.x0],
                        # 'targets': [list(t) for t in self.agent.target_vecs],
                        'time': time.time() - self.start_t,
                        'alg_server': self.run_alg_updates,
                        'traj_cost': self.latest_traj_costs.copy(),
                        'n_sent_probs': self.n_sent_probs,
                        'n_received_probs': self.n_received_probs}
                self.log_updates.append(info)
                with open(self.rollout_log, 'w+') as f:
                    f.write(str(self.log_updates))
                self.last_log_t = time.time()

        else:
            if self.rollout_log is not None:
                post_cond = [np.mean(mcts.post_cond) for mcts in self.mcts if len(mcts.post_cond)]
                prim_cost = [np.mean(mcts.prim_pre_cond[-10:]) for mcts in self.mcts if len(mcts.prim_pre_cond)]
                if len(post_cond):
                    post_cond = np.mean(post_cond)
                else:
                    post_cond = 1.
                info = {'id': self.id, 
                        'n_success': self.n_success,
                        'n_opt_calls': self.n_opt_calls,
                        'n_steps': self.n_steps,
                        'cur_step': self.cur_step,
                        # 'x0': [list(x) for x in self.agent.x0],
                        # 'targets': [list(t) for t in self.agent.target_vecs],
                        'time': time.time() - self.start_t,
                        'avg_value_per_run': np.mean([np.mean(mcts.val_per_run) for mcts in self.mcts]),
                        'avg_first_success': np.mean([mcts.first_success for mcts in self.mcts]),
                        'n_sent_probs': self.n_sent_probs,
                        'n_received_probs': self.n_received_probs,
                        'alg_server': self.run_alg_updates,
                        'avg_post_cond': post_cond,
                        'prim_cost': prim_cost,
                        }
                self.log_updates.append(info)
                with open(self.rollout_log, 'w+') as f:
                    f.write(str(self.log_updates))

            print('Finished tree search step.')


    def test_hl(self, rlen=None, save=True):
        if self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()

        chol_pol_covar = {}
        for task in self.agent.task_list:
            if task not in self.policy_opt.valid_scopes:
                task_name = 'control'
            else:
                task_name = task

            if self.policy_opt.task_map[task_name]['policy'].scale is None:
                chol_pol_covar[task] = np.eye(self.agent.dU) # self.alg_map[task].cur[0].traj_distr.chol_pol_covar
            else:
                chol_pol_covar[task] = self.policy_opt.task_map[task_name]['policy'].chol_pol_covar
          
        rollout_policies = {task: DummyPolicy(task, 
                                              self.policy_call, 
                                              chol_pol_covar=chol_pol_covar[task],
                                              scale=self.policy_opt.task_map[task if task in self.policy_opt.valid_scopes else 'control']['policy'].scale) 
                                  for task in self.agent.task_list}
        self.mcts[0].rollout_policy = rollout_policies
        prim_opts = self.agent.prob.get_prim_choices()
        if OBJ_ENUM not in prim_opts: return
        n_targs = range(len(prim_opts[OBJ_ENUM]))
        res = []
        ns = [self.config['num_targs']]
        if self.config['curric_thresh'] > 0:
            ns = list(range(1, self.config['num_targs']+1))
        # n = np.random.choice(ns, p=[flt(i)/np.sum(ns) for i in ns])
        n = np.random.choice(ns)
        s = []
        self.agent.replace_cond(0)
        x0 = self.agent.x0[0]
        targets = self.agent.target_vecs[0].copy()
        for t in range(n, n_targs[-1]):
            obj_name = prim_opts[OBJ_ENUM][t]
            targets[self.agent.target_inds['{0}_end_target'.format(obj_name), 'value']] = x0[self.agent.state_inds[obj_name, 'pose']]
        if rlen is None:
            rlen = 6 + 2*n
        self.agent.T = self.config['task_durations'][self.task_list[0]]
        val, path = self.mcts[0].test_run(x0, targets, rlen, hl=True, soft=self.config['soft'])
        s.append((val, len(path), n, time.time()-self.start_t, self.config['num_objs'], n))
        # print('EXPLORED PATH: {0}'.format([sample.task for sample in path]))
        res.append(s[0])
        self.hl_data.append(res)
        if save:
            if not len(self.hl_data) % 20:
                np.save(self.hl_test_log, np.array(self.hl_data))
        else:
            if val < 1:
                print('failed for', x0, [s.task for s in path], [s.get_prim_obs(t=0) for s in path])
            else:
                print('succeeded for', path[0].get_X(t=0))
            print('n_success', len([d for d in self.hl_data if d[0][0] > 1-1e-3]))
            print('n_runs', len(self.hl_data))
        self.last_hl_test = time.time()
        print('TESTED HL')


    def parse_opt_queue(self):
        i = 0
        if self.run_alg_updates:
            lab = 'alg_opt_rec{0}'.format(self.id)
        else:
            lab = 'rollout_opt_rec{0}'.format(self.id)
        if lab not in self.queues: return
        q = self.queues[lab]

        while i < q._maxsize and not q.empty():
            try:
                prob = q.get_nowait()
                self.sample_mp(prob, check=False)
            except queue.Empty:
                break
            i += 1
        # if i > 0 and self.run_alg_updates:
        #     print('Parsed {0} from opt ALG'.format(i))

    def parse_prob_queue(self):
        if not self.run_alg_updates: return
        i = 0
        lab = 'alg_prob_rec{0}'.format(self.id)
        if lab not in self.queues: return
        q = self.queues[lab]
        while i < q._maxsize and not q.empty():
            try:
                prob = q.get_nowait()
                self.store_prob(prob, check=False)
            except queue.Empty:
                break
            i += 1


    def run(self):
        step = 0
        while not self.stopped:
            if self.run_hl_test:
                self.test_hl()
            else:
                if not USE_ROS:
                    self.parse_opt_queue()
                    if self.run_alg_updates:
                        self.parse_prob_queue()
                self.step()
            step += 1
            if time.time() - self.start_t > self._hyperparams['time_limit']:
                break
        self.policy_opt.sess.close()


    def find_negative_ex(self, sample):
        bad = []

        for t in range(sample.T):
            x = sample.get(STATE_ENUM, t)
            inds = np.random.permutation(range(len(self.label_options)))
            for i in inds:
                l = self.label_options[i]
                cost = self.agent.cost_f(x, l, sample.condition, active_ts=(0,0), targets=sample.targets)
                if cost > 0:
                    task_vec = np.zeros(len(self.task_list))
                    task_vec[l[0]] = 1.
                    desc = [task_vec]
                    for j, enum in enumerate(self.prim_dims):
                        desc.append(np.zeros(self.prim_dims[enum]))
                        desc[-1][l[j+1]] = 1.
                    vec = np.concatenate(desc)
                    bad.append(vec)
                    break
                elif i == inds[-1]:
                    bad.append(np.zeros(sum(self.prim_dims.values())))
                    break
        assert len(bad) == sample.T
        return np.array(bad)


    def update_qvalue(self, samples, first_ts_only=False):
        dV, dO = 1, self.agent.dVal

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            mu = sample.success * np.ones((sample.T, dV))
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = sample.discount * np.ones((sample.T,))
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
            mu = np.concatenate([sample.get(enum) for enum in self.config['prim_out_include']], axis=-1)
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = sample.discount * np.array([self.prim_decay**t for t in range(sample.T)])
            wt[0] *= self.prim_first_wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dP), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))
            if self.add_negative:
                mu = self.find_negative_ex(sample)
                tgt_mu = np.concatenate((tgt_mu, mu))
                wt = -np.ones((sample.T,))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs = sample.get_prim_obs()
                obs_data = np.concatenate((obs_data, obs))
                prc = np.tile(np.eye(dP), (sample.T,1,1))
                tgt_prc = np.concatenate((tgt_prc, prc))
            
        if len(tgt_mu):
            print('Sending update to primitive net')
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 1)
           

    def gmm_inf(self, gmm, sample):
        dux = self.agent.symbolic_bound + self.agent.dU
        mu, sig = np.zeros((sample.T, dux)), np.zeros((sample.T, dux, dux))
        for t in range(sample.T):
            mu0, sig0, _, _ = gmm.inference(np.concatenate([sample.get(STATE_ENUM, t=t), sample.get_U(t=t)], axis=-1).reshape((1, -1)))
            mu[t] = mu0
            sig[t] = sig0
        return mu, sig, False, True


    def check_traj_cost(self, traj, task, targets=[], active_ts=None):
        if np.any(np.isnan(np.array(traj))):
           raise Exception('Nans in trajectory passed')
        plan = self.agent.plans[task]
        if active_ts is None:
            active_ts = (1, plan.horizon-1)
        old_free_attrs = plan.get_free_attrs()
        if len(targets):
            for targ, val in self.agent.target_inds:
                getattr(plan.params[targ], 'value')[:,0] = targets[self.agent.target_inds[targ, val]]

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

        for p in plan.params.values():
            if p.is_symbol():
                p.free_all_attr((t,t))
        # Take a quick solve to resolve undetermined symbolic values
        self.solver._backtrack_solve(plan, time_limit=2, max_priority=-1, task=task)
        plan.store_free_attrs(old_free_attrs)
       
        viols = plan.check_cnt_violation(active_ts=active_ts)
        for i in range(len(viols)):
            if np.isnan(viols[i]):
                viols[i] = 1e3
        plan_total_violation = np.sum(viols) / plan.horizon
        plan_failed_constrs = plan.get_failed_preds_by_type(active_ts=active_ts)
        return plan_total_violation, plan_failed_constrs


