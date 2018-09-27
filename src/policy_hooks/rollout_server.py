import sys

from numba import cuda

import rospy

from std_msgs import String

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.policy_solver_utils import *


class dummy_policy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call

    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class RolloutServer(object):
    def __init__(self, hyperparams):
        self.mcts = hyperparams['mcts']
        self.alg_map = hyperparams['algs']
        self.agent = self.mcts[0].agent
        self.task_list = self.agent.task_list
        self.traj_opt_steps = hyperparams['traj_opt_steps']
        self.n_samples = ['n_samples']
        self.stopped = False
        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=50) for task in self.alg_map}
        self.updaters['value'] = rospy.Publisher('value_update', PolicyUpdate, queue_size=50)
        self.updaters['primitive'] = rospy.Publisher('primitive_update', PolicyUpdate, queue_size=50)
        self.mp_subcriber = rospy.Subscriber('motion_plan_result', MotionPlanResult, self.sample_mp)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob', MotionPlanProblem, queue_size=50)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        for task in self.alg_map:
            self.alg_map[task].policy_opt.update = self.update
            if not self.alg_map[task].policy_opt.sess._closed:
                self.alg_map[task].policy_opt.sess.close()
        cuda.close()
        self.n_optimizers = hyperparams['n_optimizers']
        self.waiting_for_opt = {}
        self.sample_queue = []
        self.current_id = 0
        self.optimized = {}
        self.max_sample_queue = hyperparams['max_sample_queue']
        self.max_opt_sample_queue = hyperparams['max_opt_sample_queue']

    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received signal to terminate.')

    def update(self, mu, obs, prc, wt, task):
        msg = PolicyUpdate()
        msg.obs = obs.flatten()
        msg.mu = mu.flatten()
        msg.prc = prc.flatten()
        msg.wt = wt.flatten()
        msg.dO = self.agent.dO
        msg.dU = self.agent.dU
        msg.n = len(mu)
        msg.rollout_len = mu.shape[1]
        self.updaters[task].publish(msg)

    def policy_call(self, x, obs, t, noise, task):
        rospy.wait_for_service(task+'_policy_act', timeout=10)
        proxy = rospy.ServiceProxy(task+'_policy_act', PolicyAct)
        resp = proxy(obs, noise, task)
        return resp.act

    def value_call(self, obs):
        rospy.wait_for_service('qvalue', timeout=10)
        proxy = rospy.ServiceProxy('qvalue', QValue)
        resp = proxy(obs)
        return resp.act

    def primitive_call(self, prim_obs):
        rospy.wait_for_service('primitive', timeout=10)
        proxy = rospy.ServiceProxy('primitive', Primitive)
        resp = proxy(prim_obs)
        return resp.task_distr, resp.obj_distr, resp.obj_distr

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
        req.condition condition
        req.mean = mean

        proxy = rospy.ServiceProxy('motion_planner_'+mp_id, MotionPlan)

        resp = proxy(req)
        failed = eval(resp.failed)
        success = resp.success
        traj = np.array([resp.traj[i].data for i in range(len(resp.traj))])


    def store_for_opt(self, samples):
        self.waiting_for_opt[self.current_id] = samples
        self.sample_queue.append(self.current_id)
        self.current_id += 1
        while len(self.sample_queue) > self.max_sample_queue:
            del self.waiting_for_opt[self.sample_queue[0]]
            del sample_queue[0]


    def store_opt_sample(self, sample, plan_id):
        if plan_id in self.waiting_for_opt:
            samples = self.waiting_for_opt[plan_id]
        else:
            samples = []

        self.opt_samples.append((sample, samples))
        while len(self.opt_samples) > self.max_opt_sample_queue:
            del self.opt_samples[0]


    def sample_mp(self, msg):
        plan_id = msg.id
        traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        success = msg.success
        failed = eval(msg.failed)
        task = msg.task
        condition = msg.condition
        obj = self.agent.plans.values()[0].params[msg.obj]
        targ = self.agent.plans.values()[0].params[msg.targ]
        if success:
            opt_sample = self.agent.sample_optimal_trajectory(traj[0], task, condition, traj, traj_mean=[], fixed_targets=[obj, targ])
            self.store_opt_traj(opt_sample, plan_id)


    def step(self):
        rollout_policies = {task: dummy_policy(task, self.policy_call) for task in self.agent.task_list}

        for mcts in self.mcts:
            mcts.run(self.agent.x0[mcts.condition], self.n_samples, False, new_policies=rollout_policies, debug=True)

        sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}
        self.agent.clear_samples(keep_prob=0.1, keep_opt_prob=0.2)
        all_samples = []

        for s_list in sample_lists:
            all_samples.extend(s_list._samples)
            next_sample = s_list[0]
            state = next_sample.get(STATE_ENUM, t=0)
            task = next_sample.task
            cond = next_sample.condition
            X = next_sample.get(STATE_ENUM)
            traj_mean = []
            for t in range(next_sample.T):
                next_line = Float32MultiArray()
                next_line.data = X[t]
                traj_mean.append(next_line)
            obj = next_sample.obj
            targ = next_sample.targ
            prob = MotionPlanProblem()
            prob.state = state
            prob.task = task
            prob.cond = cond
            prob.traj_mean = traj_mean
            prob.obj = obj
            prob.targ = targ
            prob.prob_id = self.current_id
            prob.solver_id = np.random.randint(0, self.n_optimizers)
            self.store_for_opt(s_list)
            self.async_plan_publisher.publish(prob)

        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)

        self.update_primitive(path_samples)
        self.update_qvalue(all_samples)

        for step in range(self.traj_opt_steps-1):
            for task in self.agent.task_list:
                try:
                    sample_lists[task] = self.alg_map[task].iteration(sample_lists[task], self.optimal_samples, reset=not step)
                    if len(sample_lists[task]):
                        sample_lists[task] = self.agent.resample(sample_lists[task], rollout_policies[task], self.n_samples)
                    else:
                        continue
                    self.agent._samples[task] = sample_lists[task]
                except:
                    traceback.print_exception(*sys.exc_info())
        self.agent.reset_sample_refs()

    def run(self):
        while not self.stopped:
            self.step()

    def update_qvalue(self, samples, first_ts_only=False):
        dV, dO = 2, self.alg_map.values()[0].dO

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            for t in range(sample.T):
                obs = [sample.get_obs(t=t)]
                mu = [sample.success]
                prc = [np.eye(dV)]
                wt = [10. / (t+1)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))
                if first_ts_only: break

        if len(tgt_mu):
            self.update(tgt_mu, obs_data, tgt_prc, tgt_wt, 'value')

    def update_primitive(self, samples):
        dP, dO = len(self.task_list), self.alg_map.values()[0].dPrimObs
        dObj, dTarg = self.alg_map.values()[0].dObj, self.alg_map.values()[0].dTarg
        dP += dObj + dTarg
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        for sample in samples:
            for t in range(sample.T):
                obs = [sample.get_prim_obs(t=t)]
                mu = [np.concatenate([sample.get(TASK_ENUM, t=t), sample.get(OBJ_ENUM, t=t), sample.get(TARG_ENUM, t=t)])]
                prc = [np.eye(dP)]
                wt = [1. / (t+1)] # [np.exp(-sample.task_cost)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.update(tgt_mu, obs_data, tgt_prc, tgt_wt, 'primitive')
