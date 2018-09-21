import sys

import rospy

from std_msgs import String

from tamp_ros.msg import *
from tamp_ros.srv import *


class dummy_policy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call
    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class RolloutServer(object):
    def __init__(self, mcts, algs, traj_opt_steps, n_samples, n_optimizers, max_sample_queue):
        self.mcts = mcts
        self.alg_map = algs
        self.agent = mcts[0].agent
        self.traj_opt_steps = traj_opt_steps
        self.n_samples = n_samples
        self.stopped = False
        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=50) for task in self.alg_map}
        self.updaters['value'] = rospy.Publisher('value_update', PolicyUpdate, queue_size=50)
        self.updaters['primitive'] = rospy.Publisher('primitive_update', PolicyUpdate, queue_size=50)
        self.mp_subcriber = rospy.Subscriber('motion_plan_result', MotionPlanResult, self.sample_mp)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob', MotionPlanProblem, queue_size=50)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        for task in self.alg_map:
            self.alg_map[task].policy_opt.update = self.update
        self.n_optimizers = n_optimizers
        self.waiting_for_opt = {}
        self.sample_queue = []
        self.current_id = 0
        self.optimized = {}
        self.max_sample_queue = max_sample_queue

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
        proxy = rospy.ServiceProxy(task+'_policy_act', PolicyAct)
        resp = proxy(obs, noise, task)
        return resp.act

    def value_call(self, obs):
        proxy = rospy.ServiceProxy('qvalue', QValue)
        resp = proxy(obs)
        return resp.act

    def primitive_call(self, prim_obs):
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
            self.sample_queue.pop(0)


    def store_opt_sample(self, sample, plan_id):
        if plan_id in self.waiting_for_opt:
            samples = self.waiting_for_opt[plan_id]
        else:
            samples = []
        self.opt_samples.append((sample, samples))


    def sample_mp(self, msg):
        plan_id = msg.id
        traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        success = msg.success
        failed = eval(msg.failed)
        if success:
            opt_sample = self.agent.sample_optimal_trajectory()
            self.store_opt_sample(opt_sample, plan_id)


    def step(self):
        rollout_policies = {task: dummy_policy(task, self.policy_call).act for task in self.agent.task_list}

        self.mcts[cond].run(self.agent.x0[self.cond], self.n_samples, False, new_policies=rollout_policies, debug=True)

        sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}
        self.agent.clear_samples(keep_prob=0.2, keep_opt_prob=0.2)

        for step in range(self.traj_opt_steps):
            for task in self.agent.task_list:
                try:
                    sample_lists[task] = self.alg_map[task].iteration(sample_lists[task], self.agent.optimal_samples[task], reset=not step)
                    if len(sample_lists[task]) and step < traj_opt_steps - 1:
                        sample_lists[task] = self.agent.resample(sample_lists[task], rollout_policies[task], self.n_samples)
                    self.agent._samples[task] = sample_lists[task]
                except:
                    traceback.print_exception(*sys.exc_info())
        self.agent.reset_sample_refs()

    def run(self):
        while not self.stopped:
            self.step()
