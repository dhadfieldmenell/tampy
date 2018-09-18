import rospy

from std_msgs import String

from tamp_ros.msg import *
from tamp_ros.srv import *


class dummy_policy:
    def __init__(self, task):
        self.task = task
    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class RolloutServer(object):
    def __init__(self, mcts, algs, agent, cond, traj_opt_steps, n_samples, n_optimizers):
        self.mcts = mcts
        self.alg_map = algs
        self.agent = mcts.agent
        self.cond = cond
        self.traj_opt_steps = traj_opt_steps
        self.n_samples = n_samples
        self.stopped = False
        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=10) for task in self.alg_map}
        self.stop = rospy.Subscriber('terminate', String, self.end)
        for task in self.alg_map:
            self.alg_map[task].policy_opt.update = self.update
        self.n_optimizers = n_optimizers

    def end(self, msg):
        self.stopped = True

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

    def motion_plan(self, x, task, condition, traj_mean, targets):
        mp_id = np.random.randint(0, self.n_optimizers)
        proxy = rospy.ServiceProxy('motion_planner_'+mp_id, MotionPlan)
        mean = []
        for i in range(len(mean)):
            next_line = Float32MultiArray()
            next_line.data = traj_mean[i].tolist()
            mean.append(next_line)

        resp = proxy(x, task, targets[0].name, targets[1].name, condition, mean)
        failed = eval(resp.failed)
        success = resp.success
        traj = np.array([resp.traj[i].data for i in range(len(resp.traj))])


    def step(self):
        rollout_policies = {task: dummy_policy(task).act for task in self.agent.task_list}

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
