import rospy

from std_msgs import Float32MultiArray

class PolicyServer(object):
    def __init__(self, policy_opt, task):
        rospy.init_node(task+'_update_server')
        self.policy_opt = policy_opt
        self.policy_opt.hyperparams['scope'] = task
        self.task = task
        self.prob_service = rospy.Service(self.task+'_policy_prob', PolicyProb, self.prob)
        self.act_service = rospy.Service(self.task+'_policy_act', PolicyAct, self.act)
        self.update_listener = rospy.Subscriber(self.task+'_update', PolicyUpdate, self.update)

    def upate(self, msg):
        mu = np.array(msg.mu)
        mu_dims = (msg.n, msg.rollout_len, msg.dU)
        mu = mu.reshape(mu_dims)

        obs = np.array(msg.obs)
        obs_dims = (msg.n, msg.rollout_len, msg.dO)
        obs = obs.reshape(obs_dims)

        prc = np.array(msg.prc)
        prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
        prc = prc.reshape(prc_dims)

        wt = msg.wt

        self.policy_opt.store(obs, mu, prc, wt, self.task)


    def prob(self, req):
        obs = [req.obs[i].data for i in range(len(req.obs))]
        mu_out, sigma_out, _, _ = self.policy_opt.prob(np.array([obs]), task)
        mu, sigma = [], []
        for i in range(len(mu_out[0])):
            next_line = Float32MultiArray()
            next_line.data = mu[0, i]
            mu.append(next_line)

            next_line = Float32MultiArray()
            next_line.data = np.diag(sigma[0, i])

        return PolicyProbResponse(mu, sigma)


    def act(self, req):
        # Assume time invariant policy
        obs = req.obs
        noise = req.noise
        policy = self.policy_opt.task_map[self.task].policy
        if policy.scale is None:
            policy.scale = 1
            policy.bias = 0
            act = policy.act([], obs, 0, noise)
            policy.scale = None
            polcy.bias = None
        else:
            act = policy.act([], obs, 0, noise)
        return PolicyActResponse(act)
