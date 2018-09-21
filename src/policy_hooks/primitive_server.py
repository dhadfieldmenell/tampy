import rospy

from std_msgs import Float32MultiArray

from tamp_ros.msg import *
from tamp_ros.srv import *


class PrimitiveServer(object):
    def __init__(self, hyperparams, dO, dU, dObj, dTarg, dPrimObs):
        rospy.init_node('primitive_update_server')
        hyperparams['scope'] = 'primitive'
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'], dO, dU, dObj, dTarg, dPrimObs
        )
        self.task = 'primitive'
        self.primitive_service = rospy.Service('primitive', Primitive, self.primitive)
        self.updater = rospy.Subscriber('primitive_update', PolicyUpdate, self.update)
        self.stop = rospy.Subscriber('terminate', str, self.end)
        self.stoped = True
        rospy.spin()


    def run(self):
        while not self.stopped:
            rospy.sleep(0.01)


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


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

        self.policy_opt.store(obs, mu, prc, wt, 'primitive')


    def primitive(self, req):
        task_distr, obj_distr, targ_distr = self.policy_opt.value([req.prim_obs])
        return PrimitiveResponse(task_distr.tolist(), obj_distr.tolist(), targ_distr.tolist())
