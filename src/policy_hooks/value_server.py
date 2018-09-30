import rospy

from std_msgs.msg import Float32MultiArray

from tamp_ros.msg import *
from tamp_ros.srv import *


class ValueServer(object):
    def __init__(self, hyperparams):
        rospy.init_node('value_update_server')
        hyperparams['policy_opt']['scope'] = 'value'
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'], 
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dObj'],
            hyperparams['dTarg'],
            hyperparams['dPrimObs']
        )
        self.task = 'value'
        self.value_service = rospy.Service('qvalue', QValue, self.value)
        self.updater = rospy.Subscriber('value_update', PolicyUpdate, self.update)
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

        self.policy_opt.store(obs, mu, prc, wt, 'value')


    def value(self, req):
        value = self.policy_opt.value([req.obs])
        return QValueResponse(value)
