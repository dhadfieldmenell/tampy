import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, String

from tamp_ros.msg import *
from tamp_ros.srv import *


class PrimitiveServer(object):
    def __init__(self, hyperparams):
        import tensorflow as tf
        rospy.init_node('primitive_update_server')
        hyperparams['policy_opt']['scope'] = 'primitive'
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'], 
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dObj'],
            hyperparams['dTarg'],
            hyperparams['dPrimObs']
        )
        self.task = 'primitive'
        self.primitive_service = rospy.Service('primitive', Primitive, self.primitive)
        self.updater = rospy.Subscriber('primitive_update', PolicyUpdate, self.update)
        self.weight_publisher = rospy.Publisher('tf_weights', String, queue_size=1)
        self.stop = rospy.Subscriber('terminate', String, self.end)
        self.stopped = True
        rospy.spin()


    def run(self):
        while not self.stopped:
            rospy.sleep(0.01)


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def update(self, msg):
        mu = np.array(msg.mu)
        mu_dims = (msg.n, msg.rollout_len, msg.dU)
        mu = mu.reshape(mu_dims)

        obs = np.array(msg.obs)
        obs_dims = (msg.n, msg.rollout_len, msg.dPrimObs)
        obs = obs.reshape(obs_dims)

        prc = np.array(msg.prc)
        prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
        prc = prc.reshape(prc_dims)

        wt = msg.wt

        update = self.policy_opt.store(obs, mu, prc, wt, 'primitive')
        if update:
            self.weight_publisher.publish(self.policy_opt.serialize_weights(['primitive']))


    def primitive(self, req):
        task_distr, obj_distr, targ_distr = self.policy_opt.task_distr(np.array([req.prim_obs]))
        return PrimitiveResponse(task_distr.tolist(), obj_distr.tolist(), targ_distr.tolist())
