import copy
import sys
import traceback

import numpy as np
import tensorflow as tf

import rospy
from std_msgs.msg import *


from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.namo.sorting_prob_2 import *
from policy_hooks.namo.namo_policy_solver import NAMOPolicySolver

from tamp_ros.msg import *
from tamp_ros.srv import *


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.prob = prob

class NAMOMotionPlanServer(object):
    def __init__(self, hyperparams):
        self.id =  hyperparams['id']
        rospy.init_node('namo_mp_solver_'+str(self.id))
        self.solver = NAMOPolicySolver(hyperparams)
        self.task_list = hyperparams['task_list']
        plans = {}
        env = None
        openrave_bodies = {}
        for task in self.task_list:
            for c in range(hyperparams['num_objs']):
                plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)] = get_plan_for_task(task, ['{0}{1}'.format(hyperparams['obj_type'], c), '{0}{1}_end_target'.format(hyperparams['obj_type'], c)], hyperparams['num_objs'], env, openrave_bodies)
                if env is None:
                    env = plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)].env
                    for param in plans[task, '{0}{1}'.format(hyperparams['obj_type'], c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body
        # self.policy_opt = hyperparams['policy_opt']
        # self.solver.policy_opt = self.policy_opt
        self.solver.policy_opt = DummyPolicyOpt(self.prob)
        self.agent = hyperparams['agent']
        self.solver.agent = self.agent
        self.weight_dir = hyperparams['weight_dir']
        self.solver.policy_inf_fs = {}
        for task in self.solver.agent.task_list:
            self.solver.policy_inf_fs[task] = lambda s: self.prob(s, task)

        self.mp_service = rospy.Service('motion_planner_'+str(self.id), MotionPlan, self.serve_motion_plan)
        self.stopped = False
        self.mp_publishers = {i: rospy.Publisher('motion_plan_result_'+str(i), MotionPlanResult, queue_size=50) for i in range(hyperparams['n_rollout_servers'])}
        self.async_planner = rospy.Subscriber('motion_plan_prob', MotionPlanProblem, self.publish_motion_plan)
        self.weight_subscriber = rospy.Subscriber('tf_weights', String, self.store_weights, queue_size=1, buff_size=2**20)
        self.stop = rospy.Subscriber('terminate', String, self.end)

        self.prob_proxy = rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True)
        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] + '_trained'
            hyperparams['policy_opt']['scope'] = None
            self.policy_opt = hyperparams['policy_opt']['type'](
                hyperparams['policy_opt'], 
                hyperparams['dO'],
                hyperparams['dU'],
                hyperparams['dObj'],
                hyperparams['dTarg'],
                hyperparams['dPrimObs']
            )


    def run(self):
        rospy.spin()
        # while not self.stopped:
        #     rospy.sleep(0.01)


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def store_weights(self, msg):
        if self.use_local:
            self.policy_opt.deserialize_weights(msg.data)


    def prob(self, obs, task):
        if self.use_local:
            return self.policy_opt.prob(obs, task)

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
        return np.array([resp.mu[i].data for i in range(len(resp.mu))]), np.array([resp.sigma[i].data for i in range(len(resp.sigma))]), [], []


    def publish_motion_plan(self, msg):
        if msg.solver_id != self.id: return
        print 'Server {0} solving motion plan for rollout server {1}.'.format(self.id, msg.server_id)
        state = np.array(msg.state)
        task = msg.task
        cond = msg.cond
        mean = np.array([msg.traj_mean[i].data for i in range(len(msg.traj_mean))])
        targets = [msg.obj, msg.targ]
        out, failed, success = self.sample_optimal_trajectory(state, task, cond, mean, targets)
        failed = str(failed)
        resp = MotionPlanResult()
        resp.traj = []
        for t in range(len(out)):
            next_line = Float32MultiArray()
            next_line.data = out[t]
            resp.traj.append(next_line)
        resp.failed = failed
        resp.success = success
        resp.plan_id = msg.prob_id
        resp.cond = msg.cond
        resp.task = msg.task
        resp.obj = msg.obj
        resp.targ = msg.targ
        self.mp_publishers[msg.server_id].publish(resp)


    def serve_motion_plan(self, req):
        state = req.state
        task = req.task
        cond = req.condition
        mean = np.array([req.traj_mean[i].data for i in range(len(req.traj_mean))])
        targets = [reg.obj, req.targ]
        out, failed, success = self.sample_optimal_trajectory(state, task, cond, mean, targets)
        failed = str(failed)
        resp = MotionPlanResponse()
        resp.traj = out
        resp.failed = failed
        resp.success = success
        return resp


    def update_weight(self, msg):
        scope = msg.scope
        weight_dir = self.weight_dir
        variables = tf.get_colleciton(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(variables)
        saver.restore(self.policy_opt.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')


    def sample_optimal_trajectory(self, state, task, condition, traj_mean=[], fixed_targets=[]):
        exclude_targets = []
        success = False

        targets = fixed_targets
        obj = targets[0]
        targ = targets[1]

        failed_preds = []
        iteration = 0
        while not success:
            iteration += 1

            plan = self.agent.plans[task, targets[0]] 
            targets[0] = plan.params[targets[0]]
            targets[1] = plan.params[targets[1]]
            obj, targ = targets
            set_params_attrs(plan.params, plan.state_inds, state, 0)

            for param_name in plan.params:
                param = plan.params[param_name]
                if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                    plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

            for target in self.agent.targets[condition]:
                plan.params[target].value[:,0] = self.agent.targets[condition][target]

            if targ.name in self.agent.targets[condition]:
                plan.params['{0}_end_target'.format(obj.name)].value[:,0] = self.agent.targets[condition][targ.name]

            if task == 'grasp':
                plan.params[targ.name].value[:,0] = plan.params[obj.name].pose[:,0]
            
            plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
            dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
            if task == 'putdown':
                plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]
            if task == 'grasp':
                plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist+0.2]
            # self.env.SetViewer('qtcoin')
            # success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean, task=(self.task_list.index(task), self.obj_list.index(obj.name), self.targ_list.index(targ.name)))
            try:
                self.solver.save_free(plan)
                success = self.solver._backtrack_solve(plan, n_resamples=3, traj_mean=traj_mean, task=(self.agent.task_list.index(task), self.agent.obj_list.index(obj.name), self.agent.targ_list.index(targ.name)))
                # viewer = OpenRAVEViewer._viewer if OpenRAVEViewer._viewer is not None else OpenRAVEViewer(plan.env)
                # if task == 'putdown':
                #     import ipdb; ipdb.set_trace()
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
            except Exception as e:
                traceback.print_exception(*sys.exc_info())
                self.solver.restore_free(plan)
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
                success = False

            failed_preds = []
            for action in plan.actions:
                try:
                    failed_preds += [(pred.get_type(), targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3, active_ts=action.active_timesteps)]
                except:
                    pass
            exclude_targets.append(targets[0].name)

            if len(fixed_targets):
                break

        if len(failed_preds):
            success = False
        else:
            success = True

        output_traj = np.zeros((plan.horizon, self.agent.dX))
        for t in range(plan.horizon):
            fill_vector(plan.params, self.agent.state_inds, output_traj[t], t)

        return output_traj[:,:self.agent.symbolic_bound], failed_preds, success
