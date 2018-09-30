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
        self.solver = NAMOPolicySolver(hyperparams)
        plans = {}
        env = None
        openrave_bodies = {}
        for task in self.task_list:
            for c in range(num_cans):
                plans[task, 'can{0}'.format(c)] = get_plan_for_task(task, ['can{0}'.format(c), 'can{0}_end_target'.format(c)], num_cans, env, openrave_bodies)
                if env is None:
                    env = plans[task, 'can{0}'.format(c)].env
                    for param in plans[task, 'can{0}'.format(c)].params.values():
                        if not param.is_symbol():
                            openrave_bodies[param.name] = param.openrave_body
        # self.policy_opt = hyperparams['policy_opt']
        # self.solver.policy_opt = self.policy_opt
        self.solver.policy_opt = DummyPolicyOpt(self.prob)
        self.solver.agent = hyperparams['agent']
        self.weight_dir = hyperparams['weight_dir']
        self.solver.policy_inf_fs = {}
        for task in self.solver.agent.task_list:
            self.solver.policy_inf_fs[task] = lambda s: self.prob(s, task)

        self.mp_service = rospy.Service('motion_planner_'+self.id, MotionPlan, self.serve_motion_plan)
        self.stopped = False
        self.mp_publishers = {i: rospy.Publisher('motion_plan_result_'+i, MotionPlanResult, queue_size=50) for i in range(hyperparams['n_rollout_servs'])}
        self.async_planner = rospy.Subscriber('motion_plan_prob', MotionPlanProblem, self.publish_motion_plan)
        self.stop = rospy.Subscriber('terminate', str, self.end)


    def run(self):
        rospy.spin()
        # while not self.stopped:
        #     rospy.sleep(0.01)


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def prob(self, sample, task):
        rospy.wait_for_service(task+'_policy_prob', timeout=10)
        proxy_call = rospy.ServiceProxy(task+'_policy_prob', PolicyProb)
        obs = []
        s_obs = sample.get_obs()
        for i in range(len(s_obs)):
            next_line = Float32MultiArray()
            next_line.data = s_obs[i]
            obs.append(next_line)
        req = PolicyProbRequest()
        req.obs = obs
        req.task = task
        resp = proxy_call(req)
        return np.array([resp.mu[i].data for i in range(len(resp.mu))]), np.array([resp.sigma[i].data for i in range(len(resp.sigma))]), [], []


    def publish_motion_plan(self, msg):
        if msg.solver_id != self.id: return
        state = msg.state
        task = msg.task
        cond = msg.cond
        mean = [msg.traj_mean[i].data for i in range(len(msg.traj_mean))]
        targets = msg.obj, msg.targ
        out, failed, success = self.sample_optimal_trajectory(state, task, cond, mean, targets)
        failed = str(failed)
        resp = MotionPlanResult()
        resp.traj = out
        resp.failed = failed
        resp.success = success
        resp.plan_id = msg.prob_id
        self.mp_publishers[msg.server_id].publish(resp)


    def serve_motion_plan(self, req):
        state = req.state
        task = req.task
        cond = req.condition
        mean = [req.traj_mean[i].data for i in range(len(req.traj_mean))]
        targets = reg.obj, req.targ
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
        obj = fixed_targets[0]
        targ = fixed_targets[1]

        failed_preds = []
        iteration = 0
        while not success:
            iteration += 1

            plan = self.plans[task, targets[0]] 
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
