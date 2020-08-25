""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')
import numpy as np
import tensorflow as tf

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
from datetime import datetime
import threading
import pprint
import time
import traceback

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

from policy_hooks.mcts import MCTS
from policy_hooks.state_mcts import StateMCTS
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *


class LocalControl:
    def __init__(self, policy_func):
        self.act = policy_func

class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self.agent = config['agent']['type'](config['agent'])
        self._train_idx = list(range(config['num_conds']))
        self._hyperparams=config
        self._test_idx = self._train_idx
        self.iter_count = 0

        self._data_files_dir = config['common']['data_files_dir']

        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        self.replace_conds = self._hyperparams['stochastic_conditions']
        self.task_list = config['task_list']
        self.fail_value = config['fail_value']
        self.alg_map = {}
        policy_opt = None
        self.task_durations = config['task_durations']
        for task in self.task_list:
            if task not in self.task_durations: continue
            config['algorithm'][task]['policy_opt']['prev'] = policy_opt
            config['algorithm'][task]['agent'] = self.agent
            config['algorithm'][task]['init_traj_distr']['T'] = self.task_durations[task]
            self.alg_map[task] = config['algorithm'][task]['type'](config['algorithm'][task], task)
            policy_opt = self.alg_map[task].policy_opt
            self.alg_map[task].set_conditions(len(self.agent.x0))
            self.alg_map[task].agent = self.agent

        self.policy_opt = policy_opt
        self.weight_file = self._hyperparams['weight_file'] if 'weight_file' in self._hyperparams else None
        self.saver = tf.train.Saver()
        if self.weight_file is not None:
            # self.saver.restore(policy_opt.sess, self.weight_file)
            self.pretrain = False
        else:
            self.pretrain = True
        self.traj_opt_steps = self._hyperparams['traj_opt_steps']
        self.num_samples = self._hyperparams['num_samples']

        self.log_file = str(datetime.now()) + '_namo.txt'

        self.rollout_policies = {task: self.policy_opt.task_map[task]['policy'] for task in self.task_list}
        self.mcts = []
        self.state_mcts = StateMCTS(
                                  self.task_list,
                                  self.policy_opt.task_distr,
                                  self._hyperparams['plan_f'],
                                  self._hyperparams['cost_f'],
                                  self._hyperparams['goal_f'],
                                  self._hyperparams['target_f'],
                                  self._hyperparams['encode_f'],
                                  self.policy_opt.value,
                                  self.rollout_policies,
                                  self.policy_opt.distilled_policy,
                                  self.agent,
                                  self._hyperparams['branching_factor'],
                                  self._hyperparams['num_samples'],
                                  self._hyperparams['num_distilled_samples'],
                                  soft_decision=1.0,
                                  C=2.,
                                  max_depth=self._hyperparams['max_tree_depth'],
                                  always_opt=False
                          )

        for condition in range(len(self.agent.x0)):
            self.mcts.append(MCTS(
                                  self.task_list,
                                  self.policy_opt.task_distr,
                                  self._hyperparams['plan_f'],
                                  self._hyperparams['cost_f'],
                                  self._hyperparams['goal_f'],
                                  self._hyperparams['target_f'],
                                  self._hyperparams['encode_f'],
                                  self.policy_opt.value,
                                  self.rollout_policies,
                                  self.policy_opt.distilled_policy,
                                  condition,
                                  self.agent,
                                  self._hyperparams['branching_factor'],
                                  self._hyperparams['num_samples'],
                                  self._hyperparams['num_distilled_samples'],
                                  soft_decision=1.0,
                                  C=2.,
                                  max_depth=self._hyperparams['max_tree_depth'],
                                  always_opt=False
                                  ))

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        try:
            # log_file = open("avg_cost_log.txt", "a+")
            # log_file.write("\n{0}\n".format(datetime.now().isoformat()))
            # log_file.close()
            itr_start = self._initialize(itr_load)

            init_samples = []
            if self.pretrain:
                for pretrain_step in range(3):
                    self.agent.replace_conditions(len(self.agent.x0), keep=(1., 1.))
                    hl_plans = [[] for _ in range(len(self._train_idx))]
                    paths = []

                    for cond in self._train_idx:
                        failed = []
                        new_failed = []
                        stop = False
                        attempt = 0
                        cur_sample = None
                        cur_path = []
                        cur_state = self.agent.x0[cond].copy()
                        opt_hl_plan = []
                        try:
                            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
                        except:
                            hl_plan = []
                        last_reset = 0
                        while not stop and attempt < 30:
                            last_reset += 1
                            old_cur_state = cur_state
                            for step in hl_plan:
                                targets = [list(self.agent.plans.values())[0].params[p_name] for p_name in step[1]]
                                plan = self._hyperparams['plan_f'](step[0], targets)
                                if len(targets) < 2:
                                    targets.append(plan.params['{0}_end_target'.format(targets[0].name)])
                                next_sample, new_failed, success = self.agent.sample_optimal_trajectory(cur_state, step[0], cond, fixed_targets=targets)
                                init_samples.append(next_sample)
                                next_sample.success = FAIL_LABEL
                                if not success:
                                    if last_reset > 5:
                                        failed = []
                                        last_reset = 0
                                    next_sample.task_cost = self.fail_value
                                    next_sample.success = FAIL_LABEL
                                    if not len(new_failed):
                                        stop = True
                                        if not len(hl_plan):
                                            print('NAMO COULD NOT FIND HL PLAN FOR {0}'.format(cur_state))
                                    else:
                                        failed.extend(new_failed)
                                        try:
                                            hl_plan = self.agent.get_hl_plan(cur_state, cond, failed)
                                        except:
                                            hl_plan = []
                                        attempt += 1
                                    break

                                cur_path.append(next_sample)
                                cur_sample = next_sample
                                cur_state = cur_sample.get_X(t=cur_sample.T-1).copy()
                                opt_hl_plan.append(step)
                            if self._hyperparams['goal_f'](cur_state, self.agent.targets[cond], list(self.agent.plans.values())[0]) == 0:
                                self.agent.add_task_paths([cur_path])
                                hl_plans[cond] = opt_hl_plan
                                for sample in cur_path:
                                    sample.success = SUCCESS_LABEL
                                break

                            attempt += 1

                        paths.append(cur_path)

                    for path in paths:
                        if not len(path): continue
                        cur_sample = path[-1]
                        opt_val = self._hyperparams['goal_f'](cur_sample.get_X(t=cur_sample.T-1), self.agent.targets[cond], list(self.agent.plans.values())[0])
                        for sample in path:
                            sample.task_cost = opt_val

                    print(hl_plans)

                for task in self.alg_map:
                    n_samples = len(self.agent.optimal_samples[task])
                    for i in range(5):
                        print('\nIterating on initial samples (iter {0})'.format(i))
                        policy = self.rollout_policies[task]
                        if policy.scale is None:
                            print('Using lin gauss')
                            policy = self.alg_map[task].cur[0].traj_distr
                        samples = self.agent.resample(self.agent.optimal_samples[task][-n_samples:], policy, self.num_samples)
                        self.alg_map[task].iteration(samples, self.agent.optimal_samples[task])
                        self.agent.clear_samples()
                        self.agent.add_sample_batch(samples, task)

                path_samples = []
                for path in self.agent.get_task_paths():
                    path_samples.extend(path)

                self.update_primitives(path_samples)
                self.update_value_network(init_samples, first_ts_only=True)
                self.saver.save(self.policy_opt.sess, 'tf_saved/'+str(datetime.now())+'_namo_{0}.ckpt'.format(self.agent.num_cans))
                print(hl_plans)

            # import ipdb; ipdb.set_trace()
            for itr in range(itr_start, self._hyperparams['iterations']):
                print('\n\nITERATION ', itr)
                paths = []
                for cond in self._train_idx:
                    rollout_policies = {}
                    use_distilled = False
                    for task in self.task_list:
                        self.alg_map[task].set_conditions(1)
                        if self.rollout_policies[task].scale is not None:
                            rollout_policies[task] = self.rollout_policies[task]
                            use_distilled = True
                        else:
                            rollout_policies[task] = self.alg_map[task].cur[0].traj_distr

                    val = self.mcts[cond].run(self.agent.x0[cond], self._hyperparams['num_rollouts'], use_distilled, new_policies=rollout_policies, debug=True)
                    # val = self.state_mcts.run(self.agent.x0[cond], cond, self._hyperparams['num_rollouts'], use_distilled, new_policies=rollout_policies, debug=True)

                traj_sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}

                self.agent.clear_samples(keep_prob=0.2, keep_opt_prob=0.5)

                path_samples = []
                for path in self.agent.get_task_paths():
                    path_samples.extend(path)

                # self.agent.clear_task_paths(keep_prob=0.5)
                self._take_iteration(itr, traj_sample_lists, path_samples, traj_opt_steps=self.traj_opt_steps)
                self.agent.reset_sample_refs()

                next_x = self.agent.x0[0].copy()
                self.agent.reset_hist()
                for _ in range(5):
                    task, obj_name, targ_name = self.predict_condition(0, next_x)
                    print(task, obj_name, targ_name)
                    next_x = self.print_cond(task, obj_name, targ_name, state=next_x, reset_hist=False)
                # self.mcts[0].print_run(self.agent.x0[0], use_distilled=False)

            import ipdb; ipdb.set_trace()
            for itr in range(itr_start, self._hyperparams['iterations']):
                print('\n\nITERATION ', itr)
                if itr % 3 == 0:
                    self.agent.replace_conditions(len(self._train_idx), (0.5, 0.3))
                paths = []
                for cond in self._train_idx:
                    rollout_policies = {}
                    use_distilled = False
                    for task in self.task_list:
                        if self.rollout_policies[task].scale is not None:
                            rollout_policies[task] = self.rollout_policies[task]
                            use_distilled = True
                        else:
                            rollout_policies[task] = self.alg_map[task].cur[0].traj_distr

                    val = self.mcts[cond].run(self.agent.x0[cond], self._hyperparams['num_rollouts'], use_distilled, new_policies=rollout_policies, debug=True)

                traj_sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}

                self.agent.clear_samples(keep_prob=0.1)

                path_samples = []
                for path in self.agent.get_task_paths():
                    path_samples.extend(path)

                # self.agent.clear_task_paths(keep_prob=0.5)
                self._take_iteration(itr, traj_sample_lists, path_samples, traj_opt_steps=self.traj_opt_steps)
                self.agent.reset_sample_refs()

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            raise e
        # finally:
            # self._end()

        import ipdb; ipdb.set_trace()
        self.policy_opt.sess.close()

    def get_value_samples(self):
        rollouts = []
        for cond in self._train_idx:
            x0 = self.agent.x0[cond].copy()
            old_n_samples = self.state_mcts.num_samples
            self.state_mcts.num_samples = 1
            path = self.state_mcts.simulate(x0, False)
            rollouts.extend(path)
        return rollouts

    def update_primitives(self, samples):
        dP, dO = len(self.task_list), list(self.alg_map.values())[0].dPrimObs
        dObj, dTarg = list(self.alg_map.values())[0].dObj, list(self.alg_map.values())[0].dTarg
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
            self.policy_opt.update_primitive_filter(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def update_value_network(self, samples, first_ts_only=False):
        dV, dO = 2, list(self.alg_map.values())[0].dO

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
            self.policy_opt.update_value(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def update_distilled(self, optimal_samples=[]):
        """ Compute the new distilled policy. """
        dU, dO = self.agent.dU, list(self.alg_map.values())[0].dPrimObs
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dU))
        tgt_prc, tgt_wt = np.zeros((0, dU, dU)), np.zeros((0,))

        # Optimize global polciies with optimal samples as well
        for sample in optimal_samples:
            T = sample.T
            data_len = int(list(self.alg_map.values())[0].sample_ts_prob * T)
            mu = np.zeros((0, dU))
            prc = np.zeros((0, dU, dU))
            wt = np.zeros((0))
            obs = np.zeros((0, dO))

            ts = np.random.choice(range(T), data_len, replace=False)
            ts.sort()
            for t in range(data_len):
                prc[t] = np.eye(dU)
                wt[t] = self._hyperparams['opt_wt']

            for i in range(data_len):
                t = ts[i]
                mu = np.concatenate([mu, sample.get_U(t=t)])
                obs = np.concatenate([obs, sample.get_prim_obs(t=t)])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, obs))

        for task in self.alg_map:
            alg = self.alg_map[task]
            for m in range(len(alg.cur)):
                samples = alg.cur[m].sample_list
                if samples is None or not hasattr(alg, 'new_traj_distr'):
                    print("No samples for {0} in distilled update.\n".format(task))
                    break
                T = samples[0].T
                X = samples.get_X()
                N = len(samples)
                traj, pol_info = alg.new_traj_distr[m], alg.cur[m].pol_info
                data_len = int(alg.sample_ts_prob * T)
                mu = np.zeros((0, dU))
                prc = np.zeros((0, dU, dU))
                wt = np.zeros((0,))
                obs = np.zeros((0, dO))
                full_obs = np.array([sample.get_prim_obs() for sample in samples])

                ts = np.random.choice(range(T), data_len, replace=False)
                ts.sort()
                # Get time-indexed actions.
                for j in range(data_len):
                    t = ts[j]
                    # Compute actions along this trajectory.
                    prc = np.concatenate([prc, np.tile(traj.inv_pol_covar[t, :, :],
                                                       [N, 1, 1])])
                    for i in range(N):
                        mu = np.concatenate([mu, [traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :]]])
                    wt = np.concatenate([wt, pol_info.pol_wt[t] * np.ones((N,))])
                    obs = np.concatenate([obs, full_obs[:, t, :]])
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.policy_opt.update_distilled(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def save_weights(self):
        for task in self.task_list + ['value', 'primitive']:
            variables = tf.get_colleciton(tf.GraphKeys.GLOBAL_VARIABLES, scope=task)
            saver = tf.train.Saver(variables)
            saver.save(self.policy_opt.sess, 'tf_saved/namo/{0}.ckpt'.format(task))

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print(("Error: cannot find '%s.'" % algorithm_file))
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print(("Error: cannot find '%s.'" % algorithm_file))
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self._hyperparams['sample_on_policy'] \
                and self.iter_count > 0:
            pol = self.policy_opt.task_map
            on_policy = True
        else:
            pol = {}
            for task in self.task_list:
                pol[task] = {}
                if task not in self.task_durations: continue
                def act(x, o, t, noisy):
                    cur_t = 0
                    cur_task = None
                    for next_t, next_task in self.agent.task_breaks[cond]:
                        if cur_t <= t and t < next_t:
                            U = self.alg_map[next_task].cur[cond][cur_t].traj_distr.act(x, o, t-cur_t, noisy)
                            return U

                        cur_t = next_t
                        cur_task = next_task

                    U = self.alg_map[cur_task].cur[cond][cur_t].traj_distr.act(x, o, t-cur_t, noisy)
                    return U

                pol[task]['policy'] = LocalControl(act)
            on_policy = False

        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond, use_base_t=False,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond, use_base_t=False,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration(self, itr, sample_lists, paths, traj_opt_steps=1):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        # val_samples = self.get_value_samples()
        for step in range(traj_opt_steps):
            for task in self.alg_map:
                if len(sample_lists[task]) or len(self.agent.optimal_samples[task]):
                    try:
                        sample_lists[task] = self.alg_map[task].iteration(sample_lists[task], self.agent.optimal_samples[task], reset=not step)
                        if len(sample_lists[task]) and step < traj_opt_steps - 1:
                            sample_lists[task] = self.agent.resample(sample_lists[task], self.rollout_policies[task], self._hyperparams['num_samples'])
                        self.agent._samples[task] = sample_lists[task]
                    except:
                        traceback.print_exception(*sys.exc_info())

            self.agent.reset_sample_refs()

        self.update_primitives(paths)
        # self.update_distilled()

        all_samples = []
        for task in self.task_list:
            sample_lists = self.agent.get_samples(task)
            for slist in sample_lists:
                all_samples.extend(slist._samples)

        self.update_value_network(all_samples)
        self.agent.reset_sample_refs()
        for task in self.alg_map:
            if len(self.alg_map[task].cur):
                self.alg_map[task]._advance_iteration_variables()
        if self.gui:
            self.gui.stop_display_calculating()

        self.iter_count += 1

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                save_global=True, verbose=verbose, save=False, noisy=False)
        return [SampleList(samples) for samples in pol_samples]

    def print_cond(self, task, obj, targ, state=None, reset_hist=True, cond=0, noisy=False):
        if reset_hist:
            self.agent.reset_hist()

        state = self.agent.x0[cond] if state is None else state

        sample = self.agent.sample_task(self.rollout_policies[task], cond, state, [task, obj, targ], noisy=noisy)
        print(sample.get_X())
        return sample.get_X(t=sample.T-1)

    def predict_condition(self, cond, state=None):
        sample = Sample(self.agent)
        state = self.agent.x0[cond].copy() if state is None else state
        sample.set(STATE_ENUM, self.agent.x0[cond].copy(), t=0)
        sample.set(TARGETS_ENUM, self.agent.target_vecs[cond].copy(), t=0)
        distr = self.policy_opt.task_distr([sample.get_prim_obs(t=sample.T-1)])
        task = self.task_list[np.argmax(distr[0])]
        obj = self.agent.obj_list[np.argmax(distr[1])]
        targ = self.agent.targ_list[np.argmax(distr[2])]
        return task, obj, targ

    def save_rollout(self, reset_hist=True, cond=0, noisy=False):
        if reset_hist:
            self.agent.reset_hist()

        file_time = str(datetime.now())
        with open('experiments/' + file_time + '.txt', 'w+') as f:
            f.write(file_time+'\n\n')

        state = self.agent.x0[cond]
        self.agent.reset_hist()
        for i in range(8):
            sample = Sample(self.agent)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.agent.traj_hist).flatten(), 0)
            sample.set(TARGETS_ENUM, self.agent.target_vecs[cond].copy(), 0)
            task_distr, obj_distr, targ_distr = self.policy_opt.task_distr([sample.get_prim_obs(t=0)])
            task_ind = np.argmax(task_distr)
            task = self.task_list[task_ind]
            obj = self.agent.obj_list[np.argmax(obj_distr)]
            targ = self.agent.targ_list[np.argmax(targ_distr)]
            # targets = self._hyperparams['target_f'](self.agent.plans.values()[0], state, task, self.agent.targets[0])

            sample = self.agent.sample_task(self.rollout_policies[task], cond, state, [task, obj, targ], noisy=noisy)
            state = sample.get_X(t=-1)
            with open('experiments/' + file_time + '.txt', 'a+') as f:
                f.write('STEP {0}: {1}\n'.format(i, task))
                f.write('TARGETS: {0} {1}\n'.format(obj, targ))
                f.write(str(sample.get_X()))
                f.write('\n\n')

        with open('experiments/' + file_time + '.txt', 'a+') as f:
            f.write(pprint.pformat(self._hyperparams, width=1))
            f.write('\n\n\n')
            f.write(str(list(self._hyperparams['algorithm'].values())[0]['cost']['type']))
            f.write('\n\n')

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        # self.data_logger.pickle(
        #     self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
        #     copy.copy(self.algorithm)
        # )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
