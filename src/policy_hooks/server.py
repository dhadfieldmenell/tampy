import pickle as pickle
from datetime import datetime
import numpy as np
import os
import pprint
import queue
import random
import sys
import time
from software_constants import *

from PIL import Image
from scipy.cluster.vq import kmeans2 as kmeans
import tensorflow as tf

from core.internal_repr.plan import Plan
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.search_node import *


LOG_DIR = 'experiment_logs/'

class Server(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        self._hyperparams = hyperparams
        self.config = hyperparams
        self.group_id = hyperparams['group_id']

        self.start_t = hyperparams['start_t']
        self.seed = int((1e2*time.time()) % 1000.)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.solver = hyperparams['mp_solver_type'](hyperparams)
        self.opt_smooth = hyperparams.get('opt_smooth', False)
        hyperparams['agent']['master_config'] = hyperparams
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.process_id = '{0}_{1}'.format(self.id, self.group_id)
        self.agent.solver = self.solver
        self.prob = self.agent.prob
        self.solver.agent = self.agent

        self.task_queue = hyperparams['task_queue']
        self.motion_queue = hyperparams['motion_queue']
        self.rollout_queue = hyperparams['rollout_queue']
        self.ll_queue = hyperparams['ll_queue']
        self.hl_queue = hyperparams['hl_queue']

        self.pol_cls = DummyPolicy
        self.opt_cls = DummyPolicyOpt

        self._n_plans = 0
        n_plans = hyperparams['policy_opt']['buffer_sizes']['n_plans']
        self.alg_map = hyperparams['alg_map']
        for alg in list(self.alg_map.values()):
            alg.set_conditions(len(self.agent.x0))

        self.task_list = self.agent.task_list
        self.pol_list = tuple(hyperparams['policy_list'])
        self.stopped = False
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[], 'tasks':[], 'use_mask':[]}
        self.last_log_t = time.time()

        for alg in list(self.alg_map.values()):
            alg.policy_opt = DummyPolicyOpt(self.update, self.prob)
        self.current_id = 0
        self.cur_step = 0
        self.adj_eta = False
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        self.init_policy_opt(hyperparams)
        self.agent.plans, self.agent.openrave_bodies, self.agent.env = self.agent.prob.get_plans(use_tf=True)
        for plan in list(self.agent.plans.values()):
            plan.state_inds = self.agent.state_inds
            plan.action_inds = self.agent.action_inds
            plan.dX = self.agent.dX
            plan.dU = self.agent.dU
            plan.symbolic_bound = self.agent.symbolic_bound
            plan.target_dim = self.agent.target_dim
            plan.target_inds = self.agent.target_inds
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.ff_data_file = LOG_DIR+hyperparams['weight_dir']+'/ff_samples_{0}_{1}.pkl'
        self.log_file = LOG_DIR + hyperparams['weight_dir'] + '/{0}_log.txt'.format(self.id)
        self.verbose_log_file = LOG_DIR + hyperparams['weight_dir'] + '/{0}_verbose.txt'.format(self.id)

    
    def init_policy_opt(self, hyperparams):
        hyperparams['policy_opt']['split_hl_loss'] = hyperparams['split_hl_loss']
        hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] # + '_trained'
        hyperparams['policy_opt']['scope'] = None
        hyperparams['policy_opt']['gpu_fraction'] = 1./32.
        hyperparams['policy_opt']['use_gpu'] = 1.
        hyperparams['policy_opt']['allow_growth'] = True
        self.policy_opt = hyperparams['policy_opt']['type'](hyperparams['policy_opt'],
                                                            hyperparams['dO'],
                                                            hyperparams['dU'],
                                                            hyperparams['dPrimObs'],
                                                            hyperparams['dValObs'],
                                                            hyperparams['prim_bounds'])
        for alg in list(self.alg_map.values()):
            alg.local_policy_opt = self.policy_opt
        self.weights_to_store = {}


    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def spawn_problem(self):
        x0, targets = self.new_problem()
        initial, goal = self.agent.get_hl_info(x0, targets)
        problem = list(self.agent.plans.values())[0].prob
        domain = list(self.agent.plans.values())[0].domain
        abs_prob = self.agent.hl_solver.translate_problem(problem, goal=goal, initial=initial)
        for pname, attr in self.agent.state_inds:
            p = problem.init_state.params[pname]
            if p.is_symbol(): continue
            getattr(p, attr)[:,0] = x0[self.agent.state_inds[pname, attr]]
        for targ, attr in self.agent.target_inds:
            if targ in problem.init_state.params:
                p = problem.init_state.params[targ]
                getattr(p, attr)[:,0] = targets[self.agent.target_inds[targ, attr]].copy()
        prefix = []
        hlnode = HLSearchNode(abs_prob,
                             domain,
                             problem,
                             priority=0,
                             prefix=prefix,
                             llnode=None,
                             x0=x0,
                             targets=targets,
                             label=self.id)
        return hlnode


    def new_problem(self):
        x0, targets = self.prob.get_random_initial_state_vec(self.config, None, self.agent.dX, self.agent.state_inds, 1)
        x0, targets = x0[0], targets[0]
        target_vec = np.zeros(self.agent.target_dim)
        for (tname, attr), inds in self.agent.target_inds.items():
            if attr != 'value': continue
            target_vec[inds] = targets[tname]
        return x0, target_vec



    def update(self, obs, mu, prc, wt, task, rollout_len=0, acts=[], ref_acts=[], terminal=[], aux=[]):
        assert(len(mu) == len(obs))

        prc[np.where(prc > 1e10)] = 1e10
        wt[np.where(wt > 1e10)] = 1e10
        prc[np.where(prc < -1e10)] = -1e10
        wt[np.where(wt < -1e10)] = -1e10
        mu[np.where(np.abs(mu) > 1e10)] = 0
        if np.any(np.isnan(obs)):
            print((obs, task, np.isnan(obs)))
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        obs[np.where(np.abs(obs) > 1e10)] = 0

        opts = self.agent.prob.get_prim_choices()
        msg = PolicyUpdate() if USE_ROS else DummyMSG()
        msg.obs = obs.flatten().tolist()
        msg.mu = mu.flatten().tolist()
        msg.prc = prc.flatten().tolist()
        msg.wt = wt.flatten().tolist()
        if len(acts):
            msg.acts = acts.flatten().tolist()
            msg.ref_acts = ref_acts.flatten().tolist()
            msg.terminal = terminal
        if len(aux):
            msg.aux = list(aux)
        msg.dO = self.agent.dO
        msg.dPrimObs = self.agent.dPrim
        msg.dOpts = len(list(self.agent.prob.get_prim_choices().keys()))
        msg.dValObs = self.agent.dVal
        msg.dAct = np.sum([len(opts[e]) for e in opts])
        msg.nActs = np.prod([len(opts[e]) for e in opts])
        msg.dU = mu.shape[-1]
        msg.n = len(mu)
        msg.rollout_len = mu.shape[1] if rollout_len < 1 else rollout_len
        msg.task = str(task)

        if task == 'primitive':
            q = self.hl_queue
        else:
            q = self.ll_queue

        self.push_queue(msg, q)


    def policy_call(self, x, obs, t, noise, task, opt_s=None):
        if noise is None: noise = np.zeros(self.agent.dU)
        if 'control' in self.policy_opt.task_map:
            alg_key = 'control'
        else:
            alg_key = task
        if self.policy_opt.task_map[alg_key]['policy'].scale is None:
            if opt_s is not None:
                return opt_s.get_U(t) + self.alg_map[task].cur[0].traj_distr.chol_pol_covar[t].T.dot(noise)
            t = min(t, self.alg_map[task].cur[0].traj_distr.T-1)
            return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
        return self.policy_opt.task_map[alg_key]['policy'].act(x.copy(), obs.copy(), t, noise)


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None):
        if self.adj_eta: eta *= self.agent.eta_scale
        distrs = self.policy_opt.task_distr(prim_obs, eta)
        if task is not None and t % self.check_prim_t:
            for i in range(len(distrs)):
                distrs[i] = np.zeros_like(distrs[i])
                distrs[i][task[i]] = 1.
            return distrs
        if not soft: return distrs
        out = []
        for d in distrs:
            p = d / np.sum(d)
            ind = np.random.choice(list(range(len(d))), p=p)
            d[ind] += 1e1
            d /= np.sum(d)
            out.append(d)
        return out


    def store_weights(self, msg):
        self.weights_to_store[msg.scope] = msg.data


    def update_weights(self):
        scopes = list(self.weights_to_store.keys())
        for scope in scopes:
            save = self.id.endswith('0')
            data = self.weights_to_store[scope]
            self.weights_to_store[scope] = None
            if data is not None:
                self.policy_opt.deserialize_weights(data, save=save)


    def pop_queue(self, q):
        try:
            node = q.get_nowait()
        except queue.Empty:
            node = None

        return node


    def push_queue(self, prob, q):
        if q.full():
            try:
                node = q.get_nowait()
            except queue.Empty:
                node = None

            if node is not None and \
               hasattr(node, 'heuristic') and \
               node.heuristic() < prob.heuristic():
                prob = node

        try:
            q.put_nowait(prob)
        except queue.Full:
            pass


    def set_policies(self):
        if self.policy_opt.share_buffers:
            self.policy_opt.read_shared_weights()
        chol_pol_covar = {}
        for task in self.agent.task_list:
            if task not in self.policy_opt.valid_scopes:
                task_name = 'control'
            else:
                task_name = task

            if self.policy_opt.task_map[task_name]['policy'].scale is None:
                chol_pol_covar[task] = np.eye(self.agent.dU) # self.alg_map[task].cur[0].traj_distr.chol_pol_covar
            else:
                chol_pol_covar[task] = self.policy_opt.task_map[task_name]['policy'].chol_pol_covar

        rollout_policies = {task: DummyPolicy(task,
                                              self.policy_call,
                                              chol_pol_covar=chol_pol_covar[task],
                                              scale=self.policy_opt.task_map[task if task in self.policy_opt.valid_scopes else 'control']['policy'].scale)
                                  for task in self.agent.task_list}
        self.agent.policies = rollout_policies


    def run_hl_update(self):
        ### Look for saved successful HL rollout paths and send them to update the HL options policy
        ref_paths = []
        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)
            ref_paths.append(path)
        self.agent.clear_task_paths()
        
        self.update_primitive(path_samples)
        if self._hyperparams.get('save_expert', False): self.update_expert_demos(ref_paths)


    def run(self):
        raise NotImplementedError()


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(list(self.agent.prob.get_prim_choices().keys()))
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dOpts)), np.zeros((0))
        tgt_aux = np.zeros((0))

        if len(samples):
            lab = samples[0].source_label
            if lab in self.policy_opt.buf_sizes:
                with self.policy_opt.buf_sizes[lab].get_lock():
                    self.policy_opt.buf_sizes[lab].value += 1
                samples[0].source_label = ''
        for ind, sample in enumerate(samples):
            mu = np.concatenate([sample.get(enum) for enum in self.config['prim_out_include']], axis=-1)
            tgt_mu = np.concatenate((tgt_mu, mu))
            st, et = 0, sample.T # st, et = sample.step * sample.T, (sample.step + 1) * sample.T
            #aux = np.ones(sample.T)
            #if sample.task_start: aux[0] = 0.
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            wt = np.array([sample.prim_use_ts[t] * self.prim_decay**t for t in range(sample.T)])
            if sample.task_start and ind > 0 and sample.opt_strength > 0.999: wt[0] = self.prim_first_wt
            if sample.opt_strength < 1-1e-3: wt[:] *= self.explore_wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            if np.any(np.isnan(obs)):
                print((obs, sample.task, 'SAMPLE'))
            obs_data = np.concatenate((obs_data, obs))
            prc = np.concatenate([self.agent.get_mask(sample, enum) for enum in self.config['prim_out_include']], axis=-1) # np.tile(np.eye(dP), (sample.T,1,1))
            if not self.config['hl_mask']:
                prc[:] = 1.

            tgt_prc = np.concatenate((tgt_prc, prc))
        if len(tgt_mu):
            # print('Sending update to primitive net')
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 1, aux=tgt_aux)


    def get_path_data(self, path, n_fixed=0, verbose=False):
        data = []
        for sample in path:
            X = [{(pname, attr): sample.get_X(t=t)[self.agent.state_inds[pname, attr]].round(3) for pname, attr in self.agent.state_inds if self.agent.state_inds[pname, attr][-1] < self.agent.symbolic_bound} for t in range(sample.T)]
            if hasattr(sample, 'col_ts'):
                U = [{(pname, attr): (sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4), sample.col_ts[t]) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            else:
                U = [{(pname, attr): sample.get_U(t=t)[self.agent.action_inds[pname, attr]].round(4) for pname, attr in self.agent.action_inds} for t in range(sample.T)]
            info = {'X': X, 'task': sample.task, 'time_from_start': time.time() - self.start_t, 'value': 1.-sample.task_cost, 'fixed_samples': n_fixed, 'root_state': self.agent.x0[0], 'opt_strength': sample.opt_strength if hasattr(sample, 'opt_strength') else 'N/A'}
            if verbose:
                info['obs'] = sample.get_obs().round(3)
                # info['prim_obs'] = sample.get_prim_obs().round(3)
                info['targets'] = {tname: sample.targets[self.agent.target_inds[tname, attr]] for tname, attr in self.agent.target_inds}
                info['opt_success'] = sample.opt_suc
                info['tasks'] = sample.get(FACTOREDTASK_ENUM)
                info['end_state'] = sample.end_state
                # info['prim_obs'] = sample.get_prim_obs().round(4)
            data.append(info)
        return data


    def log_path(self, path, n_fixed=0):
        if self.log_file is None: return
        with open(self.log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed)
            pp_info = pprint.pformat(info, depth=120, width=120)
            f.write(pp_info)
            f.write('\n')

        with open(self.verbose_log_file, 'a+') as f:
            f.write('\n\n')
            info = self.get_path_data(path, n_fixed, True)
            pp_info = pprint.pformat(info, depth=120, width=120)
            f.write(pp_info)
            f.write('\n')


class DummyPolicy:
    def __init__(self, task, policy_call, opt_sample=None, chol_pol_covar=None, scale=None):
        self.task = task
        self.policy_call = policy_call
        self.opt_sample = opt_sample
        self.chol_pol_covar = chol_pol_covar
        self.scale = scale

    def act(self, x, obs, t, noise):
        U = self.policy_call(x, obs, t, noise, self.task, None)
        if np.any(np.isnan(x)):
            #raise Exception('Nans in policy call state.')
            print('Nans in policy call state.')
            U = np.zeros_like(U)
        if np.any(np.isnan(obs)):
            # raise Exception('Nans in policy call obs.')
            print(obs)
            print('Nans in policy call obs.')
            U = np.zeros_like(U)
        if np.any(np.isnan(U)):
            # raise Exception('Nans in policy call action.')
            print('Nans in policy call action.')
            U = np.zeros_like(U)
        return U


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


