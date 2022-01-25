import pickle as pickle
from datetime import datetime
import numpy as np
import os, psutil
import pprint
import queue
import random
import sys
import time
from software_constants import *

from PIL import Image
import pybullet as P

from core.internal_repr.plan import Plan
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.save_video import save_video
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

        self.render = hyperparams.get('load_render', False)
        self.weight_dir = self.config['weight_dir']
        self.exp_id = self.weight_dir.split('/')[-1]
        label = self.config['label_server']
        self.classify_labels = self.config['classify_labels']
        if self.config['weight_dir'].find('sawyer') >= 0 and self.render:
            self.render = self.id.find('test') >= 0 and np.random.uniform() < 0.1
            hyperparams['agent']['master_config']['load_render'] = self.render

        n_gpu = hyperparams['n_gpu']
        if n_gpu == 0:
            gpus = -1
        elif n_gpu == 1:
            gpu = 0
        else:
            gpus = np.random.choice(range(n_gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = "{0}".format(gpus)

        self.init_policy_opt(hyperparams)
        hyperparams['agent']['master_config'] = hyperparams
        try:
            P.disconnect()
        except:
            print('No need to disconnect pybullet')
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.process_id = '{0}_{1}'.format(self.id, self.group_id)
        self.map_cont_discr_tasks()
        self.prob = self.agent.prob
        
        if self.render:
            self.cur_vid_id = 0
            if not os.path.isdir(LOG_DIR+hyperparams['weight_dir']+'/videos'):
                try:
                    os.makedirs(LOG_DIR+hyperparams['weight_dir']+'/videos')
                except:
                    pass

            self.video_dir = LOG_DIR+hyperparams['weight_dir']+'/videos/'

        self.task_queue = hyperparams['task_queue']
        self.motion_queue = hyperparams['motion_queue']
        self.rollout_queue = hyperparams['rollout_queue']
        self.ll_queue = hyperparams['ll_queue']
        self.hl_queue = hyperparams['hl_queue']
        self.cont_queue = hyperparams['cont_queue']

        self.pol_cls = DummyPolicy
        self.opt_cls = DummyPolicyOpt

        self.label_type = 'base'
        self._n_plans = 0
        n_plans = hyperparams['policy_opt']['buffer_sizes']['n_plans']
        self._last_weight_read = 0.

        self.permute_hl = hyperparams['permute_hl'] > 0
        self.neg_ratio = hyperparams['perc_negative']
        self.use_neg = self.neg_ratio > 0
        self.opt_ratio = hyperparams['perc_optimal']
        self.dagger_ratio = hyperparams['perc_dagger']
        self.rollout_ratio = hyperparams['perc_rollout']
        self.verbose = hyperparams['verbose']
        self.backup = hyperparams['backup']
        self.end2end = hyperparams['end_to_end_prob']
        self.task_list = self.agent.task_list
        self.pol_list = tuple(hyperparams['policy_list'])
        self.stopped = False
        self.expert_demos = {'acs':[], 'obs':[], 'ep_rets':[], 'rews':[], 'tasks':[], 'use_mask':[]}
        self.last_log_t = time.time()
        self.policy_opts = {task: DummyPolicyOpt(self.update, self.prob) for task in self.task_list}
        
        self.current_id = 0
        self.cur_step = 0
        self.adj_eta = False
        self.prim_decay = hyperparams.get('prim_decay', 1.)
        self.prim_first_wt = hyperparams.get('prim_first_wt', 1.)
        self.explore_wt = hyperparams['explore_wt']
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        self.agent.plans, self.agent.openrave_bodies, self.agent.env = self.agent.prob.get_plans(use_tf=True)
        self.rollout_opt = hyperparams['rollout_opt']
        
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.ff_data_file = LOG_DIR+hyperparams['weight_dir']+'/ff_samples_{0}_{1}.pkl'
        self.log_file = LOG_DIR + hyperparams['weight_dir'] + '/rollout_logs/{0}_log.txt'.format(self.id)
        self.verbose_log_file = LOG_DIR + hyperparams['weight_dir'] + '/rollout_logs/{0}_verbose.txt'.format(self.id)
        self.n_plans = 0
        self.n_failed = 0

    
    def map_cont_discr_tasks(self):
        self.task_types = []
        self.discrete_opts = []
        self.continuous_opts = []
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for key, val in opts.items():
            if hasattr(val, '__len__'):
                self.task_types.append('discrete')
                self.discrete_opts.append(key)
            else:
                self.task_types.append('continuous')
                self.continuous_opts.append(key)

    
    def init_policy_opt(self, hyperparams):
        hyperparams['policy_opt']['gpu_id'] = np.random.randint(1,3)
        hyperparams['policy_opt']['use_gpu'] = 1
        hyperparams['policy_opt']['load_label'] = hyperparams['classify_labels']
        hyperparams['policy_opt']['split_hl_loss'] = hyperparams['split_hl_loss']
        hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] # + '_trained'
        hyperparams['policy_opt']['scope'] = None
        hyperparams['policy_opt']['gpu_fraction'] = 1./32.
        hyperparams['policy_opt']['allow_growth'] = True
        self.policy_opt = hyperparams['policy_opt']['type'](hyperparams['policy_opt'],
                                                            hyperparams['dO'],
                                                            hyperparams['dU'],
                                                            hyperparams['dPrimObs'],
                                                            hyperparams['dContObs'],
                                                            hyperparams['dValObs'],
                                                            hyperparams['prim_bounds'],
                                                            hyperparams['cont_bounds'])
        self.weights_to_store = {}


    def end(self, msg):
        self.stopped = True
        # rospy.signal_shutdown('Received signal to terminate.')


    def spawn_problem(self, x0=None, targets=None):
        if x0 is None or targets is None:
            x0, targets = self.new_problem()

        initial, goal = self.agent.get_hl_info(x0, targets)
        problem = list(self.agent.plans.values())[0].prob
        domain = list(self.agent.plans.values())[0].domain
        problem.goal = goal
        abs_prob = self.agent.hl_solver.translate_problem(problem, goal=goal, initial=initial)
        ref_x0 = self.agent.clip_state(x0)
        for pname, attr in self.agent.state_inds:
            p = problem.init_state.params[pname]
            if p.is_symbol(): continue
            getattr(p, attr)[:,0] = ref_x0[self.agent.state_inds[pname, attr]]
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
                             label=self.id,
                             info=self.agent.get_hist_info())
        return hlnode


    def new_problem(self):
        x0, targets = self.agent.get_random_initial_state_vec(self.config, self.agent._eval_mode, self.agent.dX, self.agent.state_inds, 1)
        x0, targets = x0[0], targets[0]
        target_vec = np.zeros(self.agent.target_dim)
        for (tname, attr), inds in self.agent.target_inds.items():
            if attr != 'value': continue
            target_vec[inds] = targets[tname]
        return x0, target_vec


    def update(self, obs, mu, prc, wt, task, label, acts=[], ref_acts=[], terminal=[], aux=[], primobs=[], x=[]):
        assert(len(mu) == len(obs))
        assert len(label)

        #prc[np.where(prc > 1e10)] = 1e10
        #wt[np.where(wt > 1e10)] = 1e10
        #prc[np.where(prc < -1e10)] = -1e10
        #wt[np.where(wt < -1e10)] = -1e10
        #mu[np.where(np.abs(mu) > 1e10)] = 0
        if np.any(np.isnan(obs)):
            print((obs, task, np.isnan(obs)))
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        #obs[np.where(np.abs(obs) > 1e10)] = 0

        primobs = [] if task in ['primitive', 'cont', 'label'] or self.end2end == 0 else primobs
        data = (obs, mu, prc, wt, aux, primobs, x, task, label)
        if task == 'primitive':
            q = self.hl_queue
        elif task == 'cont':
            q = self.cont_queue
        elif task == 'label':
            q = self.label_queue
        else:
            q = self.ll_queue[task] if task in self.ll_queue else self.ll_queue['control']

        self.push_queue(data, q)


    def policy_call(self, x, obs, t, noise, task, opt_s=None):
        if noise is None: noise = np.zeros(self.agent.dU)
        if 'control' in self.policy_opt.task_map:
            alg_key = 'control'
        else:
            alg_key = task

        if self.policy_opt.task_map[alg_key]['policy'].scale is None:
            if opt_s is not None:
                return opt_s.get_U(t)
            raise ValueError('Cannot call policy without scale or bias')
        return self.policy_opt.task_map[alg_key]['policy'].act(x.copy(), obs.copy(), t, noise)


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None):
        if self.adj_eta: eta *= self.agent.eta_scale
        distrs = self.policy_opt.task_distr(prim_obs, eta)
        if not soft: return distrs
        out = []
        opts = self.agent.prob.get_prim_choices(self.task_list)
        enums = list(opts.keys())
        for ind, d in enumerate(distrs):
            enum = enums[ind]
            if not np.isscalar(opts[enum]):
                p = d / np.sum(d)
                ind = np.random.choice(list(range(len(d))), p=p)
                d[ind] += 1e2
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
        if self.policy_opt.share_buffers and time.time() - self._last_weight_read > 120:
            self.policy_opt.read_shared_weights()
            self._last_weight_read = time.time()
        scopes = self.policy_opt.valid_scopes
        task_map = self.policy_opt.task_map
        rollout_policies = {task: DummyPolicy(task,
                                              self.policy_call,
                                              scale=task_map[task if task in scopes else 'control']['policy'].scale)
                                  for task in self.agent.task_list}

        if len(self.agent.continuous_opts):
            rollout_policies['cont'] = ContPolicy(self.policy_opt)
        self.agent.policies = rollout_policies


    def run(self):
        raise NotImplementedError()


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(self.agent.discrete_opts)
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dOpts)), np.zeros((0))
        tgt_aux = np.zeros((0))
        tgt_x = np.zeros((0, self.agent.dX))
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for ind, sample in enumerate(samples):
            mu = sample.get_prim_out()
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_x = np.concatenate((tgt_x, sample.get_X()))
            st, et = 0, sample.T
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            wt = np.array([sample.prim_use_ts[t] * self.prim_decay**t for t in range(sample.T)])
            if sample.task_start and ind > 0 and sample.opt_strength > 0.999: wt[0] = self.prim_first_wt
            if sample.opt_strength < 1-1e-3: wt[:] *= self.explore_wt
            wt[:] *= sample.wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            if np.any(np.isnan(obs)):
                print((obs, sample.task, 'SAMPLE'))
            obs_data = np.concatenate((obs_data, obs))
            prc = np.concatenate([self.agent.get_mask(sample, enum) for enum in self.discrete_opts], axis=-1) # np.tile(np.eye(dP), (sample.T,1,1))
            if not self.config['hl_mask']:
                prc[:] = 1.
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', samples[0].source_label, aux=tgt_aux, x=tgt_x)


    def update_cont_network(self, samples):
        dP, dO = self.agent.dContOut, self.agent.dCont
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        tgt_aux = np.zeros((0))
        tgt_x = np.zeros((0, self.agent.dX))
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)

        for ind, sample in enumerate(samples):
            mu = sample.get_cont_out()
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_x = np.concatenate((tgt_x, sample.get_X()))
            st, et = 0, sample.T # st, et = sample.step * sample.T, (sample.step + 1) * sample.T
            #aux = np.ones(sample.T)
            #if sample.task_start: aux[0] = 0.
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            wt = np.ones(sample.T)
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_cont_obs()
            if np.any(np.isnan(obs)):
                print((obs, sample.task, 'SAMPLE'))
            obs_data = np.concatenate((obs_data, obs))
            prc = np.tile(np.eye(dP), (sample.T,1,1))
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            # print('Sending update to primitive net')
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'cont', samples[0].source_label, aux=tgt_aux, x=tgt_x)


    def update_negative_primitive(self, samples):
        if not self.use_neg or not len(samples): return
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(self.agent.discrete_opts) 
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dOpts)), np.zeros((0))
        tgt_aux = np.zeros((0))
        tgt_x = np.zeros((0, self.agent.dX))
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        cont_mask = {}
        for enum in opts:
            cont_mask[enum] = 0. if np.isscalar(opts[enum]) else 1.

        for sample, ts, task in samples:
            mu = []
            for ind, val in enumerate(task):
                opt = self.agent.discrete_opts[ind]
                vec = np.ones(len(opts[opt]))
                if len(opts[opt]) > 1:
                    vec[val] = 0.
                    vec /= np.sum(vec)
                mu.append(vec)
            mu = [np.concatenate(mu)]
            tgt_mu = np.concatenate((tgt_mu, mu))
            wt = np.ones(1) # sample.prim_use_ts[ts:ts+1]
            obs = [sample.get_prim_obs(ts)]
            aux = np.ones(1)
            tgt_x = np.concatenate((tgt_x, sample.get_X()))
            tgt_aux = np.concatenate((tgt_aux, aux))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, obs))

            prc = np.concatenate([cont_mask[enum] * self.agent.get_mask(sample, enum) for enum in self.agent.discrete_opts], axis=-1) # np.tile(np.eye(dP), (sample.T,1,1))
            prc = prc[ts:ts+1]
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 'negative', aux=tgt_aux, x=tgt_x)

        with self.policy_opt.buf_sizes['n_negative'].get_lock():
            self.policy_opt.buf_sizes['n_negative'].value += len(tgt_mu)


    def update_labels(self, labels, obs, x):
        assert len(x) > 0
        dOpts = len(self.agent.discrete_opts)
        prc = np.ones((len(labels), 2))
        self.update(obs, labels, prc, np.ones(len(obs)), 'label', 'human', aux=[], x=x)


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
                info['targets'] = {tname: sample.targets[self.agent.target_inds[tname, attr]] for tname, attr in self.agent.target_inds if attr == 'value'}
                info['opt_success'] = sample.opt_suc
                info['tasks'] = sample.get(FACTOREDTASK_ENUM)
                #info['goal_pose'] = sample.get(END_POSE_ENUM)
                info['actions'] = sample.get(ACTION_ENUM)
                info['end_state'] = sample.end_state
                info['plan_fail_rate'] = self.n_failed / self.n_plans if self.n_plans > 0 else 0.
                info['source'] = sample.source_label
                # info['prim_obs'] = sample.get_prim_obs().round(4)
                info['memory'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                info['last_weight_read'] = self._last_weight_read 

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


    def _gen_video(self, rollout, st=0, ts=None, annotate=False, tdelta=1):
        if not self.render: return None
        old_h = self.agent.image_height
        old_w = self.agent.image_width
        self.agent.image_height = 256
        self.agent.image_width = 256
        cam_ids = self.config.get('visual_cameras', [self.agent.camera_id])
        buf = []
        for step in rollout:
            if not step.draw: continue
            old_vec = self.agent.target_vecs[0]
            self.agent.target_vecs[0] = step.targets
            ts = (st, step.T) if ts is None else ts
            ts_range = range(ts[0], ts[1], tdelta)
            st = 0

            for t in ts_range:
                if t >= step.T: break
                ims = []
                for ind, cam_id in enumerate(cam_ids):
                    if annotate and ind == 0:
                        ims.append(self.agent.get_annotated_image(step, t, cam_id=cam_id))
                    else:
                        ims.append(self.agent.get_image(step.get_X(t=t), cam_id=cam_id))
                im = np.concatenate(ims, axis=1)
                buf.append(im)
            self.agent.target_vecs[0] = old_vec
        self.agent.image_height = old_h
        self.agent.image_width = old_w
        return np.array(buf)


    def _update_policy_no_cost(self, task, optimal_samples, label, inv_cov):
        dU, dO = self.agent.dU, self.agent.dO
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dU))
        tgt_prc, tgt_wt = np.zeros((0, dU, dU)), np.zeros((0,))
        x_data = np.zeros((0, self.agent.dX))
        prim_obs_data = np.zeros((0, self.agent.dPrim))

        for sample in optimal_samples:
            prc = np.zeros((1, sample.T, dU, dU))
            wt = np.zeros((sample.T,))
            traj = self.new_traj_distr[0]
            for t in range(sample.T):
                prc[:, t, :, :] = np.tile(inv_cov, [1, 1, 1])
                wt[t] = sample.use_ts[t]

            tgt_mu = np.concatenate((tgt_mu, sample.get_U()))
            obs_data = np.concatenate((obs_data, sample.get_obs()))
            prim_obs_data = np.concatenate((prim_obs_data, sample.get_prim_obs()))
            x_data = np.concatenate((x_data, sample.get_X()))
            tgt_wt = np.concatenate((tgt_wt, wt))
            tgt_prc = np.concatenate((tgt_prc, prc.reshape(-1, dU, dU)))

        obs_data = obs_data.reshape((-1, dO))
        prim_obs_data = prim_obs_data.reshape((-1, self.agent.dPrim))
        x_data = x_data.reshape((-1, self.dX))
        tgt_mu = tgt_mu.reshape((-1, dU))
        tgt_wt = tgt_wt.reshape((-1, 1))
        tgt_prc = tgt_prc.reshape((-1, dU, dU))

        self.policy_opts[task].update(obs_data, tgt_mu, tgt_prc, tgt_wt, task, label, primobs=prim_obs_data, x=x_data)


    def save_video(self, rollout, success=None, ts=None, lab='', annotate=True, st=0):
        if not self.render: return
        init_t = time.time()
        old_h = self.agent.image_height
        old_w = self.agent.image_width
        self.agent.image_height = 256
        self.agent.image_width = 256
        suc_flag = ''
        cam_ids = self.config.get('visual_cameras', [self.agent.camera_id])
        if success is not None:
            suc_flag = 'success' if success else 'fail'
        fname = self.video_dir + '/{0}_{1}_{2}_{3}{4}_{5}.npy'.format(self.id, self.group_id, self.cur_vid_id, suc_flag, lab, str(cam_ids)[1:-1].replace(' ', ''))
        self.cur_vid_id += 1
        buf = []
        for step in rollout:
            if not step.draw: continue
            old_vec = self.agent.target_vecs[0]
            self.agent.target_vecs[0] = step.targets
            if ts is None: 
                ts_range = range(st, step.T)
            else:
                ts_range = range(ts[0], ts[1])
            st = 0

            for t in ts_range:
                ims = []
                for ind, cam_id in enumerate(cam_ids):
                    if annotate and ind == 0:
                        ims.append(self.agent.get_annotated_image(step, t, cam_id=cam_id))
                    else:
                        ims.append(self.agent.get_image(step.get_X(t=t), cam_id=cam_id))
                im = np.concatenate(ims, axis=1)
                buf.append(im)
            self.agent.target_vecs[0] = old_vec
        #np.save(fname, np.array(buf))
        #print('Time to create video:', time.time() - init_t)
        init_t = time.time()
        save_video(fname, dname=self._hyperparams['descr'], arr=np.array(buf), savepath=self.video_dir)
        self.agent.image_height = old_h
        self.agent.image_width = old_w
        #print('Time to save video:', time.time() - init_t)


    def log_node_info(self, node, success, path):
        key = 'n_ff'
        if node.label.find('post') >= 0:
            key = 'n_postcond'
        elif node.label.find('pre') >= 0:
            key = 'n_precond'
        elif node.label.find('mid') >= 0:
            key = 'n_midcond'
        elif node.label.find('rollout') >= 0:
            key = 'n_explore'

        self.infos[key] += 1
        self.infos['n_plans'] += 1
        for altkey in self.avgs:
            if altkey != key:
                self.avgs[altkey].append(0)
            else:
                self.avgs[altkey].append(1)

        failkey = key.replace('n_', 'n_fail_')
        if not success:
            self.fail_infos[failkey] += 1
            self.fail_infos['n_fail_plans'] += 1
            self.fail_avgs[failkey].append(0)
        else:
            self.fail_avgs[failkey].append(1)

        with self.policy_opt.buf_sizes[key].get_lock():
            self.policy_opt.buf_sizes[key].value += 1


    def get_log_info(self):
        info = {
                'time': time.time() - self.start_t,
                'optimization time': np.mean(self.plan_times),
                'plan length': np.mean(self.plan_horizons),
                'opt duration per ts': np.mean(self.plan_times) / np.mean(self.plan_horizons),
                }

        for key in self.infos:
            info[key] = self.infos[key]

        for key in self.fail_infos:
            info[key] = self.fail_infos[key]

        for key in self.fail_rollout_infos:
            info[key] = self.fail_rollout_infos[key]

        wind = 10
        for key in self.avgs:
            if len(self.avgs[key]):
                info[key+'_avg'] = np.mean(self.avgs[key][-wind:])

        for key in self.fail_avgs:
            if len(self.fail_avgs[key]):
                info[key+'_avg'] = np.mean(self.fail_avgs[key][-wind:])

        for key in self.opt_rollout_info:
            if len(self.opt_rollout_info[key]):
                info[key] = np.mean(self.opt_rollout_info[key][-wind:])

        if len(self.init_costs): info['mp initial costs'] = np.mean(self.init_costs[-10:])
        if len(self.rolled_costs): info['mp rolled out costs'] = np.mean(self.rolled_costs[-10:])
        if len(self.final_costs): info['mp optimized costs'] = np.mean(self.final_costs[-10:])
        return info #self.log_infos


    def write_log(self):
        with open(self.motion_log, 'a+') as f:
            info = self.get_log_info()
            pp_info = pprint.pformat(info, depth=60)
            f.write(str(pp_info))
            f.write('\n\n')


class ContPolicy:
    def __init__(self, policy_opt):
        self.policy_opt = policy_opt

    def initialized(self):
        return self.policy_opt.cont_policy.scale is not None

    def act(self, x, obs, t, noise=None):
        return self.policy_opt.cont_task(obs)


class DummyPolicy:
    def __init__(self, task, policy_call, opt_sample=None, chol_pol_covar=None, scale=None):
        self.task = task
        self.policy_call = policy_call
        self.opt_sample = opt_sample
        self.chol_pol_covar = chol_pol_covar
        self.scale = scale

    def act(self, x, obs, t, noise=None):
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


