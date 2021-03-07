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
import pybullet as P
from scipy.cluster.vq import kmeans2 as kmeans
# import tensorflow as tf

from core.internal_repr.plan import Plan
from policy_hooks.sample import Sample
from policy_hooks.sample_list import SampleList
from policy_hooks.utils.policy_solver_utils import *
from policy_hooks.msg_classes import *
from policy_hooks.save_video import save_video
from policy_hooks.search_node import *


LOG_DIR = 'experiment_logs/'

class Server(object):
    def __init__(self, hyperparams):
        global tf
        import tensorflow as tf
        self.id = hyperparams['id']
        self._hyperparams = hyperparams
        self.config = hyperparams
        self.group_id = hyperparams['group_id']

        self.start_t = hyperparams['start_t']
        self.seed = int((1e2*time.time()) % 1000.)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.render = hyperparams.get('load_render', False)
        if self.config['weight_dir'].find('sawyer') >= 0:
            if self.id.find('moretest') < 0 and self.id.find('0') < 0:# and self.id.find('Rollout') < 0:
                self.render = False
                hyperparams['load_render'] = False
                hyperparams['agent']['master_config']['load_render'] = False

        n_gpu = hyperparams['n_gpu']
        if n_gpu == 0:
            gpus = -1
        elif n_gpu == 1:
            gpu = 0
        else:
            gpus = str(list(range(1, n_gpu+1)))[1:-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = "{0}".format(gpus)

        self.solver = hyperparams['mp_solver_type'](hyperparams)
        self.opt_smooth = hyperparams.get('opt_smooth', False)
        self.alg_map = hyperparams['alg_map']
        for alg in list(self.alg_map.values()):
            #alg.set_conditions(len(self.agent.x0))
            alg.set_conditions(1)
        self.init_policy_opt(hyperparams)
        hyperparams['agent']['master_config'] = hyperparams
        try:
            P.disconnect()
        except:
            print('No need to disconnect pybullet')
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        self.agent.process_id = '{0}_{1}'.format(self.id, self.group_id)
        self.agent.solver = self.solver
        self.map_cont_discr_tasks()
        self.prob = self.agent.prob
        self.solver.agent = self.agent
        
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

        self.pol_cls = DummyPolicy
        self.opt_cls = DummyPolicyOpt

        self.label_type = 'base'
        self._n_plans = 0
        n_plans = hyperparams['policy_opt']['buffer_sizes']['n_plans']
        self._last_weight_read = time.time()

        self.permute_hl = hyperparams['permute_hl'] > 0
        self.use_neg = hyperparams['negative']
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
        self.explore_wt = hyperparams['explore_wt']
        self.check_prim_t = hyperparams.get('check_prim_t', 1)
        self.agent.plans, self.agent.openrave_bodies, self.agent.env = self.agent.prob.get_plans(use_tf=True)
        self.dagger_window = hyperparams['dagger_window']
        task_plans = list(self.agent.plans.items())
        for task, plan in task_plans:
            #self.agent.plans[task[0]] = plan
            plan.state_inds = self.agent.state_inds
            plan.action_inds = self.agent.action_inds
            plan.dX = self.agent.dX
            plan.dU = self.agent.dU
            plan.symbolic_bound = self.agent.symbolic_bound
            plan.target_dim = self.agent.target_dim
            plan.target_inds = self.agent.target_inds
            for param in plan.params.values():
                for attr in param.attrs:
                    if (param.name, attr) not in plan.state_inds:
                        if type(getattr(param, attr)) is not np.ndarray: continue
                        val = getattr(param, attr)[:,0]
                        if np.any(np.isnan(val)):
                            getattr(param, attr)[:] = 0.
                        else:
                            getattr(param, attr)[:,:] = val.reshape((-1,1))
        self.expert_data_file = LOG_DIR+hyperparams['weight_dir']+'/'+str(self.id)+'_exp_data.npy'
        self.ff_data_file = LOG_DIR+hyperparams['weight_dir']+'/ff_samples_{0}_{1}.pkl'
        self.log_file = LOG_DIR + hyperparams['weight_dir'] + '/{0}_log.txt'.format(self.id)
        self.verbose_log_file = LOG_DIR + hyperparams['weight_dir'] + '/{0}_verbose.txt'.format(self.id)
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
        hyperparams['policy_opt']['split_hl_loss'] = hyperparams['split_hl_loss']
        hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] # + '_trained'
        hyperparams['policy_opt']['scope'] = None
        hyperparams['policy_opt']['gpu_fraction'] = 1./32.
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
        x0, targets = self.agent.get_random_initial_state_vec(self.config, self.agent.plans, self.agent.dX, self.agent.state_inds, 1)
        x0, targets = x0[0], targets[0]
        target_vec = np.zeros(self.agent.target_dim)
        for (tname, attr), inds in self.agent.target_inds.items():
            #if attr != 'value': continue
            target_vec[inds] = targets[tname]
        return x0, target_vec



    def update(self, obs, mu, prc, wt, task, label, acts=[], ref_acts=[], terminal=[], aux=[], primobs=[], x=[]):
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

        data = (obs, mu, prc, wt, aux, primobs, x, task, label)
        if task == 'primitive':
            q = self.hl_queue
        else:
            q = self.ll_queue

        self.push_queue(data, q)


    def policy_call(self, x, obs, t, noise, task, opt_s=None):
        if noise is None: noise = np.zeros(self.agent.dU)
        if 'control' in self.policy_opt.task_map:
            alg_key = 'control'
        else:
            alg_key = task

        if self.policy_opt.task_map[alg_key]['policy'].scale is None:
            if opt_s is not None:
                return opt_s.get_U(t) # + self.alg_map[task].cur[0].traj_distr.chol_pol_covar[t].T.dot(noise)
            t = min(t, self.alg_map[task].cur[0].traj_distr.T-1)
            return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
        return self.policy_opt.task_map[alg_key]['policy'].act(x.copy(), obs.copy(), t, noise)


    def primitive_call(self, prim_obs, soft=False, eta=1., t=-1, task=None):
        if self.adj_eta: eta *= self.agent.eta_scale
        distrs = self.policy_opt.task_distr(prim_obs, eta)
        #if task is not None and t % self.check_prim_t:
        #    for i in range(len(distrs)):
        #        distrs[i] = np.zeros_like(distrs[i])
        #        distrs[i][task[i]] = 1.
        #    return distrs
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
               node.heuristic() < prob.heuristic():
                prob = node

        try:
            q.put_nowait(prob)
        except queue.Full:
            pass


    def set_policies(self):
        inter = 180
        if self.policy_opt.share_buffers and time.time() - self._last_weight_read > inter:
            self.policy_opt.read_shared_weights()
        chol_pol_covar = {}
        self._last_weight_read = time.time()
        #for task in self.agent.task_list:
        #    if task not in self.policy_opt.valid_scopes:
        #        task_name = 'control'
        #    else:
        #        task_name = task

        #    #if self.policy_opt.task_map[task_name]['policy'].scale is None:
        #    #    chol_pol_covar[task] = np.eye(self.agent.dU) # self.alg_map[task].cur[0].traj_distr.chol_pol_covar
        #    #else:
        #    #    chol_pol_covar[task] = self.policy_opt.task_map[task_name]['policy'].chol_pol_covar

        rollout_policies = {task: DummyPolicy(task,
                                              self.policy_call,
                                              scale=self.policy_opt.task_map[task if task in self.policy_opt.valid_scopes else 'control']['policy'].scale)
                                  for task in self.agent.task_list}
        self.agent.policies = rollout_policies


    def run_hl_update(self, label=None):
        ### Look for saved successful HL rollout paths and send them to update the HL options policy
        ref_paths = []
        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)
            ref_paths.append(path)
        self.agent.clear_task_paths()
        if label is not None:
            for s in path_samples:
                s.source_label = label
        
        self.update_primitive(path_samples)
        if self._hyperparams.get('save_expert', False): self.update_expert_demos(ref_paths)


    def run(self):
        raise NotImplementedError()


    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(list(self.agent.prob.get_prim_choices(self.agent.task_list).keys()))
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dOpts)), np.zeros((0))
        tgt_aux = np.zeros((0))
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)

        #if len(samples):
        #    lab = samples[0].source_label
        #    lab = 'n_plans' if lab == 'optimal' else 'n_rollout'
        #    if lab in self.policy_opt.buf_sizes:
        #        with self.policy_opt.buf_sizes[lab].get_lock():
        #            self.policy_opt.buf_sizes[lab].value += 1
        #        samples[0].source_label = ''

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
            wt[:] *= sample.wt
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs = sample.get_prim_obs()
            if np.any(np.isnan(obs)):
                print((obs, sample.task, 'SAMPLE'))
            obs_data = np.concatenate((obs_data, obs))
            prc = np.concatenate([self.agent.get_mask(sample, enum) for enum in opts], axis=-1) # np.tile(np.eye(dP), (sample.T,1,1))
            if not self.config['hl_mask']:
                prc[:] = 1.
            tgt_prc = np.concatenate((tgt_prc, prc))

        if len(tgt_mu):
            # print('Sending update to primitive net')
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', samples[0].source_label, aux=tgt_aux)


    def update_negative_primitive(self, samples):
        if not self.use_neg or not len(samples): return
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        dOpts = len(list(self.agent.prob.get_prim_choices(self.agent.task_list).keys()))
        ### Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dOpts)), np.zeros((0))
        tgt_aux = np.zeros((0))
        opts = self.agent.prob.get_prim_choices(self.agent.task_list)
        for sample, ts, task in samples:
            wt = sample.prim_use_ts[ts:ts+1]
            if sample.opt_strength < 1-1e-3: wt[0] *= self.explore_wt
            wt[:] *= sample.wt
            if wt[0] == 0: continue
            obs = sample.get_prim_obs(ts)
            aux = int(sample.opt_strength) * np.ones(sample.T)
            tgt_aux = np.concatenate((tgt_aux, aux))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, obs))
            prc = np.concatenate([self.agent.get_mask(sample, enum) for enum in opts], axis=-1) # np.tile(np.eye(dP), (sample.T,1,1))
            prc = prc[t:t+1]
            if not self.config['hl_mask']:
                prc[:] = 1.
            tgt_prc = np.concatenate((tgt_prc, prc))
           
            mu = []
            for ind, enum in enumerate(opts.keys()):
                if np.isscalar(opts[enum]):
                    mu.append(task[ind])
                else:
                    vec = np.zeros(len(opts[enum]))
                    vec[task[ind]] = 1.
                    mu.append(vec)
                mu = np.r_[mu]
            mu = np.concatenate([sample.get(enum) for enum in self.config['prim_out_include']], axis=-1)
            tgt_mu = np.concatenate((tgt_mu, mu))

        wt *= -1
        assert np.all(wt < 0)
        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', self.label_type, aux=tgt_aux)

        with self.policy_opt.buf_sizes['n_negative'].get_lock():
            self.policy_opt.buf_sizes['n_negative'].value += len(tgt_mu)


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
                #info['goal_pose'] = sample.get(END_POSE_ENUM)
                info['actions'] = sample.get(ACTION_ENUM)
                info['end_state'] = sample.end_state
                info['plan_fail_rate'] = self.n_failed / self.n_plans if self.n_plans > 0 else 0.
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


    def save_image(self, rollout=None, success=None, ts=0, render=True, x=None):
        if not self.render: return
        suc_flag = ''
        if success is not None:
            suc_flag = 'succeeded' if success else 'failed'
        fname = '/home/michaelmcdonald/Dropbox/videos/{0}_{1}_{2}.png'.format(self.id, self.cur_vid_id, suc_flag)
        self.cur_vid_id += 1
        if rollout is not None: self.agent.target_vecs[0][:] = rollout.targets
        if render:
            if x is None:
                x = rollout.get_X(t=ts)
            im = self.agent.get_image(x)
        else:
            im = rollout.get(IM_ENUM, t=ts).reshape((self.agent.image_height, self.agent.image_width, 3))
            im = (128 * im) + 128
            im = im.astype(np.uint8)
        im = Image.fromarray(im)
        im.save(fname)


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
            self.agent.target_vecs[0] = step.targets
            if not step.draw: continue
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
        #np.save(fname, np.array(buf))
        print('Time to create video:', time.time() - init_t)
        init_t = time.time()
        save_video(fname, dname=self._hyperparams['descr'], arr=np.array(buf), savepath=self.video_dir)
        self.agent.image_height = old_h
        self.agent.image_width = old_w
        print('Time to save video:', time.time() - init_t)


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


