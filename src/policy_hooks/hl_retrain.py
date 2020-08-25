import pickle as pickle
import numpy as np
import os


def retrain(rollout_server, hl_dir, ll_dir, maxlen=100000, incr=5000, rlen=5):
    log_infos = []
    hl_file = 'tf_saved/'+hl_dir+'/primitive_data.pkl'
    with open(hl_file, 'r') as f:
        hl_data = pickle.load(f)
    for scope in rollout_server.policy_opt.valid_scopes:
        rollout_server.policy_opt.restore_ckpt(scope, dirname=ll_dir)

    iters = rollout_server.config['train_iterations']
    mu, obs, prc, wt = hl_data[:4]
    val_mu, val_obs, val_prc, val_wt = hl_data[4:8]
    cur_ind = 0
    key = 'primitive'
    while cur_ind < maxlen:
        print(('Loading data at ind', cur_ind))
        rollout_server.policy_opt.store(obs[key][key][cur_ind:cur_ind+incr],
                                        mu[key][key][cur_ind:cur_ind+incr],
                                        prc[key][key][cur_ind:cur_ind+incr],
                                        wt[key][key][cur_ind:cur_ind+incr], key, key, val_ratio=-1)
        if cur_ind+incr < len(val_mu[key][key]):
            rollout_server.policy_opt.store(val_obs[key][key][cur_ind:cur_ind+incr],
                                            val_mu[key][key][cur_ind:cur_ind+incr],
                                            val_prc[key][key][cur_ind:cur_ind+incr],
                                            val_wt[key][key][cur_ind:cur_ind+incr], key, key, val_ratio=1)
        updated = rollout_server.policy_opt.run_update([key])
        update_log_data(log_infos, rollout_server, cur_ind)
        if updated:
            rollout_server.policy_opt.store_scope_weights([key])
            rollout_server.policy_opt.write_shared_weights([key])
        cur_ind += incr
        for _ in range(20):
            rollout_server.agent.replace_cond(0)
            rollout_server.test_hl(rlen=rlen, save=True, debug=False)
        if cur_ind > len(obs[key][key]): break

    print(('Finished retrain', cur_ind))


def retrain_hl_from_samples(policy_server, hl_dir):
    hl_files = os.listdir('tf_saved/'+hl_dir)
    key = 'primitive'
    for fname in hl_files:
        if not fname.find('ff_samples') >= 0: continue
        with open('tf_saved/'+hl_dir+'/'+fname, 'r') as f:
            samples = pickle.load(f)
        for s in samples:
            s.agent = policy_server.agent
            s.reinit()

        psid = fname.split('.')[0].split('_')[-1]
        psid = int(psid)
        val_ratio = 1. if psid >= 25 else -1.
        obs, mu, prc, wt = policy_server.get_prim_update(samples)
        policy_server.policy_opt.store(obs,
                                        mu,
                                        prc,
                                        wt, key, key, val_ratio=val_ratio)
        policy_server.full_N += len(mu)
        policy_server.update_network() # policy_server.policy_opt.run_update()
    print('Finished retrain')



def retrain_from_samples(rollout_server, hl_dir, ll_dir, maxlen=100000, incr=5000, rlen=5):
    hl_files = os.listdir('tf_saved/'+hl_dir)
    key = 'primitive'
    for fname in hl_files:
        if not fname.find('ff_samples'): continue
        with open('tf_saved/'+hl_dir+'/'+fname, 'r') as f:
            samples = pickle.load(f)
        for s in samples:
            s.agent = rollout_server.agent
        obs, mu, prc, wt = rollout_server.get_prim_update(samples)
        rollout_server.policy_opt.store(obs,
                                        mu,
                                        prc,
                                        wt, key, key)
        updated = rollout_server.policy_opt.run_update()
        if updated:
            rollout_server.policy_opt.store_scope_weights([key])
            rollout_server.policy_opt.write_shared_weights([key])
        for _ in range(50):
            rollout_server.test_hl(rlen=rlen, save=True, debug=False)
    print(('Finished retrain', cur_ind))


def update_log_data(log_infos, rollout_server, full_N, time=0):
    policy_opt_log = 'tf_saved/'+rollout_server.config['weight_dir']+'/'+'policy_primitive_log.pkl'
    if not len(rollout_server.policy_opt.average_losses) or not len(rollout_server.policy_opt.average_val_losses):
        return log_infos
    losses = (rollout_server.policy_opt.average_losses[-1], rollout_server.policy_opt.average_val_losses[-1])
    policy_loss = (np.sum(losses[0]), np.sum(losses[1]))
    policy_component_loss = losses
    info = {
            'time': 0,
            'train_loss': policy_loss[0],
            'train_component_loss': policy_component_loss[0],
            'val_loss': policy_loss[1],
            'val_component_loss': policy_component_loss[1],
            'scope': 'primitive',
            'n_data': full_N,
            'N': full_N,
            }
    log_infos.append(info)
    with open(policy_opt_log, 'w+') as f:
        f.write(str(log_infos))
    return log_infos
