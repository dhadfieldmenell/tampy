import cPickle as pickle


def retrain(rollout_server, hl_dir, ll_dir, maxlen=100000, incr=5000, rlen=10):
    hl_file = 'tf_saved/'+hl_dir+'/primitive_data.pkl'
    with open(hl_file, 'r') as f:
        hl_data = pickle.load(f)
    for scope in rollout_server.policy_opt.valid_scopes:
        rollout_server.policy_opt.restore_ckpt(scope, dirname=ll_dir)

    mu, obs, prc, wt = hl_data[:4]
    cur_ind = 0
    key = 'primitive'
    rollout_server.policy_opt._hyperparams['iterations'] = incr // 10
    while cur_ind < maxlen:
        print('Loading data at ind', cur_ind)
        rollout_server.policy_opt.store(obs[key][key][cur_ind:cur_ind+incr], 
                                        mu[key][key][cur_ind:cur_ind+incr], 
                                        prc[key][key][cur_ind:cur_ind+incr], 
                                        wt[key][key][cur_ind:cur_ind+incr], key, key)
        updated = rollout_server.policy_opt.run_update([key])
        if updated:
            rollout_server.policy_opt.store_scope_weights([key])
            rollout_server.policy_opt.write_shared_weights([key])
        cur_ind += incr
        for _ in range(5):
            rollout_server.test_hl(rlen=rlem, save=True, debug=False)
        if cur_ind > len(obs[key][key]): break

    print('Finished retrain', cur_ind)

