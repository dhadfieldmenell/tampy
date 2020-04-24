for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -softev -descr four_grasp_full_obs_ll_split&
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v50 -no 1 -nt 1 -softev -descr four_target_simple_obs_ll_split&
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

done

