for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -spl -soft -descr four_grasp_softtrain_ll_nosplit&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -soft -descr four_grasp_softtrain_ll_split&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

done

