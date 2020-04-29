for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -spl -q -softev -descr qlearn_base_fourbyfour &
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -spl -q -her -softev -descr qlearn_her_fourbyfour &
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -spl -softev -cur 4 -ncur 5 -descr q_cur_4_5_fourbyfour &
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 1 -nt 1 -spl -softev -descr base_fourbyfour &
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

done

