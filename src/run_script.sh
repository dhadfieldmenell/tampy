for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 2 -nt 2 -spl -cur 4 -ncur 5 -softev -descr two_obj_1by4_cur4_5 &
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 2 -nt 2 -spl -softev -q -descr two_obj_1by4_q &
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v40 -no 2 -nt 2 -spl -softev -descr two_obj_1by4_base &
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
done

