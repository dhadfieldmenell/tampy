for N in 1 2 3 4 5
do
    for S in base
    do

        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -x_select ${S} -eta 5 -softev -hln 2 -hldim 64 -ff 1. -descr unified_compact_${S} & 
        sleep 1h 30m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -x_select ${S} -eta 5 -softev -hln 2 -hldim 64 -ff 1. -descr split_compact_${S} & 
        sleep 1h 30m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

