for N in 1 2 3 4 5
do
 
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -eta 8 -softev -ff 1. -descr network1by32&
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
    
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v82 -no 2 -nt 1 -spl -eta 8 -softev -ff 1. -descr network2by32&
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v83 -no 2 -nt 1 -spl -eta 8 -softev -ff 1. -descr network1by64&
    sleep 1h 30m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
 
done

