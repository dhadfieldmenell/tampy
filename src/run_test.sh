for N in 1
do
   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 1 -nt 1 -render -spl -descr compactest  -eta 5 -softev -ff 1. -test namo_objs1_1/exp_id0_random_compact0&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s


done

