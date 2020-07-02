for N in 1
do
   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 2 \
                                                  -render -spl -descr weighttest  -eta 1 -softev -ff 1.\
                                                  -llpol namo_objs2_2/exp_id0_weighted_newretime_6 \
                                                  -hlpol namo_objs2_2/exp_id0_weighted_newretime_6 \
                                                  -test namo_objs2_2/exp_id0_weighted_newretime_6&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s


done

