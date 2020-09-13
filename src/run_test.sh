for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 5 -nt 5 -spl -llus 5000  -hlus 5000  -ff 1. -hln 2 -hldim 64 -lldim 64 -eta 10 -softev -mask -hist_len 2 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -add_noop 3 -fail -failmode random -obs_del -test namo_objs5_5/base_namo_random_train_from_fail_4 -descr test_base_namo
        sleep 2h 30m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
   done
done

