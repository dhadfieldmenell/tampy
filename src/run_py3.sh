for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 5 -nt 5 -spl -llus 5000  -hlus 10000  -ff 1. -hln 2 -hldim 64 -lldim 64 -eta 3 -softev -mask -hist_len 3 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 4 -fail -failmode random -obs_del -descr namo_base_onegrasp_morehist & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
   done
done

