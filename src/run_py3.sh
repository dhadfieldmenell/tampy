for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 5 -nt 5 -spl -llus 5000  -hlus 10000  -retime -ff 1. -hln 2 -hldim 64 -lldim 64 -prim_first_wt 1 -eta 5 -softev -mask -hist_len 2 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 2 --task_hist_len 1 -task_hist -permute 5 -fail -failmode random -obs_del -descr permute & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
   done
done

