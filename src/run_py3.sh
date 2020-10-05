for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 4 -nt 4 -spl -llus 5000  -hlus 10000  -ff 1. -retime -hln 2 -mask -hldim 64 -lldim 64 -eta 3 -softev -obs_del -hist_len 3 -prim_first_wt 1 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 2 -fail -failmode collision -col_coeff 0. -descr replan_with_col & 
        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 4 -nt 4 -spl -llus 5000  -hlus 10000  -ff 1. -retime -hln 2 -mask -hldim 64 -lldim 64 -eta 5 -softev -obs_del -hist_len 3 -prim_first_wt 1 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 2 -fail -failmode random -col_coeff 0. -descr replan_no_col & 
        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

