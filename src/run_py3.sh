for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 4 -nt 4 -spl -llus 5000  -hlus 10000  -ff 1. -retime -hln 2 -mask -hldim 64 -lldim 64 -eta 4 -softev -obs_del -hist_len 2 -prim_first_wt 1 -lr 0.0005 -hllr 0.0005 -lldec 0.00 -hldec 0.0005 -add_noop 2 --permute_hl 1 -fail -failmode multistep -expl_wt 100 -tfwarmup 300 -expl_eta 3 -expl_n 10 -expl_suc 5 -col_coeff 0. -descr polresample_N10_s5 & 
        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

