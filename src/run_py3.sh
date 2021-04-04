for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 \
                                                       -no 5 -llus 5000  -hlus 10000 \
                                                       -spl -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.0 \
                                                       -motion 22 \
                                                       -rollout 8 \
                                                       -task 2 \
                                                       -post -pre \
                                                       -render \
                                                       -warm 100 \
                                                       -descr verify_dagger_check_fri & 
        sleep 4h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

