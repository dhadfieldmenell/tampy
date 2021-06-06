for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.swap_hyp \
                                                       -no 2 -llus 1000  -hlus 2000 \
                                                       -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.25 -eta 5 -softev \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 10 -lr 0.0004 \
                                                       -hllr 0.0002 -lldec 0.00004 -hldec 0.0001 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 20 \
                                                       -rollout 12 \
                                                       -task 2 \
                                                       -swap \
                                                       -post -pre \
                                                       -render -verbose \
                                                       -warm 200 \
                                                       -roll_opt \
                                                       -neg_ratio 0.0 -opt_ratio 0.5 -dagger_ratio 0.5 \
                                                       -descr fri_redo_swap_targets &
        sleep 5h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

