for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.swap_hyp \
                                                       -no 3 -llus 5000  -hlus 10000 \
                                                       -spl -mask -hln 2 -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.0004 -lldec 0.0001 -hldec 0.00001 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 20 \
                                                       -rollout 12 \
                                                       -task 2 \
                                                       -swap \
                                                       -post -pre \
                                                       -render -verbose \
                                                       -warm 200 \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0.4 \
                                                       -descr tues_adj_swap_targets &
        sleep 5h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

