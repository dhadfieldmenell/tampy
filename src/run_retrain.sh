for N in 0 1 2 3 4
do
    for D in 32 64 128
    do
        for L in 2 4
        do
            python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 \
                                                          -no 2 -nt 1 -spl \
                                                          -eta 5 -softev -ff 1. \
                                                          -hl_retrain \
                                                          -hln $L -hldim $D -hldec 0.001 \
                                                          -llpol namo_objs2_1/exp_id0_8targ2obj_network2by64$N \
                                                          -hlpol namo_objs2_1/exp_id0_8targ2obj_network2by64$N \
                                                          -descr retrain_${L}by${D}_lr001_eta5&
            sleep 10s

        done
    done
    wait

done
