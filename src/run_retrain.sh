for N in 0 1 2 3 4
do
    for D in 32 64 128
    do
        for L in 2
        do
            for LR in 001
            do
                for US in 2000
                do
                    for ETA in 5 8 10
                    do
                        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 \
                                                                      -no 2 -nt 1 -spl \
                                                                      -eta ${ETA} -softev -ff 1. \
                                                                      -hl_retrain \
                                                                      -hlus ${US} \
                                                                      -hln $L -hldim $D -hldec 0.${LR} \
                                                                      -llpol namo_objs1_1/exp_id0_8targ2obj_network2by64$N \
                                                                      -hlpol namo_objs1_1/exp_id0_8targ2obj_network2by64$N \
                                                                      -descr retrain_us${US}_${L}by${D}_lr${LR}_eta${ETA}&
                        sleep 10s
                    done
                done
            done
        done
    done
    wait

done
