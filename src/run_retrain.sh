for N in 0 
do
    for D in 32
    do
        for US in 5000
        do
            for LR in 0001 001
            do
                for IT in 100 1000 10000
                do
                    for DEC in 01 001
                    do
                        for ETA in 5 10
                        do
                            python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 \
                                                                          -no 2 -nt 1 -spl \
                                                                          -eta ${ETA} -softev -ff 1. \
                                                                          -hl_only_retrain \
                                                                          -hlus ${US} \
                                                                          -iters ${IT} \
                                                                          -lr 0.${LR} \
                                                                          --batch_size 100 \
                                                                          -hln 2 -hldim $D -hldec 0.${DEC} \
                                                                          -llpol namo_objs2_1/exp_id0_valcheck_2 \
                                                                          -hldata namo_objs2_1/exp_id0_valcheck_2 \
                                                                          -descr newretrain_us${US}_IT${IT}_2by${D}_dec${DEC}_lr${LR}_eta${ETA}_${N}&
                            sleep 5s
                        done
                    done
                done
            done
        done
    done
done
wait
