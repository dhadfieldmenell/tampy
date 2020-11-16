for N in 0
do
    for D in 32 128
    do
        for US in 5000
        do
            for LR in 0001 001
            do
                for IT in 1000
                do
                    for DEC in 0
                    do
                        for ETA in 5
                        do
                            python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v93 \
                                                                          -no 2 -nt 1 -spl \
                                                                          -eta ${ETA} -softev -ff 1. \
                                                                          -hl_only_retrain \
                                                                          -hlus ${US} \
                                                                          -iters ${IT} \
                                                                          -lr 0.${LR} \
                                                                          --batch_size 100 \
                                                                          -hln 2 -hldim $D -hldec 0.${DEC} \
                                                                          -llpol namo_objs2_1/exp_id0_valcheck_2 \
                                                                          -hldata namo_objs2_1/exp_id0_newff_data_gen_0 \
                                                                          -descr taskpureloss_us${US}_IT${IT}_2by${D}_dec${DEC}_lr${LR}_eta${ETA}_${N}&
                            sleep 5s
                            python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v91 \
                                                                          -no 2 -nt 1 -spl \
                                                                          -eta ${ETA} -softev -ff 1. \
                                                                          -hl_only_retrain \
                                                                          -hlus ${US} \
                                                                          -iters ${IT} \
                                                                          -lr 0.${LR} \
                                                                          --batch_size 100 \
                                                                          -hln 2 -hldim $D -hldec 0.${DEC} \
                                                                          -llpol namo_objs2_1/exp_id0_valcheck_2 \
                                                                          -hldata namo_objs2_1/exp_id0_newff_data_gen_0 \
                                                                          -descr goalpureloss_us${US}_IT${IT}_2by${D}_dec${DEC}_lr${LR}_eta${ETA}_${N}&
                            sleep 5s
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
                                                                          -hldata namo_objs2_1/exp_id0_newff_data_gen_0 \
                                                                          -descr plainpureloss_us${US}_IT${IT}_2by${D}_dec${DEC}_lr${LR}_eta${ETA}_${N}&
                            sleep 5s
                        done
                    done
                done
            done
        done
    done
done
wait
