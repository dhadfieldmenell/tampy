for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v86 \
                                                       -no 2 -nt 2 -spl -llus 5000  -hlus 5000 \
                                                       -ff 1. -mask -hln 2 -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -fail -failmode random \
                                                       -hist_len 2 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0001 \
                                                       -add_noop 3 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.005 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr base & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

