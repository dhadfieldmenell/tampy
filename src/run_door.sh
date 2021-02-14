for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.door_hyp \
                                                       -no 1 -nt 1 -spl -llus 5000  -hlus 5000 \
                                                       -ff 1. -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.5 -eta 5 -softev \
                                                       -fail -failmode random \
                                                       -obs_del -hist_len 2 -prim_first_wt 50 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -post -pre \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 20 \
                                                       -task 2 \
                                                       -rollout 0 \
                                                       -roll_hl \
                                                       -descr door_domain & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

