for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.door_hyp \
                                                       -no 1 -nt 1 -llus 10000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 64 -lldim 64 \
                                                       -retime -vel 0.4 -eta 5 -softev \
                                                       -fail -failmode random \
                                                       -obs_del -hist_len 3 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.0004 -lldec 0.000 -hldec 0.000 \
                                                       -add_noop 3 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 20 \
                                                       -task 2 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -load_render \
                                                       -descr adj_door_domain & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

