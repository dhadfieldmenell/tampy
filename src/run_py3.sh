for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v86 \
                                                       -no 2 -nt 2 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 \
                                                       -fail -failmode random \
                                                       -obs_del -hist_len 2 -prim_first_wt 1 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.001 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -post -pre \
                                                       -expl_wt 50 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 2 \
                                                       -rollout 12 \
                                                       -roll_hl \
                                                       --load_render \
                                                       -descr rel_post_continuous_ctrl & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

