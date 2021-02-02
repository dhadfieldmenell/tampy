for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robots.hyperparams \
                                                       -no 1 -nt 1 -spl -llus 10000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -retime -vel 0.02 \
                                                       -fail -failmode random \
                                                       -hist_len 2 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0004 -lldec 0.000 -hldec 0.001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -motion 38 \
                                                       -task 2 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -view --load_render \
                                                       -descr again_retimed_with_coverage & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

