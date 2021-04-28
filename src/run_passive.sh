for N in 1 2 3 4 5
do


        python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -llus 1500  -hlus 4000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -warm 100 \
                                                       -verbose \
                                                       -random \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.55 -dagger_ratio 0.45 -roll_ratio 0.000 \
                                                       -descr random_feedback_problem_distribution & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


done

