for N in 1 2 3 4 5
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.holdout_hyp \
                                                       -no 4 -llus 4000  -hlus 8000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 18 \
                                                       -task 2 \
                                                       -rollout 10 \
                                                       -warm 100 \
                                                       -verbose \
                                                       --load_render \
                                                       -neg_pre -neg_post \
                                                       -pre -post \
                                                       -roll_opt \
                                                       -neg_ratio 0.02 -opt_ratio 0.5 -dagger_ratio 0.48 -roll_ratio 0.000 \
                                                       -descr holdout_obj_pairs_sawyer_4objs & 
        sleep 6h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

done

