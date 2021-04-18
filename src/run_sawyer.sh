for N in 1 2 3 4 5
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \
                                                       -no 1 -nt 1 -llus 2000  -hlus 6000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 20 \
                                                       -task 2 \
                                                       -rollout 10 \
                                                       --load_render \
                                                       -warm 200 \
						       -verbose \
						       -neg_pre -neg_post \
						       -neg_ratio 0.05 -opt_ratio 0.475 -dagger_ratio 0.475 -roll_ratio 0.000 \
                                                       -descr random_init_fiveperc_negative_dagger& 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \
                                                       -no 1 -nt 1 -llus 2000  -hlus 6000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 20 \
                                                       -task 2 \
                                                       -rollout 10 \
                                                       --load_render \
                                                       -warm 200 \
						       -verbose \
						       -neg_pre -neg_post \
						       -neg_ratio 0.0 -opt_ratio 0.5 -dagger_ratio 0.5 -roll_ratio 0.000 \
                                                       -descr random_init_dagger& 
        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s



done

