for N in 1 2 3 4 5
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robodesk.sort_hyp \
                                                       -no 1 -llus 1200  -hlus 2000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 96 -lldim 48 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 1 -lr 0.0004 \
                                                       -hllr 0.0004 -lldec 0.000001 -hldec 0.000001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -image -hl_image \
                                                       -imwidth 84 -imheight 84 \
                                                       -batch 64 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 18 \
                                                       -task 2 \
                                                       -rollout 8 \
                                                       -warm 300 \
                                                       -verbose \
                                                       -n_resample 4 \
                                                       --load_render \
                                                       -post -pre \
                                                       -roll_opt \
                                                       -neg_pre -neg_post \
                                                       -trans_obs -imchannels 6 \
                                                       -neg_ratio 0.0 -opt_ratio 0.5 -dagger_ratio 0.5 -roll_ratio 0.000 \
                                                       -descr robodesk_tues_trans_obs_lift_ball & 
        sleep 6h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

done

