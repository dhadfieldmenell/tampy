for N in 1 2 3 4 5
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robodesk.inshelf_hyp \
                                                       -no 1 -llus 4000  -hlus 12000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 96 -lldim 96 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 3 -lr 0.0001 \
                                                       -hllr 0.0001 -lldec 0.000001 -hldec 0.00001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -image -hl_image \
                                                       -imwidth 112 -imheight 112 \
                                                       -batch 50 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 24 \
                                                       -task 2 \
                                                       -rollout 4 \
                                                       -warm 300 \
                                                       -verbose \
                                                       --load_render \
                                                       -post -pre \
                                                       -neg_pre -neg_post \
                                                       -roll_opt \
                                                       -neg_ratio 0.0 -opt_ratio 0.75 -dagger_ratio 0.25 -roll_ratio 0.000 \
                                                       -descr robodesk_inshelf_with_grip_cam & 
        sleep 10h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

done
