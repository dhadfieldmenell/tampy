for N in 1 2 3 4 5
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \
                                                       -no 1 -nt 1 -llus 3000  -hlus 3000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -lr_policy adaptive \
                                                       -hist_len 2 -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -neg -neg_pre -neg_post \
                                                       -motion 20 \
                                                       -task 2 \
                                                       -rollout 8 \
                                                       --load_render \
                                                       -warm 200 \
                                                       -backup -verbose \
                                                       -descr wed_backup_dagger& 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \
                                                       -no 1 -nt 1 -llus 3000  -hlus 3000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -lr_policy adaptive \
                                                       -hist_len 2 -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 20 \
                                                       -neg -neg_pre -neg_post \
                                                       -task 2 \
                                                       -rollout 8 \
                                                       --load_render \
                                                       -warm 200 \
                                                       -descr wed_nobackup_dagger& 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


done

