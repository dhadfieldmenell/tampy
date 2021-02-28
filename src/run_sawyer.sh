for N in 1 2 3 4 5
do
    for S in third
    do

        python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_grasp_hyp \
                                                       -no 1 -nt 1 -llus 3000  -hlus 3000 \
                                                       -ff 1. -mask -hln 2 -hldim 64 -lldim 96 \
                                                       -eta 5 \
                                                       -goal_type grasp \
                                                       -vel 0.02 \
                                                       -hist_len 1 -prim_first_wt 1 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.00 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -post -pre -mid \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 30 \
                                                       -task 2 \
                                                       -rollout 30 \
                                                       -roll_hl \
                                                       --load_render \
                                                       -warm 200 \
                                                       -descr sat_2_noprec_jnt_sawyer_grasp & 
        sleep 4h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

