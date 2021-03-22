for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000 -hlus 20000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.4 -eta 7 -softev \
                                                       -obs_del -hist_len 1 \
                                                       -prim_first_wt 10 -lr 0.0004 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.000 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -render -hl_image \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 5 -expl_eta 5 \
                                                       -col_coeff 0.1 \
                                                       -motion 26 \
                                                       -task 2 \
                                                       -rollout 10 \
                                                       -pre -post -mid \
                                                       -warm 200 \
                                                       -descr verify_fp_sat_redo_nostate_image_hl & 

        sleep 7h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
       
    done
done

