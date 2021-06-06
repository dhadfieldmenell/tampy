for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000 -hlus 20000 \
                                                       -hl_retrain --hl_data namo_objs2_2/verify_fp_mixed_nofc_sun_image_hl_0 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -obs_del -hist_len 1 \
                                                       -prim_first_wt 5 -lr 0.0004 \
                                                       -hllr 0.0002 -lldec 0.0000 -hldec 0.0004 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -render -hl_image \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 0 \
                                                       -task 0 \
                                                       -rollout 0 \
                                                       -pre -post \
                                                       -warm 200 \
                                                       -descr retrain_image 

        sleep 7h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
       
    done
done

