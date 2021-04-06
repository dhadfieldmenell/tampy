for N in 1 2 3 4 5
do
    for S in third
    do

    
        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 3 -spl -llus 10000 -hlus 20000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -hist_len 2 \
                                                       -end2end 0. \
                                                       -prim_first_wt 10 -lr 0.0004 -hllr 0.0004 \
                                                       -lldec 0.0001 -hldec 0.000 -contdec 0.0001 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -render -hl_image \
                                                       -imwidth 112 -imheight 112 \
                                                       -expl_wt 5 -expl_eta 5 \
                                                       -col_coeff 0.0 \
                                                       -motion 18 \
                                                       -task 2 \
                                                       -rollout 6 \
                                                       -pre -post \
                                                       -warm 200 \
                                                       -backup \
                                                       -descr verify_withcont_dec_image_backup & 

        sleep 2h 30m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 3 -spl -llus 10000 -hlus 20000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -hist_len 2 \
                                                       -end2end 0. \
                                                       -prim_first_wt 10 -lr 0.0004 -hllr 0.0004 \
                                                       -lldec 0.0001 -hldec 0.000 -contdec 0.0001 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -render -hl_image \
                                                       -imwidth 112 -imheight 112 \
                                                       -expl_wt 5 -expl_eta 5 \
                                                       -col_coeff 0.0 \
                                                       -motion 18 \
                                                       -task 2 \
                                                       -rollout 6 \
                                                       -pre -post \
                                                       -warm 200 \
                                                       -descr verify_withcont_dec_image_nobackup & 

        sleep 2h 30m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
 

    done
done

