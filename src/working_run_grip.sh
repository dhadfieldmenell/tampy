for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000 -hlus 50000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -obs_del -hist_len 2 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.000 -hldec 0.00 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -post -pre -render -hl_image \
                                                       -imwidth 128 -imheight 128 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 8 \
                                                       -rollout 12 \
                                                       -roll_hl \
                                                       -warm 200 \
                                                       -descr monday_slower_permute_conttask_grip_images_128_hl & 

        sleep 6h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

