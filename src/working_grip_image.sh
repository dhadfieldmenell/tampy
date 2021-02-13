for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000  -hlus 10000 \
                                                       -ff 1. -hln 3 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.5 -eta 5 \
                                                       -obs_del -hist_len 3 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.000 \
                                                       -add_noop 4 --permute_hl 0 \
                                                       -post -render -hl_image \
                                                       -imwidth 64 -imheight 64 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr redo_conttask_grip_images_64_hl & 

        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000  -hlus 20000 \
                                                       -ff 1. -hln 2 -mask -hldim 96 -lldim 64 \
                                                       -retime -vel 0.6 -eta 5 \
                                                       -obs_del -hist_len 3 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0001 -lldec 0.0001 -hldec 0.0005 \
                                                       -add_noop 3 --permute_hl 0 \
                                                       -post -render -hl_image \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 36 \
                                                       -task 4 \
                                                       -rollout 0 \
                                                       -roll_hl \
                                                       -descr alt_conttask_grip_images_96_hl & 

        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


    done
done

