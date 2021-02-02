for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v88 \
                                                       -no 2 -nt 2 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -hln 2 -mask -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.0001 \
                                                       -add_noop 3 --permute_hl 0 \
                                                       -post -pre -hl_image -render \
                                                       -imwidth 64 -imheight 64 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr aux_not_fed_through_images_64_hl & 

        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v87 \
                                                       -no 2 -nt 2 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -hln 2 -mask -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.0001 \
                                                       -add_noop 3 --permute_hl 0 \
                                                       -post -pre -hl_image -render \
                                                       -imwidth 64 -imheight 64 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr aux_endpos_images_64_hl & 

        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

