for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v87 \
                                                       -no 2 -nt 2 -spl -llus 5000  -hlus 20000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.4 -eta 7 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 50 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.0002 \
                                                       -add_noop 3 --permute_hl 1 \
                                                       -post -pre -hl_image -render \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr permute_images_96_hl & 
        sleep 10h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done
