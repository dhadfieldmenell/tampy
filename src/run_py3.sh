for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 \
                                                       -no 3 -nt 3 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 50 \
                                                       -task 2 \
                                                       -rollout 0 \
                                                       -roll_hl \
                                                       -render \
                                                       -descr verify_3obj_fixedstart_retime_grip & 
        sleep 1h 40m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 \
                                                       -no 3 -nt 3 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -vel 0.3 -eta 5 -softev \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -post -pre -mid \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 40 \
                                                       -task 2 \
                                                       -rollout 20 \
                                                       -roll_hl \
                                                       -render \
                                                       -descr verify_3obj_fixedstart_supervised_dagger_grip & 
        sleep 1h 40m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 \
                                                       -no 3 -nt 3 -spl -llus 5000  -hlus 10000 \
                                                       -ff 1. -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -vel 0.3 -eta 5 -softev \
                                                       -fail -failmode random \
                                                       -lr_policy adaptive \
                                                       -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 40 \
                                                       -task 2 \
                                                       -rollout 20 \
                                                       -roll_hl \
                                                       -render \
                                                       -descr verify_3obj_fixedstart_random_dagger_grip & 
        sleep 1h 40m
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
    done
done

