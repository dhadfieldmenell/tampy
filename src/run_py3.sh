for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 -no 2 -nt 2 -spl -llus 5000  -hlus 5000  -ff 1. -hln 2 -hldim 64 -lldim 64 -retime -vel 0.25 -mask -eta 5 -softev -hist_len 2 -prim_first_wt 100 -lr 0.0002 -hllr 0.0004 -lldec 0.0001 -hldec 0.0002 -add_noop 5 --permute_hl 1 -fail --fail_mode random -roll_post -expl_wt 5 -expl_eta 4 -col_coeff 0.01 -descr two_object_gripper & 
        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

