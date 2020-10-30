for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 -no 3 -nt 3 -spl -llus 10000  -hlus 10000  -ff 1. -hln 2 -hldim 64 -lldim 64 -retime -vel 0.3 -mask -eta 5 -softev -hist_len 2 -prim_first_wt 20 -lr 0.0002 -hllr 0.0002 -lldec 0.0001 -hldec 0.0001 -add_noop 4 --permute_hl 1 -roll_post -expl_wt 5 -expl_eta 4 -col_coeff 0.1 -descr adjusteta_gripper & 
        sleep 8h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

