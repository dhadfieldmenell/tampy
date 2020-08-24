for N in 1 2 3 4 5
do
    for S in third
    do

        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v91 -no 3 -nt 3 -spl -llus 5000  -hlus 5000  -ff 1. -hln 2 -hldim 64 -lldim 64 -eta 5 -softev -mask -hist_len 2 -fail -failmode random -obs_del -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -add_noop 3 -descr contgoal_failtrain_gripper & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
   done
done

