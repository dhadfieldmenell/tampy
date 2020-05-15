for N in 1 5 10
do
   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -descr eta$N  -eta $N -softev -ff 1. -test namo_objs2_1/exp_id0_8targ2obj_network2by640&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
 
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -descr eta$N  -eta $N -softev -ff 1. -test namo_objs2_1/exp_id0_8targ2obj_network2by641&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
  
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -descr eta$N  -eta $N -softev -ff 1. -test namo_objs2_1/exp_id0_8targ2obj_network2by642&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -descr eta$N  -eta $N -softev -ff 1. -test namo_objs2_1/exp_id0_8targ2obj_network2by643&
    sleep 10m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
 

done

