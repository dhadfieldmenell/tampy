for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 -no 2 -nt 2 -spl -eta 10 -softev -ff 1. -descr test_prgraph&
    sleep 2h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 -no 2 -nt 2 -spl -eta 10 -softev -ff 0.5 -descr test_prgraph_50perc&
    sleep 2h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
 

done

