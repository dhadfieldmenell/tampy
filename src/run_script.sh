for N in 1 2 3 4 5
do
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 -no 1 -nt 1 -spl -eta 10 -softev -descr test_no_graph&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s

    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 -no 1 -nt 1 -spl -eta 10 -softev -ff 1. -descr test_prgraph&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
   
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v80 -no 1 -nt 1 -spl -eta 10 -softev -ff 0.5 -descr test_prgraph_50perc&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
 

done

