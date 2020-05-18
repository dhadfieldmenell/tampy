for N in 1 2 3 4 5
do
 
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v87 -no 2 -nt 1 -spl -eta 10 -softev -ff 1. -descr dummycompare_compact&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s
    
    python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v88 -no 2 -nt 1 -spl -eta 10 -softev -ff 1. -descr dummycompare_spread&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 5s

done

