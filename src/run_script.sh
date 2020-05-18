for N in 1 2 3 4 5
do
    for M in base random end
    do
        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v81 -no 2 -nt 1 -spl -eta 5 -softev -hln 2 -hldim 32 -ff 1. -descr ${M}_compact -x_select ${M}& 
        sleep 1h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
    done

done

