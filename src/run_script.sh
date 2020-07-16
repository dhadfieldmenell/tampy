for N in 1 2 3 4 5
do
    for S in third
    do

        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v90 -no 1 -nt 1 -spl -llus 1000  -hlus 1000  -ff 1. -hln 3 -hldim 64 -lldim 64 -eta 5 -softev -vel 0.3 -hist_len 3 -fail -failmode random -obs_del -descr redo & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


   done
done

