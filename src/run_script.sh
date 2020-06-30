for N in 1 2 3 4 5
do
    for S in third
    do

        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v90 -no 2 -nt 2 -spl -llus 10000  -hlus 10000  -ff 1. -hln 3 -hldim 64 -lldim 64 -eta 5 -softev -retime -vel 0.3 -hist_len 3 -obs_del -descr mjclidar & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


   done
done

