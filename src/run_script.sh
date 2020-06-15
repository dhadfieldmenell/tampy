for N in 1 2 3 4 5
do
    for S in third
    do


        python -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v84 -no 2 -nt 2 -spl -llus 10000  -hlus 10000  -ff 1. -softev -eta 5 -retime -vel 0.3 -hist_len 2 -descr sun_avoiddomain_retime & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


   done
done

