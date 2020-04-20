for N in 1 2 3
do
    python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v40 -no 1 -nt 1 -spl -soft -descr lowlevel_prev_nosplit_hlsoft&
    sleep 1h
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
done

