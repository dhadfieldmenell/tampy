python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v30 -no 1 -nt 1&
sleep 45m
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v30 -no 2 -nt 1&
sleep 1h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v30 -no 2 -nt 2&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v30 -no 3 -nt 2&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

