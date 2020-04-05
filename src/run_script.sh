python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 1 -neg&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 1&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 2&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 2 -cur 10 -ncur 5&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 1 -q&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 1 -her&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v10 -no 2 -nt 2 -her&
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

