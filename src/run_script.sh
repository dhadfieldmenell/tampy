python -W ignore policy_hooks/run_training.py -f policy_hooks/namo/runexp.txt &
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -f policy_hooks/namo/runexp2.txt &
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -f policy_hooks/namo/runexp3.txt &
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

python -W ignore policy_hooks/run_training.py -f policy_hooks/namo/runexp4.txt &
sleep 2h
pkill -f run_train -9
pkill -f ros -9
sleep 10s

