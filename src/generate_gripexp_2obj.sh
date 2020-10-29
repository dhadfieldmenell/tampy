python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.gripleftgoal_hyp -render -no 2 -nt 2 -spl  -ff 1. -retime -vel 0.3  -save_exp -descr 2obj_gripper_data & 
sleep 10h
pkill -f run_train -9
pkill -f ros -9
sleep 5s

