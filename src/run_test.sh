for N in 1
do
    python -W ignore policy_hooks/run_training.py -no 1 -nt 1 -softev -test namo_objs1_1/exp_id0_qlearn_base_1by40
done

