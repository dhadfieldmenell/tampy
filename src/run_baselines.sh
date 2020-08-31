for N in 1 2 3 4 5
do
    for T in 50 100 500 700
    do
        python3 -W ignore policy_hooks/run_training.py \
                -c policy_hooks.namo.hyperparams_v93 \
                -no 3 -nt 3 \
                --expert_path tf_saved/namo_objs3_3/exp_id0_gripper_dom_${N}/ \
                -baseline gail \
                --traj_limitation ${T} \
                --index ${N} \
                -descr gail_baseline_3obj_${T}_trajectories & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
    done
done

