for N in 1 2 3 4 5
do
    for T in 2000 3000
    do
        mpirun -np 32 python3 -W ignore policy_hooks/run_training.py \
                -c policy_hooks.namo.hyperparams_v93 \
                -no 1 -nt 1 \
                --expert_path tf_saved/namo_objs1_1/gail_data_6 \
                -baseline gail \
                --traj_limitation ${T} \
                --index ${N} \
                --num_timesteps 5000000 \
                --episode_timesteps 200\
                -descr gail_baseline_1obj_${T}_trajectories & 
        sleep 3h
        pkill -f run_train -9
        sleep 5s
    done
done

