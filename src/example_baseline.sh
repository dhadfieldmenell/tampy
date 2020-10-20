for N in 1 2 3 4 5
do
    for T in 2000 3000
    do
        python3 -W ignore policy_hooks/run_training.py \
                -c policy_hooks.namo.hyperparams_v93 \
                -no 1 -nt 1 \
                -baseline example \
                --traj_limitation ${T} \
                --index ${N} \
                --num_timesteps 5000000 \
                --episode_timesteps 200 \
		-ref_key ex_baseline \
		-run_baseline \
                -descr ex_baseline_1obj_${T}_trajectories
    done
done

