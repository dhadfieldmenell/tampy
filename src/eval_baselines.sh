python3 -W ignore policy_hooks/run_training.py \
        -c policy_hooks.namo.hyperparams_v93 \
        -no 1 -nt 1 \
        --expert_path tf_saved/namo_objs1_1/exp_id0_data_gen_0/ \
        -baseline gail \
        --traj_limitation 10 \
        --index 0 \
        -ref_key gail_baseline \
        --task evaluate \
        --num_timesteps 500000 \
        --episode_timesteps 200\
        -descr gail_baseline_1obj_trajectories

