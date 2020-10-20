python3 -W ignore policy_hooks/run_training.py \
	-c policy_hooks.namo.hyperparams_v93 \
	-no 1 -nt 1 \
	-baseline example \
	--index 0 \
	--num_timesteps 5000000 \
	--episode_timesteps 200 \
	-ref_key ex_baseline \
	-run_baseline \
	-descr ex_baseline_1obj
