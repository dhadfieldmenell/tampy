python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 20 -ncp -tv -dist -beta 1 &
python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv  -dist -beta 1 &
