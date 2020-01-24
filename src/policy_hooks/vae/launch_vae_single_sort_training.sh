# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -beta 1 &
# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv  -dist -beta 1 &
(sleep 1 & python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -rnn -beta 1) &
# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -rnn -beta 1 &
# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -rnn -dist -beta 5 &
# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -rnn -over -beta 1 &
(sleep 2 & python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -rnn -over -beta 1) &
# python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -uncond & 
(sleep 3 & python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -ncp -rl 20 -tv -uncond -beta 1) &
