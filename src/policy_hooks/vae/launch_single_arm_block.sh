RL=20
python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -vae -ncp -id 0 &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 0) &
(sleep 2 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 1) &
(sleep 3 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 2) &
(sleep 4 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 3) &
(sleep 5 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 4) &
(sleep 6 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 5) &
(sleep 7 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 6) &
(sleep 8 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 7) &
(sleep 9 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 8) &
(sleep 10 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -sim -ncp -id 9) &
