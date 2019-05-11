RL=20
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -vae -ncp -id 0) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 0) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 1) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 2) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 3) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 4) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 5) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 6) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 7) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 8) &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 9) &
