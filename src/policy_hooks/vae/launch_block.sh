RL=20
python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -vae -ncp -id 0 &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 0) &
(sleep 2 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 1) &
(sleep 3 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 2) &
(sleep 4 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 3) &
(sleep 5 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 4) &
(sleep 6 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 5) &
(sleep 7 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 6) &
(sleep 8 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 7) &
(sleep 9 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 8) &
(sleep 10 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 9) &

(sleep 21 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 20) &
(sleep 22 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 21) &
(sleep 23 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 22) &
(sleep 24 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 23) &
(sleep 25 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 24) &
(sleep 26 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 25) &
(sleep 27 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 26) &
(sleep 28 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 27) &
(sleep 29 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 28) &
(sleep 20 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 29) &

(sleep 31 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 30) &
(sleep 22 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 31) &
(sleep 33 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 32) &
(sleep 34 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 33) &
(sleep 35 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 34) &
(sleep 36 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 35) &
(sleep 37 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 36) &
(sleep 38 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 37) &
(sleep 39 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 38) &
(sleep 30 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterBlockStackEnv -rl $RL -sim -ncp -id 39) &
