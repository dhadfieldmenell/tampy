RL=20
python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL -vae -ncp -id 0 &
(sleep 1 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 0) &
(sleep 2 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 1) &
(sleep 3 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 2) &
(sleep 4 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 3) &
(sleep 5 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 4) &
(sleep 6 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 5) &
(sleep 7 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 6) &
(sleep 8 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 7) &
(sleep 9 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 8) &
(sleep 10 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 9) &
(sleep 11 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 10) &
(sleep 12 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 11) &
(sleep 13 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 12) &
(sleep 14 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 13) &
(sleep 15 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 14) &
(sleep 16 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 15) &
(sleep 17 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 16) &
(sleep 18 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 17) &
(sleep 19 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 18) &
(sleep 20 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 19) &
(sleep 21 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 20) &
(sleep 22 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 21) &
(sleep 23 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 22) &
(sleep 24 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 23) &
(sleep 25 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 24) &
(sleep 26 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 25) &
(sleep 27 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 26) &
(sleep 28 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 27) &
(sleep 29 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 28) &
(sleep 30 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 29) &
# (sleep 31 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 30) &
# (sleep 32 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 31) &
# (sleep 32 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 32) &
# (sleep 33 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 33) &
# (sleep 34 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 34) &
# (sleep 35 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 35) &
# (sleep 36 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 36) &
# (sleep 37 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 37) &
# (sleep 38 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 38) &
# (sleep 39 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 39) &
# (sleep 40 ; python policy_hooks/vae/run_training.py -env_path baxter_gym.envs.baxter_block_stack_env -env BaxterLeftBlockStackEnv -rl $RL --rollout_server -ncp -id 40) &
