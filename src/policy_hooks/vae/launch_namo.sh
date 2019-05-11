RL=20
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -vae -id 0) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 0) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 1) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 2) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 3) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 4) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 5) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 6) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 7) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 8) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 9) &
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 5 -rl $RL -ncp -sim -id 10) &
