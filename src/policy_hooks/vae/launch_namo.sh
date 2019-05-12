RL=20
(sleep 1 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -vae -id 0) &
(sleep 2 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 0) &
(sleep 3 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 1) &
(sleep 4 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 2) &
(sleep 5 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 3) &
(sleep 6 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 4) &
(sleep 7 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 5) &
(sleep 8 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 6) &
(sleep 9 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 7) &
(sleep 10 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 8) &
(sleep 11 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 9) &
(sleep 12 ; python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl $RL -ncp -sim -id 10) &
