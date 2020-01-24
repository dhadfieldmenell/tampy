# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 25 -ncp -tv -beta 1 &
# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 25 -ncp -tv -dist -beta 1 &
# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 25 -ncp -tv -rnn -over -beta 1 &
(sleep 1 & python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 20 -ncp -tv -rnn -beta 1) &
(sleep 2 & python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 20 -ncp -tv -rnn -over -beta 1) &
# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 20 -ncp -tv -rnn -dist -beta 1 &
# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 25 -ncp -tv -rnn -beta 1 &
# python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 25 -ncp -tv -uncond -beta 1 &
(sleep 3 & python policy_hooks/vae/run_training.py --config namo.namo_hyperparams -n 1 -o 4 -rl 20 -ncp -tv -uncond -beta 1) &
