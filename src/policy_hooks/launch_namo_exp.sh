tmux new-session -d -s "mcts" "python policy_hooks/run_training.py -c namo.namo_hyperparams -mcts"
tmux new-session -d -s "pol" "python policy_hooks/run_training.py -c namo.namo_hyperparams -pol"
tmux new-session -d -s "mp" "python policy_hooks/run_training.py -c namo.namo_hyperparams -mp"

