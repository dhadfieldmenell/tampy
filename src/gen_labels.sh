for N in 1 2 3
do


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1200  -hlus 2400 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -ref_dir experiment_logs/michael_labels \
                                                       -warm 50 \
                                                       -max_label 100 \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_100_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1000  -hlus 2000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -ref_dir experiment_logs/ananya_labels \
                                                       -warm 50 \
                                                       -max_label 100 \
                                                       -verbose \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_100_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1200  -hlus 2400 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -max_label 100 \
                                                       -ref_dir experiment_logs/cassidy_labels \
                                                       -warm 50 \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_100_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s



	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1200  -hlus 2400 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -ref_dir experiment_logs/michael_labels \
                                                       -warm 50 \
                                                       -max_label 200 \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_200_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1000  -hlus 2000 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -ref_dir experiment_logs/ananya_labels \
                                                       -warm 50 \
                                                       -max_label 200 \
                                                       -verbose \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_200_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


	python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.pick_hyp \
                                                       -no 1 -nt 1 -llus 1200  -hlus 2400 \
                                                       -ff 1. -mask -hln 2 -lln 2 -hldim 64 -lldim 64 \
                                                       -eta 5 \
                                                       -hist_len 1 -prim_first_wt 5 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.000001 -hldec 0.0001 \
                                                       -add_noop 0 --permute_hl 0 \
                                                       -expl_wt 5 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 14 \
                                                       -max_label 200 \
                                                       -ref_dir experiment_logs/cassidy_labels \
                                                       -warm 50 \
                                                       --load_render \
                                                       -neg_ratio 0.0 -opt_ratio 0.6 -dagger_ratio 0. -roll_ratio 0.000 -human_ratio 0.4 \
                                                       -descr offline_with_200_human_labels & 
        sleep 2h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


done

