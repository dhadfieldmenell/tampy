for N in 1 2 3 4 5
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training_spawn.py -c policy_hooks.namo.flathyp \
                                                       -no 1 -llus 5000  -hlus 5000 \
                                                       -spl -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -hist_len 1 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -col_coeff 0.0 \
                                                       -motion 0 \
                                                       -rollout 0 \
                                                       -task 0 \
                                                       -warm 100 \
                                                       -run_baseline \
                                                       -baseline stable \
                                                       --total_timesteps 4000000 \
                                                       --n_proc 60 \
                                                       -flat \
                                                       --algo ppo2 \
                                                       -descr 2d_pickplace_1obj_baseline_PPO2 &

        sleep 9h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training_spawn.py -c policy_hooks.namo.flathyp \
                                                       -no 2 -llus 5000  -hlus 5000 \
                                                       -spl -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -hist_len 1 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -col_coeff 0.0 \
                                                       -motion 0 \
                                                       -rollout 0 \
                                                       -task 0 \
                                                       -warm 100 \
                                                       -run_baseline \
                                                       -baseline stable \
                                                       --total_timesteps 4000000 \
                                                       --n_proc 60 \
                                                       -flat \
                                                       --algo ppo2 \
                                                       -descr 2d_pickplace_2obj_baseline_PPO2 &

        sleep 9h 
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

