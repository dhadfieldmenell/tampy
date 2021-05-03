for N in 1
do
    for S in third
    do


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.flathyp \
                                                       -no 1 -llus 5000  -hlus 5000 \
                                                       -spl -mask -hln 2 -hldim 96 -lldim 64 \
                                                       -retime -vel 0.3 -eta 5 -softev \
                                                       -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.0 \
                                                       -motion 18 \
                                                       -rollout 12 \
                                                       -task 2 \
                                                       -post -pre \
                                                       -warm 100 \
                                                       -run_baseline \
                                                       -baseline stable \
                                                       --total_timesteps 100000 \
                                                       --n_proc 30 \
                                                       --algo ppo2 \
                                                       -descr 2d_pickplace_1obj_PPO2_baseline

    done
done

