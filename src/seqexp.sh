python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                       -no 2 -spl -llus 10000 -hlus 15000 \
                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                       -retime -vel 0.3 -eta 7 -softev \
                                       -obs_del -hist_len 2 \
                                       -end2end 0. \
                                       -prim_first_wt 5 -lr 0.0002 -hllr 0.0004 \
                                       -lldec 0.0001 -hldec 0.0000 -contdec 0.00001 \
                                       -add_noop 0 --permute_hl 0 \
                                       -render -hl_image -cont_image \
                                       -imwidth 80 -imheight 80 \
                                       -expl_wt 5 -expl_eta 5 \
                                       -col_coeff 0.05 \
                                       -motion 22 \
                                       -task 4 \
                                       -rollout 10 \
                                       -pre -post \
                                       -warm 200 \
                                       -verbose \
                                       -swap \
                                       -seq \
                                       -neg_ratio 0.0 -opt_ratio 0.5 -dagger_ratio 0.5 \
                                       -descr sun_fullset_noneg_image_swap_3obj 



