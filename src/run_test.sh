python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 2 -nt 2 -spl -llus 5000  -hlus 5000  -ff 1. -hln 2 -hldim 64 -lldim 64 -eta 3 -softev -mask -hist_len 3 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -add_noop 4 -fail -failmode random -obs_del -test namo_objs2_2/polresample_N10_s5_0 -descr test2witheta -ntest 20

