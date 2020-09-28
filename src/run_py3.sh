for N in 1 2 3 4 5
do
    for S in third
    do
        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v86 -no 2 -nt 2 -spl -llus 5000  -hlus 5000  -retime -ff 1. -hln 2 -hldim 64 -lldim 64 -prim_first_wt 1 -eta 5 -softev -mask -hist_len 3 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 2 -hl_image -permute 0 -fail -failmode random -obs_del -descr image_input & 
        sleep 3h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s
    done
done

