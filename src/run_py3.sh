for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v86 -no 4 -nt 4 -spl -llus 10000  -hlus 10000  -ff 1. -hln 2 -hldim 64 -retime -vel 0.25 -lldim 64 -mask -eta 5 -softev -hist_len 2 -prim_first_wt 10 -lr 0.0005 -hllr 0.001 -lldec 0.0001 -hldec 0.001 -add_noop 2 --permute_hl 1 -fail -failmode random -expl_wt 10 -expl_eta 4 -expl_n 15 -expl_suc 8 -expl_m 4 -col_coeff 0.1 -descr col01_singlegrasp_gridtargets & 
        sleep 8h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

