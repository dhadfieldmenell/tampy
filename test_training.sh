cd src
python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.test_hyp -no 2 -nt 2 -spl -llus 5000  -hlus 5000  -ff 1. -retime -hln 2 -hldim 64 -lldim 64 -eta 5 -softev -obs_del -hist_len 2 -prim_first_wt 10 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.0001 -add_noop 2 -fail -failmode random -descr test_two_objects & 
sleep 3h
pkill -f run_train -9
sleep 5s

python3 policy_hooks/plot_test.py

python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.test_hyp -no 2 -nt 2 -eta 5 -softev -test namo_objs2_2/test_two_objects_0 -descr two_object_test_run_data -ntest 20

python3 policy_hooks/save_video.py tf_saved/namo_objs2_2/two_object_test_run_data two_object_test_videos

