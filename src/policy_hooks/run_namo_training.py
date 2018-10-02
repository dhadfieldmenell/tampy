from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain
from policy_hooks.namo.namo_hyperparams import config

pretrain = MultiProcessPretrainMain(config)
pretrain.run()

# master = MultiProcessMain(config)
# master.start(kill_all=True)
