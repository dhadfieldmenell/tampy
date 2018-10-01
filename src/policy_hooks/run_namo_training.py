from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.namo.namo_hyperparams import config

master = MultiProcessMain(config)
master.start(kill_all=True)
