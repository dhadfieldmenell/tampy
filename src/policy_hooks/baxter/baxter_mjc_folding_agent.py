from policy_hooks.baxter.baxter_mjc_agent import BaxterMJCAgent
from policy_hooks.baxter.baxter_mjc_env import BaxterMJCEnv
from policy_hooks.tamp_agent import TAMPAgent
from policy_hooks.utils.mjc_xml_utils import *

class BaxterMJCFoldingAgent(BaxterMJCAgent):
    def __init__(self, hyperparams):
        plans = hyperparams['plans']
        table = plans.values()[0].params['table']
        table_info = get_param_xml(table)

        cloth_wid = hyperparams['cloth_width']
        cloth_len = hyperparams['cloth_length']

        folding_cloth = get_deformable_cloth(cloth_wid, cloth_len)
        self.env = BaxterMJCEnv(items=[table, folding_cloth])
        self.cur_t = 0
        self.cur_task_ind = 0
        super(TAMPAgent, self).__init__(hyperparams)
