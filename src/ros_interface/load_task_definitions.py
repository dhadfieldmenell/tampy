from core.parsing import parse_domain_config, parse_problem_config
from pma.hl_solver import FFSolver


def plan_from_str(ll_plan_str, num_cloth):
    '''Convert a plan string (low level) into a plan object.'''
    domain_fname = '../domains/laundry_domain/laundry_policy.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_{0}.prob'.format(num_cloth))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    plan = hls.get_plan(ll_plan_str, domain, problem)
    return plan

def get_tasks(self, file_name):
    with open(file_name) as f:
        task_map = eval(f.read())
    return task_map
    
def replace_param(plan_str, old_param_name, new_param_name):
    return plan_str.replace(old_param_name, new_param_name)
