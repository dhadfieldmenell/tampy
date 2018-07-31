import main

from core.parsing import parse_domain_config, parse_problem_config
from pma.hl_solver import FFSolver


def plan_from_str(ll_plan_str, num_cloth):
    '''Convert a plan string (low level) into a plan object.'''
    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_{0}.prob'.format(num_cloth))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    plan = hls.get_plan(ll_plan_str, domain, problem)
    return plan

def get_tasks(file_name):
    with open(file_name, 'r+') as f:
        task_info = f.read().split(';')
        task_map = eval(task_info[0])
    return task_map
    
def get_task_durations(file_name):
    with open(file_name, 'r+') as f:
        task_info = f.read().split(';')
        task_durations = eval(task_info[1])
    return task_durations

def fill_params(plan_str, values):
    for i in range(len(plan_str)):
        plan_str[i] = plan_str[i].format(*values.append(i))
    return plan_str

def get_task_encoding(task_list):
    encoding = {}
    for i in range(len(task_list)):
        encoding[task_list[i]] = i

    return encoding

def compare_task_states(state1, state2):
    num_states = 0.
    num_states_match = 0.
    for key in state1:
        if key not in state2: continue
        num_states += 1.
        if state1[key] -- state2[key]:
            num_states_match += 1.

    return num_states_match / num_states
