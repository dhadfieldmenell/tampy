from core.internal_repr.plan import Plan
from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.baxter_constants as const
from pma.hl_solver import FFSolver
from pma.robot_ll_solver import RobotLLSolver
import ros_interface.utils as utils


def plan_from_str(ll_plan_str):
    '''Convert a plan string (low level) into a plan object.'''
    domain_fname = '../domains/laundry_domain/laundry_policy.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_1.prob')
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    plan = hls.get_plan(ll_plan_str, domain, problem)
    return plan

def get_task(self, task_name):
	with open(task_name) as f:
		task_str = f.read().split("\n")
	return plan_from_str(task_str)

def get_tasks(self, file_name):
	tasks = []
	with open(file_name) as f:
		task_map = eval(f.read())
		for task in task_map:
			task.append((task, plan_from_str(task_map[task])))
	return tasks
	