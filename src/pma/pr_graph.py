from IPython import embed as shell
from core.parse_config_to_solvers import ParseConfigToSolvers
from core.parse_config_to_domain import ParseConfigToDomain
from core.parse_config_to_problem import ParseConfigToProblem
from Queue import PriorityQueue
from prg_search_node import HLSearchNode, LLSearchNode

"""
Many methods called in p_mod_abs have detailed documentation.
"""
def p_mod_abs(domain_config, problem_config, solvers_config, max_iter=100):
    hl_solver, ll_solver = ParseConfigToSolvers(solvers_config, domain_config).parse()
    domain = ParseConfigToDomain(domain_config).parse()
    problem = ParseConfigToProblem(problem_config, domain).parse()
    if problem.goal_test():
        print "Goal is already satisfied. No planning done."
        return False
    n0 = HLSearchNode(hl_solver.translate_problem(problem), problem)

    Q = PriorityQueue()
    Q.put((0, n0))
    for _ in range(max_iter):
        n = Q.get()[1]
        if n.is_hl_node():
            c_plan = n.plan(hl_solver)
            c = LLSearchNode(c_plan)
            Q.put((-n.heuristic(), n))
            Q.put((-c.heuristic(), c))
        elif n.is_ll_node():
            n.plan(ll_solver)
            if n.solved():
                return n.curr_plan
            Q.put((-n.heuristic(), n))
            if n.gen_child():
                fail_step, fail_pred = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred)
                c = HLSearchNode(hl_solver.translate_problem(n_problem), n_problem, prefix=n.curr_plan.prefix(fail_step))
                Q.put((-c.heuristic(), c))

    return False
