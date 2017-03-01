from IPython import embed as shell
from core.parsing.parse_solvers_config import ParseSolversConfig
from core.parsing.parse_domain_config import ParseDomainConfig
from core.parsing.parse_problem_config import ParseProblemConfig
from Queue import PriorityQueue
from prg_search_node import HLSearchNode, LLSearchNode

"""
Many methods called in p_mod_abs have detailed documentation.
"""
def p_mod_abs(domain_config, problem_config, solvers_config, max_iter=100):
    hl_solver, ll_solver = ParseSolversConfig.parse(solvers_config, domain_config)
    domain = ParseDomainConfig.parse(domain_config)
    problem = ParseProblemConfig.parse(problem_config, domain)
    if problem.goal_test():
        return False, "Goal is already satisfied. No planning done."
    n0 = HLSearchNode(hl_solver.translate_problem(problem), domain, problem)

    Q = PriorityQueue()
    Q.put((0, n0))
    for _ in range(max_iter):
        n = Q.get()[1]
        if n.is_hl_node():
            c_plan = n.plan(hl_solver)
            c = LLSearchNode(c_plan, n.concr_prob)
            Q.put((-n.heuristic(), n))
            Q.put((-c.heuristic(), c))
        elif n.is_ll_node():
            n.plan(ll_solver)
            if n.solved():
                return n.curr_plan, None
            if n.gen_child():
                # Expand the node
                fail_step, fail_pred = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred)
                c = HLSearchNode(hl_solver.translate_problem(n_problem), domain, n_problem, prefix=n.curr_plan.prefix(fail_step))
                Q.put((-c.heuristic(), c))
            else:
                # Refine the current node
                Q.put((-n.heuristic(), n))

    return False, "Hit iteration limit, aborting."
