from IPython import embed as shell
from core.parse_config_to_problem import ParseConfigToProblem
from parse_config_to_solvers import ParseConfigToSolvers
from Queue import PriorityQueue
from prg_search_node import HLSearchNode, LLSearchNode

"""
Many methods called in p_mod_abs have detailed documentation.
"""
def p_mod_abs(config, max_iter=100):
    problem = ParseConfigToProblem(config).parse()
    if problem.goal_test():
        print "Goal is already satisfied. No planning done."
        return False
    hl_solver, ll_solver = ParseConfigToSolvers(config).parse()
    n0 = HLSearchNode(hl_solver.translate(problem, config), problem)

    Q = PriorityQueue()
    Q.put((0, n0))
    for _ in range(max_iter):
        n = Q.get()[1]
        if n.is_hl_node():
            abs_plan = n.plan(hl_solver)
            c = LLSearchNode(abs_plan, n.concr_prob)
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
                c = HLSearchNode(hl_solver.translate(n_problem, config), n_problem, prefix=n.curr_plan.prefix(fail_step))
                Q.put((-c.heuristic(), c))

    return False
