from core.parsing.parse_solvers_config import ParseSolversConfig
from core.parsing.parse_domain_config import ParseDomainConfig
from core.parsing.parse_problem_config import ParseProblemConfig
from core.util_classes.learning import PostLearner
from core.internal_repr.plan import Plan
from Queue import PriorityQueue
from prg_search_node import HLSearchNode, LLSearchNode

"""
Many methods called in p_mod_abs have detailed documentation.
"""
#def p_mod_abs(domain_config, problem_config, solvers_config, suggester = None, max_iter=100, debug = False):
def p_mod_abs(hl_solver, ll_solver, domain, problem, initial=None, goal=None, suggester = None, max_iter=10, debug = False, label=''):
    #hl_solver, ll_solver = ParseSolversConfig.parse(solvers_config, domain_config)
    #domain = ParseDomainConfig.parse(domain_config)
    #problem = ParseProblemConfig.parse(problem_config, domain)
    if problem.goal_test():
        return None, "Goal is already satisfied. No planning done."

    n0 = HLSearchNode(hl_solver.translate_problem(problem, initial, goal), domain, problem, priority=0, label=label)
    Q = PriorityQueue()
    Q.put((n0.heuristic(), n0))
    for _ in range(max_iter):
        n = Q.get()[1]
        if n.is_hl_node():
            c_plan = n.plan(hl_solver)
            if c_plan == Plan.IMPOSSIBLE:
                print('IMPOSSIBLE PLAN IN PR GRAPH')
                if debug: print('Found impossible plan')
                continue
            c = LLSearchNode(c_plan, n.concr_prob, priority=n.priority + 1)
            Q.put((n.heuristic(), n))
            Q.put((c.heuristic(), c))
        elif n.is_ll_node():
            n.plan(ll_solver)
            if n.solved():
                print('SOLVED PR GRAPH')
                return n.curr_plan, None
            Q.put((n.heuristic(), n))
            if n.gen_child():
                # Expand the node
                fail_step, fail_pred = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred, suggester)
                c = HLSearchNode(hl_solver.translate_problem(n_problem, goal=goal), domain, n_problem, priority=n.priority + 1, prefix=n.curr_plan.prefix(fail_step), label=label)
                Q.put((c.heuristic(), c))

        if debug:
            if n.is_hl_node():
                print "Current Iteration: HL Search Node with priority {}".format( n.priority)
            elif n.is_ll_node():
                print "Current Iteration: LL Search Node with priority {}".format( n.priority)
                # print "plan str: {}".format(n.curr_plan.get_plan_str())



    print('PR GRAPH hit iteration limit')
    return None, "Hit iteration limit, aborting."
