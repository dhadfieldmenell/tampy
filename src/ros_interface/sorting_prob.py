from pma.hl_solver import FFSolver
from policy_hooks.load_task_definitions import get_tasks, plan_from_str, fill_params

def get_sorting_problem(plan, color_map):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects blue_target white_target yellow_target green_target"
    for cloth in color_map:
        hl_plan_str += " {0}".format(cloth)
    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(plan)

    hl_plan_str += "(:goal (and"
    for cloth in color_map:
        if color_map[cloth][0] == BLUE:
            hl_plan_str += "(ClothAtLeftTarget {0} blue_target)"
        elif color_map[cloth][0] == WHITE:
            hl_plan_str += "(ClothAtLeftTarget {0} white_target)"
        elif color_map[cloth][0] == YELLOW:
            hl_plan_str += "(ClothAtRightTarget {0} yellow_target)"
        elif color_map[cloth][0] == GREEN:
            hl_plan_str += "(ClothAtRightTarget {0} green_target)"
    hl_plan_str += "))\n"

    hl_plan_str += "\n)"
    return hl_plan_str

def parse_initial_state(plan):
    hl_init_state = "(and "
    blue_target = plan.params['blue_target']
    green_target = plan.params['green_target']
    white_target = plan.params['white_target']
    yellow_target = plan.params['yellow_target']
    for param in plan.params:
        if param._type == "Cloth":
            if param.pose[1,0] > 0:
                hl_init_state += "(ClothInLeftRegion {0})".format(param.name)
            else:
                hl_init_state += "(ClothInRightRegion {0})".format(param.name)

            for target in [blue_target, white_target]:
                if np.all(np.abs(target.value - param.pose[:,0]) < 0.03):
                    hl_init_state += "(ClothAtLeftTarget {0} {1})".format(param.name, target.name)
                else:
                    hl_init_state += "(not (ClothAtLeftTarget {0} {1}))".format(param.name, target.name)
            
            for target in [green_target, yellow_target]:
                if np.all(np.abs(target.value - param.pose[:,0]) < 0.03):
                    hl_init_state += "(ClothAtRightTarget {0} {1})".format(param.name, target.name)
                else:
                    hl_init_state += "(not (ClothAtRightTarget {0} {1}))".format(param.name, target.name)
    
    hl_init_state += ")\n"
    return hl_init_state

def get_hl_plan(prob):
    with open('../domains/laundry_domain/sorting_domain.pddl', 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver()
    return hl_solver._run_planner(domain, prob)

def 
