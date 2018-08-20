(define (problem sorting_problem)
(:domain sorting_domain)
(:objects can0 - Can can1 - Can can1_init_target - Target middle_target - Target can1_end_target - Target can0_init_target - Target can0_end_target - Target)
(:init  (CanAtTarget can0 can0_init_target) (CanAtTarget can1 can1_init_target))
(:goal (and (CanAtTarget can0 can0_end_target) (CanAtTarget can1 can1_end_target)))

)