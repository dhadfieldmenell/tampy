(define (problem sorting_problem)
(:domain sorting_domain)
(:objects can0 - Can can1 - Can middle_target - Target left_target - Target can1_end_target - Target right_target - Target can0_end_target - Target)
(:init  (CanInReach can0) (CanInGripper can0) (CanInReach can1) (CanObstructs can1 can0) (CanObstructsTarget can1 can0_end_target))
(:goal (and (CanAtTarget can0 can0_end_target) (CanAtTarget can1 can1_end_target)))

)