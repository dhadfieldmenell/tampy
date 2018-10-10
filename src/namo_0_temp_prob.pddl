(define (problem sorting_problem)
(:domain sorting_domain)
(:objects can2 - Can can3 - Can can0 - Can can1 - Can can2_end_target - Target right_target - Target can3_end_target - Target left_target - Target can0_end_target - Target middle_target - Target can1_end_target - Target)
(:init  (CanInReach can2) (CanInReach can3) (NearCan can3) (CanInReach can0) (CanInReach can1))
(:goal (and (CanAtTarget can0 can0_end_target) (CanAtTarget can1 can1_end_target) (CanAtTarget can2 can2_end_target) (CanAtTarget can3 can3_end_target)))

)