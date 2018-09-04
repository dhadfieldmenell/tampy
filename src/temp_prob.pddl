(define (problem sorting_problem)
(:domain sorting_domain)
(:objects cloth2 - Cloth cloth3 - Cloth cloth0 - Cloth cloth1 - Cloth cloth1_end_target - RightTarget cloth3_end_target - RightTarget right_mid_target - RightTarget cloth2_end_target - LeftTarget cloth0_end_target - LeftTarget left_mid_target - LeftTarget)
(:init  (ClothInLeftRegion cloth2) (ClothInLeftRegion cloth3) (ClothInLeftRegion cloth0) (ClothInLeftRegion cloth1))
(:goal (and (ClothAtLeftTarget cloth0 cloth0_end_target) (ClothAtRightTarget cloth1 cloth1_end_target) (ClothAtLeftTarget cloth2 cloth2_end_target) (ClothAtRightTarget cloth3 cloth3_end_target)))

)