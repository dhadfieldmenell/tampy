(define (domain sorting_domain)
    (:predicates (ClothInLeftRegion ?cloth)
                 (ClothInRightRegion ?cloth)
                 (ClothAtLeftTarget ?cloth ?target)
                 (ClothAtRightTarget ?cloth ?target)
                 (BasketAtTarget ?target)
    )

    (:action move_cloth_to_left_target
        :parameters (?cloth ?target)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (ClothAtLeftTarget ?cloth ?target)
    )

    (:action move_cloth_to_right_target
        :parameters (?cloth ?target)
        :precondition (ClothInRightRegion ?cloth)
        :effect (ClothAtRightTarget ?cloth ?target)
    )

    (:action move_cloth_to_left_region
        :parameters (?cloth)
        :precondition (ClothInRightRegion ?cloth)
        :effect (ClothInLeftRegion ?cloth)
    )

    (:action move_cloth_to_right_region
        :parameters (?cloth)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (ClothInRightRegion ?cloth)
    )
)
