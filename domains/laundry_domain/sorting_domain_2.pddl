(define (domain sorting_domain)
    (:predicates (ClothInLeftRegion ?cloth)
                 (ClothInRightRegion ?cloth)
                 (ClothAtBlueTarget ?cloth)
                 (ClothAtGreenTarget ?cloth)
                 (ClothAtWhiteTarget ?cloth)
                 (ClothAtYellowTarget ?cloth)
    )

    (:action move_cloth_to_blue_target
        :parameters (?cloth)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (ClothAtBlueTarget ?cloth)
    )

    (:action move_cloth_to_green_target
        :parameters (?cloth)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (ClothAtGreenTarget ?cloth)
    )

    (:action move_cloth_to_yellow_target
        :parameters (?cloth)
        :precondition (ClothInRightRegion ?cloth)
        :effect (ClothAtYellowTarget ?cloth)
    )

    (:action move_cloth_to_white_target
        :parameters (?cloth)
        :precondition (ClothInRightRegion ?cloth)
        :effect (ClothAtWhiteTarget ?cloth)
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
