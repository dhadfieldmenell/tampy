(define (domain sorting_domain)
    (:predicates (ClothInLeftRegion ?cloth)
                 (ClothInRightRegion ?cloth)
                 (ClothAtBlueTarget ?cloth)
                 (ClothAtGreenTarget ?cloth)
                 (ClothAtWhiteTarget ?cloth)
                 (ClothAtYellowTarget ?cloth)
                 (ClothInCenterRegion ?cloth)
    )

    (:types cloth)

    (:action move_cloth_to_blue_target
        :parameters (?cloth - cloth)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (and (ClothAtBlueTarget ?cloth)
                     (not (ClothInCenterRegion ?cloth)))
    )

    (:action move_cloth_to_green_target
        :parameters (?cloth - cloth)
        :precondition (ClothInLeftRegion ?cloth)
        :effect (and (ClothAtGreenTarget ?cloth)
                     (not (ClothInCenterRegion ?cloth)))
    )

    (:action move_cloth_to_yellow_target
        :parameters (?cloth - cloth)
        :precondition (ClothInRightRegion ?cloth)
        :effect (and (ClothAtYellowTarget ?cloth)
                     (not (ClothInCenterRegion ?cloth)))
    )

    (:action move_cloth_to_white_target
        :parameters (?cloth - cloth)
        :precondition (ClothInRightRegion ?cloth)
        :effect (and (ClothAtWhiteTarget ?cloth)
                     (not (ClothInCenterRegion ?cloth)))
    )

    (:action move_cloth_to_left_region
        :parameters (?cloth - cloth)
        :precondition (and (ClothInRightRegion ?cloth)
                           (forall (?c - cloth) (not (ClothInCenterRegion ?c))))
        :effect (and (ClothInLeftRegion ?cloth)
                     (ClothInCenterRegion ?cloth))
    )

    (:action move_cloth_to_right_region
        :parameters (?cloth - cloth)
        :precondition (and (ClothInLeftRegion ?cloth)
                           (forall (?c - cloth) (not (ClothInCenterRegion ?c))))
        :effect (and (ClothInRightRegion ?cloth)
                     (ClothInCenterRegion ?cloth))
    )
)
