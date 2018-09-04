(define (domain sorting_domain)
    (:requirements :equality)
    
    (:predicates (ClothInLeftRegion ?cloth - Cloth)
                 (ClothInRightRegion ?cloth - Cloth)
                 (ClothAtLeftTarget ?cloth - Cloth ?target - LeftTarget)
                 (ClothAtRightTarget ?cloth - Cloth ?target - RightTarget)
                 (ClothInLeftGripper ?cloth - Cloth)
                 (ClothInRightGripper ?cloth - Cloth)
                 (ClothInMiddle ?cloth - Cloth)
    )

    (:types Cloth LeftTarget RightTarget)

    (:action move_cloth_to_left_target
        :parameters (?cloth - Cloth ?target - LeftTarget)
        :precondition (and (ClothInLeftRegion ?cloth)
                           (forall (?c - Cloth) (not (ClothAtLeftTarget ?c  ?target))))
        :effect (and (ClothAtLeftTarget ?cloth ?target)
                     (not (ClothInMiddle ?cloth))
                     (not (ClothInRightRegion ?cloth)))
    )

    (:action move_cloth_to_right_target
        :parameters (?cloth - Cloth ?target - RightTarget)
        :precondition (and (ClothInRightRegion ?cloth)
                           (forall (?c - Cloth) (not (ClothAtRightTarget ?c ?target))))
        :effect (and (ClothAtRightTarget ?cloth ?target)
                     (not (ClothInMiddle ?cloth))
                     (not (ClothInLeftRegion ?cloth)))
    )

    (:action move_cloth_to_left_region
        :parameters (?cloth - Cloth)
        :precondition (and (ClothInRightRegion ?cloth)
                           (forall (?c - Cloth) (not (ClothInMiddle ?c))))
        :effect (and (ClothInLeftRegion ?cloth)
                     (forall (?t - LeftTarget) (not (ClothAtLeftTarget ?cloth ?t)))
                     (forall (?t - RightTarget) (not (ClothAtRightTarget ?cloth ?t)))
                     (ClothInMiddle ?cloth))
    )

    (:action move_cloth_to_right_region
        :parameters (?cloth - Cloth)
        :precondition (and (ClothInLeftRegion ?cloth)
                           (forall (?c - Cloth) (not (ClothInMiddle ?c))))
        :effect (and (ClothInRightRegion ?cloth)
                     (forall (?t - LeftTarget) (not (ClothAtLeftTarget ?cloth ?t)))
                     (forall (?t - RightTarget) (not (ClothAtRightTarget ?cloth ?t)))
                     (ClothInMiddle ?cloth))
    )
)
